# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generation-phase FMHA for MiniMax-M3, on kernels built for a decode shape.

MsaPrefillFmha's fmha_sm100 kernel schedules a generation row like a context
row, a single query token occupying a 128-row Q tile. This library takes the
generation phase instead and dispatches by layer: a sparse layer to the Triton
block-sparse decode kernel over the blocks the indexer selected, a dense layer
to trtllm-gen over the full page table.

Both need a uniform query length across the generation rows and a geometry they
support, and neither has a fallback: prepare() settles the query length per
step as metadata.msa_decode_span, and ensure_msa_available and
_validate_decode_kernel_support settle the geometry once per run.

Every import of the kernels below is function-local. This module is on the
import path of every attention.backends.trtllm import, and the minimax_m3
package init reaches msa_backend, which subclasses TrtllmAttention, so a
module-scope import here would close a cycle.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch

from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

    from ..sparse.minimax_m3.kernels.trtllm_gen_dense_decode import DenseDecodeWorkspaceLayout


class MsaDecodeFmha(PhasedFmha):
    """MiniMax-M3 generation attention on the Triton and trtllm-gen kernels.

    Generation only: the context phase is MsaPrefillFmha's, and a mixed batch
    is served by the two together through CombinedFmha. run_context is left to
    the base class, which refuses it.
    """

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        # Where this layer's trtllm-gen scratch sits inside the shared
        # attention workspace, settled by prepare_workspace each step.
        self._dense_layout: Optional[DenseDecodeWorkspaceLayout] = None

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        from ..sparse.minimax_m3.kernels.msa_utils import is_msa_layer

        return is_msa_layer(attn)

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: "AttentionForwardArgs",
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        # Every generation row is this library's and every context row
        # MsaPrefillFmha's, whatever the step looks like, so each library
        # claims its one phase and their partition of the phases is total. The
        # phase-less query asks for the whole step, which neither can serve
        # alone: a mixed step belongs to the two together through CombinedFmha.
        return phase is FmhaPhase.GENERATION

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: "AttentionForwardArgs",
        workspace: torch.Tensor,
    ) -> None:
        if forward_args.sparse_runtime_params.sparse_attn_indices is not None:
            # A sparse layer; the Triton kernel takes its split-K scratch from
            # the arena itself, sized by the grid it just chose.
            return
        self._reserve_dense_workspace(q, metadata, workspace)

    def _reserve_dense_workspace(
        self,
        q: torch.Tensor,
        metadata: "TrtllmAttentionMetadata",
        workspace: torch.Tensor,
    ) -> None:
        """Size the shared attention workspace for this layer's trtllm-gen scratch.

        Taking the scratch from the workspace every FMHA library shares is what
        puts it in a captured graph's memory pool. Growing that workspace
        mid-capture would allocate behind the recorded kernels, so it is
        refused.
        """
        from ..sparse.minimax_m3.kernels.trtllm_gen_dense_decode import (
            dense_decode_workspace_layout,
        )

        # The manager has the pool: _validate_decode_kernel_support refused the
        # run without it, so no phase would have reached here.
        kv_pool, _ = metadata.kv_cache_manager.get_kv_subpage_pool(self.attn.layer_idx, "HND")
        q_dtype = torch.float8_e4m3fn if kv_pool.dtype == torch.float8_e4m3fn else q.dtype
        layout = dense_decode_workspace_layout(
            q_dtype=q_dtype,
            num_heads=self.attn.num_heads,
            head_dim=self.attn.head_dim,
            num_kv_heads=int(kv_pool.shape[1]),
            max_num_requests=int(metadata.max_num_requests),
            device=q.device,
        )
        current_bytes = workspace.numel() * workspace.element_size()
        if current_bytes < layout.total_bytes:
            if metadata.is_cuda_graph and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "The attention CUDA graph workspace holds "
                    f"{current_bytes} bytes, fewer than the {layout.total_bytes} "
                    "MiniMax-M3 dense decode needs. The scratch is sized by the "
                    "head geometry alone, so this should not move."
                )
            workspace.resize_((math.ceil(layout.total_bytes / workspace.element_size()),))
        self._dense_layout = layout

    def run_generation(self, params: FmhaParams) -> None:
        from ..sparse.minimax_m3.kernels.msa_utils import write_msa_phase_kv

        metadata = params.meta
        span = metadata.msa_decode_span
        # Both kernels below map a query token to its request through the span,
        # while the phase params carry PhasedFmha's own derivation of the same
        # boundary. Checking them against each other rejects a step whose
        # generation rows were never described and one where the two disagree.
        phase = (params.seq_offset, params.input_seq_length)
        if span != phase:
            raise RuntimeError(
                "MsaDecodeFmha ran on a generation phase its decode span does "
                f"not describe: the span is {span}, while the phase starts at "
                f"row {params.seq_offset} with {params.input_seq_length} query "
                "tokens per request."
            )
        write_msa_phase_kv(
            params.attn,
            params.key_input,
            params.value_input,
            metadata,
            params.fwd.attention_input_type,
            token_offset=params.token_offset,
        )
        row_first = params.seq_offset
        row_last = row_first + metadata.num_generations
        block_table = metadata.msa_block_table[row_first:row_last]
        seq_lens = metadata.msa_seq_lens_cuda[row_first:row_last]

        kv_block_indexes = params.fwd.sparse_runtime_params.sparse_attn_indices
        if kv_block_indexes is not None:
            self._run_sparse(params, kv_block_indexes, block_table, seq_lens)
        else:
            self._run_dense(params, block_table, seq_lens)

    def _run_sparse(
        self,
        params: FmhaParams,
        kv_block_indexes: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        # This kernel also pulls in Triton, a second reason to keep it out of
        # module scope.
        from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

        from ..sparse.minimax_m3.kernels.msa_utils import msa_paged_kv
        from ..sparse.minimax_m3.kernels.triton_sparse_decode import minimax_m3_sparse_attn_decode

        attn = params.attn
        head_dim = attn.head_dim
        num_tokens = params.num_tokens
        manager = params.meta.kv_cache_manager
        # The kernel reads the top-k table head-major and the indexer builds it
        # that way for every step, so this is a view. It reads every stride, so a
        # mixed step's strided suffix works too.
        topk = kv_block_indexes[params.token_offset : params.token_offset + num_tokens].permute(
            1, 0, 2
        )
        offload = None
        if getattr(params.meta, "msa_offload_state", None) is not None and (
            manager.is_sparse_offload_layer(attn.layer_idx)
        ):
            offload, views = self._fetch_offloaded_history(params, topk)
            k_paged, v_paged = views[Role.KEY][0], views[Role.VALUE][0]
        else:
            k_paged, v_paged = msa_paged_kv(manager, attn.layer_idx)
        nvfp4_args = {}
        if getattr(manager, "is_nvfp4_layer", lambda _: False)(attn.layer_idx):
            from .msa_prefill import _aligned_nvfp4_dequant_scales

            if params.fwd.kv_scale_quant_orig is None:
                raise RuntimeError("NVFP4 sparse decode requires dequantization scales")
            k_scale, v_scale = _aligned_nvfp4_dequant_scales(attn, params.fwd.kv_scale_quant_orig)
            if offload is not None:
                k_block_scale = views[Role.KEY_BLOCK_SCALE][0]
                v_block_scale = views[Role.VALUE_BLOCK_SCALE][0]
            else:
                scales = manager.get_block_scale_buffers(attn.layer_idx, "HND")
                k_block_scale, v_block_scale = scales[:, 0], scales[:, 1]
            nvfp4_args = dict(
                k_block_scale=k_block_scale,
                v_block_scale=v_block_scale,
                k_global_scale=k_scale,
                v_global_scale=v_scale,
            )
        # q may still be FP8 from a fused producer; the kernel widens it
        # in-register, so it is passed through as it arrives.
        minimax_m3_sparse_attn_decode(
            params.query_input.view(num_tokens, attn.num_heads, head_dim),
            k_paged,
            v_paged,
            topk,
            block_table,
            seq_lens,
            sm_scale=(head_dim**-0.5) / float(attn.q_scaling),
            output=params.output.view(num_tokens, attn.num_heads, head_dim),
            decode_query_len=params.input_seq_length,
            offload=offload,
            **nvfp4_args,
        )

    def _fetch_offloaded_history(self, params: FmhaParams, topk: torch.Tensor):
        """Fetch the history this layer's heads picked and say where each pick lives.

        With KV cache offload the native cache keeps one selection per request
        and layer, so the step fetches the union of the heads' history picks; each
        pick then finds its block's arena cell through its union column, and a
        pick of the block being written reads its request's tail. Returns the
        kernel's offload inputs and the layer's (arena, tail) page views.
        """
        from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

        from ..sparse.minimax_m3.kernels.triton_sparse_decode import SparseDecodeOffload
        from ..sparse.offload_kernels import select_sparse_history_unit_union

        metadata = params.meta
        manager = metadata.kv_cache_manager
        state = metadata.msa_offload_state
        layer = params.attn.layer_idx
        num_tokens = params.num_tokens
        if params.input_seq_length != 1 or params.seq_offset != 0:
            raise RuntimeError("MiniMax-M3 KV cache offload decodes one token per request")
        columns = metadata.msa_offload_pick_columns[: topk.shape[0], :num_tokens]
        select_sparse_history_unit_union(
            topk,
            metadata.msa_offload_request_indices[:num_tokens],
            state.batch_rows,
            state.active_request_count,
            state.history_lengths,
            state.selected_history_units,
            columns,
            unit_tokens=int(manager.tokens_per_block),
        )
        cells = manager.fetch_sparse_selection(state, layer)
        descriptor = state.layers[layer]
        # A buffer reads its cells at its output slice, which buffers with
        # coinciding arena indices share.
        out_slots = state.out_slots[layer]
        views = manager.get_sparse_offload_page_views(layer)
        nvfp4 = Role.KEY_BLOCK_SCALE in views
        offload = SparseDecodeOffload(
            pick_columns=columns,
            cells=cells,
            rows=state.batch_rows[:num_tokens],
            tail_k=views[Role.KEY][1],
            tail_v=views[Role.VALUE][1],
            cell_columns=(
                out_slots[descriptor.buffer_index(Role.KEY)],
                out_slots[descriptor.buffer_index(Role.VALUE)],
                out_slots[descriptor.buffer_index(Role.KEY_BLOCK_SCALE)] if nvfp4 else -1,
                out_slots[descriptor.buffer_index(Role.VALUE_BLOCK_SCALE)] if nvfp4 else -1,
            ),
            tail_k_block_scale=views[Role.KEY_BLOCK_SCALE][1] if nvfp4 else None,
            tail_v_block_scale=views[Role.VALUE_BLOCK_SCALE][1] if nvfp4 else None,
        )
        return offload, views

    def _run_dense(
        self,
        params: FmhaParams,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        """Attend the full page table on trtllm-gen, for a dense M3 layer.

        This is the kernel FlashInferTrtllmGenFmha runs, called directly rather
        than through that library, which cannot address M3's pool. See the
        trtllm_gen_dense_decode module docstring for why.
        """
        from ..sparse.minimax_m3.kernels.trtllm_gen_dense_decode import (
            minimax_m3_trtllm_gen_dense_decode,
            split_dense_decode_workspace,
        )

        attn = params.attn
        metadata = params.meta
        head_dim = attn.head_dim
        num_tokens = params.num_tokens
        row_first = params.seq_offset
        # The sub-page block table prepare() staged, if it could; the kernel
        # expands its own when the factor does not match this layer's.
        staged_table, staged_factor = metadata.msa_subpage_rows(
            row_first, row_first + metadata.num_generations
        )
        if self._dense_layout is None:
            raise RuntimeError(
                "MiniMax-M3 dense decode ran before prepare_workspace sized its "
                "scratch out of the shared attention workspace."
            )
        workspace, counters = split_dense_decode_workspace(params.workspace, self._dense_layout)
        minimax_m3_trtllm_gen_dense_decode(
            params.query_input.view(num_tokens, attn.num_heads, head_dim),
            metadata.kv_cache_manager,
            attn.layer_idx,
            block_table,
            seq_lens,
            sm_scale=(head_dim**-0.5) / float(attn.q_scaling),
            output=params.output.view(num_tokens, attn.num_heads, head_dim),
            decode_query_len=params.input_seq_length,
            max_seq_len=int(metadata.msa_max_kv_len),
            max_num_requests=int(metadata.max_num_requests),
            staged_subpage_table=staged_table,
            staged_subpages_per_slot=staged_factor,
            workspace=workspace,
            counters=counters,
        )


__all__ = ["MsaDecodeFmha"]
