# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMaxM3KVCacheManagerV2 with sparse KV cache offload, on the native KV cache manager.

Each sparse layer's K and V (with NVFP4, also their block scales) keep their
history in host pages. A decode step fetches the union of the KV heads' picks
into a per-request GPU cache, and the block being written lives in a tail page
per Batch row. These tests write token markers the way the model writes K/V:
each step's token to its row's tail and to the block's host page. Every page a
fetch returns, and every tail, must then hold the markers of the tokens,
layer, buffer and request that wrote it.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._torch.attention.backends.sparse.offload import (
    SparseOffloadState,
    allocate_sparse_offload_state,
)
from tensorrt_llm._torch.attention.backends.sparse.offload_kernels import (
    SPARSE_IN_FLIGHT_COLUMN,
    select_sparse_history_unit_union,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MiniMaxM3SparseAttentionConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

NUM_LAYERS = 5
SPARSE_LAYERS = (3, 4)
NUM_KV_HEADS = 4
HEAD_DIM = 128
TOKENS_PER_BLOCK = 128
TOPK = 16
MAX_BATCH_SIZE = 4
# The union of four heads' history picks: each head also picks the block being written.
SELECTION = NUM_KV_HEADS * (TOPK - 1)
# Two int32 words per (head, token): the token, then who wrote it.
MARKER_WORDS = 2
_DTYPES = {"bf16": DataType.BF16, "fp8": DataType.FP8, "nvfp4": DataType.NVFP4}


def _pretrained_config() -> SimpleNamespace:
    freq = [0 if layer not in SPARSE_LAYERS else 1 for layer in range(NUM_LAYERS)]
    return SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=NUM_LAYERS,
            sparse_attention_config={"sparse_attention_freq": freq},
        )
    )


def _manager(kv_dtype: str, *, offload: bool = True, cache_factor: float = 1.0):
    sparse_config = MiniMaxM3SparseAttentionConfig(
        implementation="msa",
        indexer_kv_dtype="fp8",
        sparse_topk_blocks=TOPK,
        enable_kv_cache_offload=offload,
        kv_cache_offload_config=(
            {"gpu_cache_factor": cache_factor, "host_history_size": 2 << 30} if offload else {}
        ),
    )
    return MiniMaxM3KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=32 * 1024,
            host_cache_size=0,
            event_buffer_max_size=0,
            dtype={"bf16": "bfloat16"}.get(kv_dtype, kv_dtype),
        ),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=16 * 1024,
        max_batch_size=MAX_BATCH_SIZE,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=_DTYPES[kv_dtype],
        sparse_attention_config=sparse_config,
        pretrained_config=_pretrained_config(),
    )


def _request(request_id: int, prompt_len: int, max_new_tokens: int = 512) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=max_new_tokens,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


def _words(address: int, rows: int, page_bytes: int, row_stride_bytes: int) -> torch.Tensor:
    """Return [rows, H, tokens, words] int32 marker views of pages at ``address``."""
    words_per_token = page_bytes // (NUM_KV_HEADS * TOKENS_PER_BLOCK * 4)
    pages = convert_to_torch_tensor(
        TensorWrapper(
            address, torch.int32, (rows, page_bytes // 4), strides=(row_stride_bytes // 4, 1)
        )
    )
    return pages.view(rows, NUM_KV_HEADS, TOKENS_PER_BLOCK, words_per_token)


def _code(request_id: int, layer: int, buffer: int) -> int:
    return (request_id * 64 + layer) * 8 + buffer


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8", "nvfp4"])
def test_offload_keeps_kv_history_in_host_memory_and_index_k_on_gpu(kv_dtype):
    manager = _manager(kv_dtype)
    try:
        assert manager.sparse_layer_ids == list(SPARSE_LAYERS)
        assert manager._sparse_offload_topk == SELECTION
        offloaded = {str(Role.KEY), str(Role.VALUE)}
        if kv_dtype == "nvfp4":
            offloaded |= {str(Role.KEY_BLOCK_SCALE), str(Role.VALUE_BLOCK_SCALE)}
        for layer in manager.kv_cache_manager_py_config.layers:
            roles = {str(buffer.role): buffer for buffer in layer.buffers}
            if int(manager.pp_layers[int(layer.layer_id)]) in SPARSE_LAYERS:
                assert {role for role, buffer in roles.items() if buffer.is_sparse} == offloaded
                assert not roles[str(Role.INDEX_KEY)].is_sparse
                assert layer.sparse_cache.num_selected_blocks == SELECTION
                # A unit is a whole block, and the cache holds one selection plus it.
                assert layer.sparse_cache.max_cached_units == SELECTION + 1
            else:
                assert layer.sparse_cache is None
                assert not any(buffer.is_sparse for buffer in layer.buffers)
        for local_layer, layer in enumerate(manager.pp_layers):
            device_bytes = manager.get_layer_bytes_per_token(local_layer, Role.ALL)
            if layer in SPARSE_LAYERS:
                # Only the FP8 index-K stays on the GPU.
                assert device_bytes == HEAD_DIM
            else:
                assert device_bytes > 0
        descriptors = manager.get_sparse_offload_descriptors()
        assert sorted(descriptors) == list(SPARSE_LAYERS)
        for layer, descriptor in descriptors.items():
            assert descriptor.unit_tokens == TOKENS_PER_BLOCK
            assert {str(buffer.role) for buffer in descriptor.buffer_ids} == offloaded
            # Every sparse buffer is read; none is zero-filled padding.
            assert descriptor.padding_regions == ()
            tail = manager.get_sparse_offload_tail(layer)
            assert list(tail.pages) == [buffer.role for buffer in descriptor.buffer_ids]
            for buffer, page_bytes in zip(descriptor.buffer_ids, tail.page_bytes):
                pages = tail.pages[buffer.role]
                assert pages.shape == (MAX_BATCH_SIZE, page_bytes)
                assert page_bytes == manager.impl.get_page_stride(buffer.layer_id, buffer.role)
    finally:
        manager.shutdown()


def test_offload_scratch_counts_tails_and_pick_columns():
    manager = _manager("fp8")
    try:
        config = manager.kv_cache_manager_py_config
        total = manager.get_sparse_offload_scratch_bytes(config)
        shared = super(MiniMaxM3KVCacheManagerV2, manager).get_sparse_offload_scratch_bytes(config)
        # Each metadata copy: one union column per (KV head, pick) and a tail slot per row.
        assert total - shared == 2 * MAX_BATCH_SIZE * (NUM_KV_HEADS * TOPK + 1) * 4
        state = 2 * SparseOffloadState.device_bytes(
            max_batch_size=MAX_BATCH_SIZE,
            topk=SELECTION,
            buffer_counts={2},
            nvfp4_head_dim=None,
        )
        # One FP8 K and one FP8 V page per Batch row and sparse layer, held once.
        tails = MAX_BATCH_SIZE * len(SPARSE_LAYERS) * 2 * NUM_KV_HEADS * TOKENS_PER_BLOCK * HEAD_DIM
        assert shared == state + tails
    finally:
        manager.shutdown()


def test_without_offload_nothing_is_sparse():
    manager = _manager("fp8", offload=False)
    try:
        assert not manager._enable_kv_cache_offload
        for layer in manager.kv_cache_manager_py_config.layers:
            assert layer.sparse_cache is None
            assert not any(buffer.is_sparse for buffer in layer.buffers)
        assert manager.get_sparse_offload_scratch_bytes(manager.kv_cache_manager_py_config) == 0
    finally:
        manager.shutdown()


class _Markers:
    """Host-page, tail and arena views of every offloaded buffer, as int32 markers."""

    def __init__(self, manager: MiniMaxM3KVCacheManagerV2):
        self.descriptors = manager.get_sparse_offload_descriptors()
        self.host = {}
        self.tails = {}
        self.arena = {}
        for layer, descriptor in self.descriptors.items():
            tail = manager.get_sparse_offload_tail(layer)
            for index, buffer in enumerate(descriptor.buffer_ids):
                page_bytes = tail.page_bytes[index]
                converter = manager.impl.get_page_index_converter(buffer.layer_id, buffer.role)
                slots = (
                    manager.impl.get_page_index_upper_bound(buffer.layer_id, buffer.role)
                    + int(converter.layer_offset)
                ) // int(converter.scale)
                self.host[layer, index] = _words(
                    manager.impl.get_mem_pool_base_address(buffer.layer_id, buffer.role),
                    slots,
                    page_bytes,
                    int(converter.scale) * page_bytes,
                )
                self.tails[layer, index] = _words(
                    tail.pages[buffer.role].data_ptr(), MAX_BATCH_SIZE, page_bytes, page_bytes
                )
                self.arena[layer, index] = _words(
                    descriptor.arena_base_addresses[index],
                    descriptor.arena_row_counts[index],
                    page_bytes,
                    page_bytes,
                )

    def write(self, views, page: int, tokens: torch.Tensor, codes: torch.Tensor) -> None:
        """Mark ``tokens`` of one page; positions are taken within the block."""
        positions = (tokens % TOKENS_PER_BLOCK).long()
        views[page, :, positions, 0] = tokens.to(torch.int32)
        views[page, :, positions, 1] = codes.to(torch.int32)

    def check(self, views, page: int, tokens: torch.Tensor, code: int, what: str) -> None:
        positions = (tokens % TOKENS_PER_BLOCK).long()
        got = views[page, :, positions, :MARKER_WORDS].cpu()
        expected = torch.stack(
            (tokens.to(torch.int32), torch.full_like(tokens, code, dtype=torch.int32)), dim=-1
        ).expand(NUM_KV_HEADS, -1, -1)
        bad = (got != expected).any(dim=-1).any(dim=0)
        assert not bad.any(), f"{what}: tokens {tokens[bad].tolist()} read {got[0, bad].tolist()}"

    def check_blocks(self, views, pages, blocks, codes, what: str) -> None:
        """Check whole blocks at once: page ``pages[i]`` holds block ``blocks[i]``."""
        got = views[torch.tensor(pages, device="cuda").long(), :, :, :MARKER_WORDS].cpu()
        tokens = torch.tensor(blocks)[:, None] * TOKENS_PER_BLOCK + torch.arange(TOKENS_PER_BLOCK)
        expected = (
            torch.stack((tokens, torch.tensor(codes)[:, None].expand_as(tokens)), dim=-1)
            .to(torch.int32)[:, None]
            .expand_as(got)
        )
        bad = (got != expected).any(dim=-1).any(dim=-1).any(dim=-1)
        assert not bad.any(), (
            f"{what}: blocks {torch.tensor(blocks)[bad].tolist()} of codes "
            f"{torch.tensor(codes)[bad].tolist()} hold other tokens"
        )


@pytest.mark.parametrize("kv_dtype", ["fp8", "nvfp4"])
@pytest.mark.parametrize("seed", [0, 1])
def test_fetched_pages_and_tails_hold_written_history_across_evictions(kv_dtype, seed):
    """Decode three requests past several block boundaries with a cache smaller than the history.

    Prompts arrive mid-block, as from a context worker, so the first step seeds the
    tail from the prompt's last host page. Requests join at different steps and
    input runs newest first, so input positions and Batch rows disagree.
    """
    generator = torch.Generator().manual_seed(seed)
    manager = _manager(kv_dtype)
    markers = _Markers(manager)
    num_buffers = len(next(iter(markers.descriptors.values())).buffer_ids)
    # Long enough to outgrow the cache (one selection plus the block being written),
    # each ending mid-block; the last one ends just before a boundary.
    prompts = [70 * TOKENS_PER_BLOCK + 37, 66 * TOKENS_PER_BLOCK + 100, 63 * TOKENS_PER_BLOCK + 127]
    requests = [_request(5 + index, prompt) for index, prompt in enumerate(prompts)]
    join_step = {request.py_request_id: 50 * index for index, request in enumerate(requests)}
    state = allocate_sparse_offload_state(
        lambda name, shape, dtype: torch.empty(shape, dtype=dtype, device="cuda"),
        markers.descriptors,
        max_batch_size=MAX_BATCH_SIZE,
        topk=SELECTION,
    )
    active: list[LlmRequest] = []
    slots: dict[int, list[int]] = {}
    num_steps = 2 * TOKENS_PER_BLOCK + 40
    try:
        for step in range(num_steps):
            for request in requests:
                rid = request.py_request_id
                if join_step[rid] != step:
                    continue
                assert manager.prepare_disagg_gen_init(request)
                cache = manager.kv_cache_map[rid]
                slots[rid] = list(cache.get_base_page_indices(0))
                # The context worker's transfer: every prompt token's K/V in its host page.
                prompt = torch.arange(request.prompt_len, device="cuda")
                for (layer, index), views in markers.host.items():
                    for block in range(prompt.numel() // TOKENS_PER_BLOCK + 1):
                        tokens = prompt[block * TOKENS_PER_BLOCK : (block + 1) * TOKENS_PER_BLOCK]
                        codes = torch.full_like(tokens, _code(rid, layer, index))
                        markers.write(views, slots[rid][block], tokens, codes)
                active.append(request)
            for request in active:
                assert manager.try_allocate_generation(request)
            scheduled = SimpleNamespace(context_requests=[], generation_requests=active[::-1])
            manager.prepare_resources(scheduled)
            request_ids = [request.py_request_id for request in scheduled.generation_requests]
            for rid in request_ids:
                # Capacity grows a page at a time; an allocated page never moves.
                pages = list(manager.kv_cache_map[rid].get_base_page_indices(0))
                for block, slot in enumerate(slots[rid][: len(pages)]):
                    assert slot < 0 or pages[block] == slot, (rid, block)
                slots[rid] = pages
            manager.prepare_sparse_offload(state, request_ids, beam_width=1)
            rows = state.batch_rows.tolist()[: len(request_ids)]
            histories = [manager.kv_cache_map[rid].history_length for rid in request_ids]
            # Each head picks the block being written and up to 15 history blocks.
            head_topk = torch.full((NUM_KV_HEADS, len(request_ids), TOPK), -1, dtype=torch.int32)
            for position, history in enumerate(histories):
                blocks = history // TOKENS_PER_BLOCK
                for head in range(NUM_KV_HEADS):
                    picks = torch.randperm(blocks, generator=generator)[: TOPK - 1]
                    picks = torch.cat((picks, torch.tensor([blocks]))).sort().values
                    head_topk[head, position, : picks.numel()] = picks.to(torch.int32)
            head_topk = head_topk.cuda()
            columns = torch.empty_like(head_topk)
            request_indices = torch.arange(len(request_ids), dtype=torch.int32, device="cuda")
            for layer in SPARSE_LAYERS:
                # The model's write: this step's token to its row's tail and its host page.
                for row, rid, history in zip(rows, request_ids, histories):
                    token = torch.tensor([history], device="cuda")
                    block = history // TOKENS_PER_BLOCK
                    for index in range(num_buffers):
                        code = torch.tensor([_code(rid, layer, index)], device="cuda")
                        markers.write(markers.tails[layer, index], row, token, code)
                        markers.write(markers.host[layer, index], slots[rid][block], token, code)
                writes = manager.prepare_sparse_write_indices(state, layer)
                for row, history in zip(rows, histories):
                    completes = (history + 1) % TOKENS_PER_BLOCK == 0
                    assert (writes[row] >= 0).all().item() == completes
                manager.complete_sparse_offload_tails(layer, writes)
                select_sparse_history_unit_union(
                    head_topk,
                    request_indices,
                    state.batch_rows,
                    state.active_request_count,
                    state.history_lengths,
                    state.selected_history_units,
                    columns,
                    unit_tokens=TOKENS_PER_BLOCK,
                )
                fetched = manager.fetch_sparse_selection(state, layer).cpu()
                picks = head_topk.cpu()
                pick_columns = columns.cpu()
                for index in range(num_buffers):
                    cells, blocks, codes = [], [], []
                    for position, (row, rid, history) in enumerate(
                        zip(rows, request_ids, histories)
                    ):
                        code = _code(rid, layer, index)
                        in_flight = history // TOKENS_PER_BLOCK
                        fetched_blocks = {}
                        for head in range(NUM_KV_HEADS):
                            for k in range(TOPK):
                                block = picks[head, position, k].item()
                                column = pick_columns[head, position, k].item()
                                if block < 0:
                                    assert column == -1
                                elif block == in_flight:
                                    assert column == SPARSE_IN_FLIGHT_COLUMN
                                else:
                                    assert column >= 0, (rid, block, history)
                                    fetched_blocks[block] = fetched[row, index, column].item()
                        assert all(cell >= 0 for cell in fetched_blocks.values())
                        # Distinct blocks occupy distinct cells.
                        assert len(set(fetched_blocks.values())) == len(fetched_blocks)
                        cells += fetched_blocks.values()
                        blocks += fetched_blocks
                        codes += [code] * len(fetched_blocks)
                        markers.check(
                            markers.tails[layer, index],
                            row,
                            torch.arange(in_flight * TOKENS_PER_BLOCK, history + 1),
                            code,
                            f"request {rid} layer {layer} buffer {index} tail",
                        )
                    markers.check_blocks(
                        markers.arena[layer, index],
                        cells,
                        blocks,
                        codes,
                        f"step {step} layer {layer} buffer {index}",
                    )
            manager.update_forward_resources(scheduled, wait_for_input_copy=lambda: None)
            for rid, history in zip(request_ids, histories):
                assert manager.kv_cache_map[rid].history_length == history + 1
        torch.cuda.synchronize()
        # Every block completed during decode reached its host page whole.
        for request in active:
            rid = request.py_request_id
            history = manager.kv_cache_map[rid].history_length
            blocks = list(
                range(request.prompt_len // TOKENS_PER_BLOCK, history // TOKENS_PER_BLOCK)
            )
            assert blocks
            for (layer, index), views in markers.host.items():
                markers.check_blocks(
                    views,
                    [slots[rid][block] for block in blocks],
                    blocks,
                    [_code(rid, layer, index)] * len(blocks),
                    f"request {rid} layer {layer} buffer {index} host",
                )
    finally:
        for request in active:
            manager.free_resources(request)
        manager.shutdown()
