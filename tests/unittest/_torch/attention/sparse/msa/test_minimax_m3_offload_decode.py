# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 sparse decode reading KV cache offload's arena and tails.

With offload, a sparse layer's history blocks sit in arena cells that fetch
names per buffer, and the block being written sits in its request's tail page.
Given the same pages placed that way, the decode kernel must produce exactly
the output it produces from the resident page table.
"""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.triton_sparse_decode import (
    NVFP4_SF_VEC_SIZE,
    SPARSE_BLOCK_SIZE,
    SparseDecodeOffload,
    minimax_m3_sparse_attn_decode,
)
from tensorrt_llm._torch.attention.backends.sparse.offload_kernels import SPARSE_IN_FLIGHT_COLUMN
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

PAGE = SPARSE_BLOCK_SIZE
HEAD_DIM = 128
TOPK = 16
BATCH_ROWS = 8


def _pages(num_pages: int, num_kv_heads: int, kv_dtype: str, generator) -> dict:
    """Random K/V pages, and NVFP4 block scales, in the HND page layout."""
    shape = (num_pages, num_kv_heads, PAGE, HEAD_DIM)
    if kv_dtype == "nvfp4":
        # Any byte is two valid E2M1 codes; scales stay finite E4M3 near one.
        data = (num_pages, num_kv_heads, PAGE, HEAD_DIM // 2)
        scales = (num_pages, num_kv_heads, PAGE, HEAD_DIM // NVFP4_SF_VEC_SIZE)
        return {
            "k": torch.randint(0, 256, data, generator=generator, device="cuda").to(torch.uint8),
            "v": torch.randint(0, 256, data, generator=generator, device="cuda").to(torch.uint8),
            "ks": torch.randint(0x30, 0x40, scales, generator=generator, device="cuda").to(
                torch.uint8
            ),
            "vs": torch.randint(0x30, 0x40, scales, generator=generator, device="cuda").to(
                torch.uint8
            ),
        }
    dtype = torch.bfloat16 if kv_dtype == "bf16" else torch.float8_e4m3fn
    return {name: torch.randn(shape, generator=generator, device="cuda").to(dtype) for name in "kv"}


def _picks(seq_lens, num_kv_heads: int, generator) -> torch.Tensor:
    """Ascending per-head block picks with -1 padding; the block being written is included."""
    picks = torch.full((num_kv_heads, len(seq_lens), TOPK), -1, dtype=torch.int32)
    for request, seq_len in enumerate(seq_lens):
        blocks = (seq_len + PAGE - 1) // PAGE
        for head in range(num_kv_heads):
            chosen = torch.randperm(blocks - 1, generator=generator)[: TOPK - 1]
            chosen = torch.cat((chosen, torch.tensor([blocks - 1]))).sort().values
            picks[head, request, : chosen.numel()] = chosen.to(torch.int32)
    return picks.cuda()


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8", "nvfp4"])
@pytest.mark.parametrize(
    "seq_lens",
    [[1, 128, 129, 300], [2049, 4097, 8192, 1000, 7]],
    ids=["short", "long"],
)
@pytest.mark.parametrize("num_topk_chunks", [1, 4])
def test_offload_decode_matches_the_resident_page_table(kv_dtype, seq_lens, num_topk_chunks):
    if kv_dtype == "nvfp4" and get_sm_version() not in (100, 103):
        pytest.skip("NVFP4 sparse decode requires SM100/SM103")
    generator = torch.Generator().manual_seed(len(seq_lens) * 7 + num_topk_chunks)
    cuda_generator = torch.Generator(device="cuda").manual_seed(11)
    num_kv_heads, group = 4, 8
    batch = len(seq_lens)
    max_blocks = max((s + PAGE - 1) // PAGE for s in seq_lens)
    pages = _pages(batch * max_blocks, num_kv_heads, kv_dtype, cuda_generator)
    block_table = torch.randperm(batch * max_blocks, generator=generator).to(torch.int32)
    block_table = block_table.view(batch, max_blocks).cuda()
    seq_lens_dev = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    q = torch.randn(batch, num_kv_heads * group, HEAD_DIM, generator=cuda_generator, device="cuda")
    q = q.to(torch.bfloat16)
    picks = _picks(seq_lens, num_kv_heads, generator)
    names = ["k", "v", "ks", "vs"] if kv_dtype == "nvfp4" else ["k", "v"]

    # Place each request's history picks in arena cells and its block being written
    # in its row's tail. Every buffer gets its own cell numbering.
    rows = torch.randperm(BATCH_ROWS, generator=generator)[:batch].to(torch.int32)
    union = num_kv_heads * (TOPK - 1)
    cells = torch.full((BATCH_ROWS, len(names), union), -1, dtype=torch.int32)
    columns = torch.full(picks.shape, -1, dtype=torch.int32)
    arena = {
        name: torch.zeros_like(pages[name][:1]).repeat(batch * union + 3, 1, 1, 1) for name in names
    }
    tails = {name: torch.zeros_like(pages[name][:1]).repeat(BATCH_ROWS, 1, 1, 1) for name in names}
    free = {
        name: torch.randperm(arena[name].shape[0], generator=generator).tolist() for name in names
    }
    picks_cpu = picks.cpu()
    for request, seq_len in enumerate(seq_lens):
        row = rows[request].item()
        in_flight = (seq_len - 1) // PAGE
        history = sorted(
            {b for b in picks_cpu[:, request].flatten().tolist() if 0 <= b < in_flight}
        )
        for column, block in enumerate(history):
            page = block_table[request, block].item()
            for index, name in enumerate(names):
                cell = free[name].pop()
                cells[row, index, column] = cell
                arena[name][cell] = pages[name][page]
        for name in names:
            tails[name][row] = pages[name][block_table[request, in_flight].item()]
        for head in range(num_kv_heads):
            for k, block in enumerate(picks_cpu[head, request].tolist()):
                if block == in_flight:
                    columns[head, request, k] = SPARSE_IN_FLIGHT_COLUMN
                elif block >= 0:
                    columns[head, request, k] = history.index(block)

    def run(**kwargs) -> torch.Tensor:
        out = torch.empty_like(q)
        nvfp4 = {}
        sources = kwargs.pop("sources")
        if kv_dtype == "nvfp4":
            nvfp4 = {
                "k_block_scale": sources["ks"],
                "v_block_scale": sources["vs"],
                "k_global_scale": torch.ones(1, device="cuda"),
                "v_global_scale": torch.ones(1, device="cuda"),
            }
        minimax_m3_sparse_attn_decode(
            q,
            sources["k"],
            sources["v"],
            picks,
            kwargs.pop("block_table"),
            seq_lens_dev,
            sm_scale=HEAD_DIM**-0.5,
            output=out,
            decode_query_len=1,
            num_topk_chunks=num_topk_chunks,
            **nvfp4,
            **kwargs,
        )
        return out

    resident = run(sources=pages, block_table=block_table)
    offload = SparseDecodeOffload(
        pick_columns=columns.cuda(),
        cells=cells.cuda(),
        rows=rows.cuda(),
        tail_k=tails["k"],
        tail_v=tails["v"],
        cell_columns=(0, 1, 2, 3) if kv_dtype == "nvfp4" else (0, 1, -1, -1),
        tail_k_block_scale=tails.get("ks"),
        tail_v_block_scale=tails.get("vs"),
    )
    # The block table must not be read: point it at pages that hold nothing useful.
    offloaded = run(sources=arena, block_table=torch.zeros_like(block_table), offload=offload)
    assert torch.equal(offloaded, resident)


def test_offload_decode_replays_in_a_cuda_graph_with_new_cells():
    generator = torch.Generator().manual_seed(3)
    num_kv_heads, group, seq_len = 2, 8, 700
    blocks = (seq_len + PAGE - 1) // PAGE
    pages = _pages(blocks, num_kv_heads, "bf16", torch.Generator(device="cuda").manual_seed(5))
    q = torch.randn(1, num_kv_heads * group, HEAD_DIM, device="cuda").to(torch.bfloat16)
    picks = torch.full((num_kv_heads, 1, TOPK), -1, dtype=torch.int32, device="cuda")
    picks[:, 0, :blocks] = torch.arange(blocks, dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    union = num_kv_heads * (TOPK - 1)
    arena = {name: torch.zeros_like(pages[name][:1]).repeat(2 * union, 1, 1, 1) for name in "kv"}
    tails = {name: torch.zeros_like(pages[name][:1]).repeat(BATCH_ROWS, 1, 1, 1) for name in "kv"}
    cells = torch.full((BATCH_ROWS, 2, union), -1, dtype=torch.int32, device="cuda")
    columns = torch.full(picks.shape, -1, dtype=torch.int32, device="cuda")
    columns[:, 0, : blocks - 1] = torch.arange(blocks - 1, dtype=torch.int32)
    columns[:, 0, blocks - 1] = SPARSE_IN_FLIGHT_COLUMN
    rows = torch.zeros(1, dtype=torch.int32, device="cuda")
    offload = SparseDecodeOffload(columns, cells, rows, tails["k"], tails["v"], (0, 1, -1, -1))
    out = torch.empty_like(q)

    def decode():
        minimax_m3_sparse_attn_decode(
            q,
            arena["k"],
            arena["v"],
            picks,
            torch.zeros((1, blocks), dtype=torch.int32, device="cuda"),
            seq_lens,
            sm_scale=HEAD_DIM**-0.5,
            output=out,
            decode_query_len=1,
            offload=offload,
        )

    decode()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode()
    expected = torch.empty_like(q)
    block_table = torch.arange(blocks, dtype=torch.int32, device="cuda")[None]
    minimax_m3_sparse_attn_decode(
        q,
        pages["k"],
        pages["v"],
        picks,
        block_table,
        seq_lens,
        sm_scale=HEAD_DIM**-0.5,
        output=expected,
        decode_query_len=1,
    )
    for step in range(3):
        # A new row and new cells each replay, as fetch and Batch membership change.
        row = (step * 3 + 1) % BATCH_ROWS
        order = torch.randperm(2 * union, generator=generator)
        rows.fill_(row)
        for index, name in enumerate("kv"):
            arena[name].zero_()
            for block in range(blocks - 1):
                cell = order[2 * block + index].item()
                cells[row, index, block] = cell
                arena[name][cell] = pages[name][block]
            tails[name].zero_()
            tails[name][row] = pages[name][blocks - 1]
        graph.replay()
        assert torch.equal(out, expected), step
