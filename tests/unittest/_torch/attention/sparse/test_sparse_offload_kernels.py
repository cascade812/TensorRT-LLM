# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the kernels that assemble sparse offload arena rows."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.offload_kernels import (
    SPARSE_IN_FLIGHT_COLUMN,
    complete_sparse_tails,
    copy_pages,
    gather_sparse_batch_rows,
    merge_sparse_arena_indices,
    select_sparse_history_unit_union,
    select_sparse_history_units,
)

_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _cuda_tensor(value, strided: bool = False) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.int32, device="cuda")
    if not strided:
        return tensor
    storage = torch.full(
        tuple(2 * n for n in tensor.shape), 12345, dtype=torch.int32, device="cuda"
    )
    view = storage[tuple(slice(None, None, 2) for _ in tensor.shape)]
    view.copy_(tensor)
    return view


def _merge_reference(
    topk, requests, batch_rows, active, lengths, history, fetched, writes, unit_tokens
):
    out = torch.full(topk.shape, -1, dtype=torch.int32)
    for query, request in enumerate(requests.tolist()):
        if query >= active or not 0 <= request < min(active, lengths.numel()):
            continue
        row = batch_rows[request].item()
        if not 0 <= row < history.numel():
            continue
        history_units = history[row].item() // unit_tokens
        for k, unit in enumerate(topk[query].tolist()):
            if not 0 <= unit < lengths[request].item():
                continue
            if unit < history_units:
                out[query, k] = fetched[row, k]
            elif unit == history_units:
                out[query, k] = writes[row]
    return out


@_requires_cuda
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("active", [0, 2, 3])
def test_merge_takes_fetched_history_and_the_write_row(strided, active):
    # One-token units: a decode step's kv length is its history plus the token
    # being written, and that token's ordinal equals the history. Batch rows are
    # slots: input positions 0, 1 and 2 sit in rows 2, 3 and 0, and row 1 is free.
    history = torch.tensor([7, 0, 12, 5], dtype=torch.int32)
    batch_rows = torch.tensor([2, 3, 0, -1], dtype=torch.int32)
    lengths = torch.tensor([13, 6, 8], dtype=torch.int32)
    requests = torch.tensor([2, 0, 1], dtype=torch.int32)
    topk = torch.tensor(
        [[7, 6, 0, 8, -1], [12, 0, -1, 11, 13], [0, 1, -1, -1, -1]], dtype=torch.int32
    )
    fetched = torch.arange(4 * 5, dtype=torch.int32).reshape(4, 5) + 100
    writes = torch.tensor([40, 41, 42, 43], dtype=torch.int32)
    out = torch.full((3, 5), 999, dtype=torch.int32, device="cuda")
    merge_sparse_arena_indices(
        _cuda_tensor(topk, strided),
        _cuda_tensor(requests, strided),
        _cuda_tensor(batch_rows, strided),
        _cuda_tensor([active]),
        _cuda_tensor(lengths, strided),
        _cuda_tensor(history, strided),
        _cuda_tensor(fetched, strided),
        _cuda_tensor(writes, strided),
        out,
        unit_tokens=1,
    )
    expected = _merge_reference(
        topk, requests, batch_rows, active, lengths, history, fetched, writes, 1
    )
    torch.testing.assert_close(out.cpu(), expected)
    if active == 3:
        # Query 0 is input position 2, in row 0 with history 7: ordinal 7 is the
        # token being written, and older ordinals come from row 0's fetch.
        assert out[0].tolist() == [40, 101, 102, -1, -1]
        # Query 1 is input position 0, in row 2 with history 12; 13 is past its length.
        assert out[1].tolist() == [42, 111, -1, 113, -1]


@_requires_cuda
@pytest.mark.parametrize("topk_size", [1, 7, 2048])
def test_merge_matches_selection_and_fetch_order(topk_size):
    generator = torch.Generator().manual_seed(7)
    # Rows 1 and 3 are free; positions 0, 1 and 2 sit in rows 4, 0 and 2.
    history = torch.tensor([41, 0, 1, 0, 3000], dtype=torch.int32)
    batch_rows = torch.tensor([4, 0, 2, -1, -1], dtype=torch.int32)
    lengths = torch.tensor([3001, 42, 2], dtype=torch.int32)
    requests = torch.tensor([1, 2, 0], dtype=torch.int32)
    topk = torch.stack(
        [torch.randperm(3001, generator=generator)[:topk_size] for _ in range(3)]
    ).to(torch.int32)
    topk[:, 2::7] = -1
    selected = torch.empty((5, topk_size), dtype=torch.int32, device="cuda")
    select_sparse_history_units(
        _cuda_tensor(topk),
        _cuda_tensor(requests),
        _cuda_tensor(batch_rows),
        _cuda_tensor([3]),
        _cuda_tensor(lengths),
        _cuda_tensor(history),
        selected,
        unit_tokens=1,
    )
    assert (selected[[1, 3]] == -1).all(), "Free rows select nothing"
    # A fetch returns an arena row for every selected history unit, in slot order.
    fetched = torch.where(selected >= 0, selected * 3 + 1, -1)
    writes = _cuda_tensor([5, 6, 7, 8, 9])
    out = torch.empty((3, topk_size), dtype=torch.int32, device="cuda")
    merge_sparse_arena_indices(
        _cuda_tensor(topk),
        _cuda_tensor(requests),
        _cuda_tensor(batch_rows),
        _cuda_tensor([3]),
        _cuda_tensor(lengths),
        _cuda_tensor(history),
        fetched,
        writes,
        out,
        unit_tokens=1,
    )
    expected = _merge_reference(
        topk, requests, batch_rows, 3, lengths, history, fetched.cpu(), writes.cpu(), 1
    )
    torch.testing.assert_close(out.cpu(), expected)


@_requires_cuda
def test_merge_replays_in_a_cuda_graph_with_new_watermarks_and_rows():
    topk = _cuda_tensor([[0, 1, 2, 3], [0, 1, -1, -1]])
    requests = _cuda_tensor([0, 1])
    batch_rows = _cuda_tensor([0, 1, -1])
    active = _cuda_tensor([2])
    history = _cuda_tensor([2, 1, 0])
    lengths = _cuda_tensor([3, 2])
    fetched = _cuda_tensor([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
    writes = _cuda_tensor([7, 8, 9])
    out = torch.empty((2, 4), dtype=torch.int32, device="cuda")
    args = (topk, requests, batch_rows, active, lengths, history, fetched, writes, out)
    merge_sparse_arena_indices(*args, unit_tokens=1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        merge_sparse_arena_indices(*args, unit_tokens=1)
    for row0, hist0, length0, active_count, expected0 in (
        (0, 2, 3, 2, [10, 11, 7, -1]),
        (0, 3, 4, 2, [10, 11, 12, 7]),
        (0, 0, 1, 2, [7, -1, -1, -1]),
        (0, 3, 4, 0, [-1, -1, -1, -1]),
        # The request moved to row 2 between steps: the replay follows the map.
        (2, 1, 3, 2, [30, 9, -1, -1]),
        (-1, 1, 3, 2, [-1, -1, -1, -1]),
    ):
        batch_rows[0] = row0
        history[max(row0, 0)] = hist0
        lengths[0] = length0
        active[0] = active_count
        graph.replay()
        torch.testing.assert_close(out[0].cpu(), torch.tensor(expected0, dtype=torch.int32))


@_requires_cuda
@pytest.mark.parametrize(
    "invalid", ["dtype", "width", "requests", "batch_rows", "writes", "lengths", "unit"]
)
def test_merge_rejects_invalid_layout(invalid):
    topk = _cuda_tensor([[0, 1], [1, 0]])
    requests, batch_rows, active, lengths, history = (
        _cuda_tensor([0, 1]),
        _cuda_tensor([0, 1, -1, -1]),
        _cuda_tensor([2]),
        _cuda_tensor([2, 2]),
        _cuda_tensor([1] * 4),
    )
    fetched = _cuda_tensor([[0, 0]] * 4)
    writes = _cuda_tensor([0] * 4)
    unit = 1
    if invalid == "dtype":
        topk = topk.to(torch.int64)
    elif invalid == "width":
        fetched = _cuda_tensor([[0]] * 4)
    elif invalid == "requests":
        requests = requests[:1]
    elif invalid == "batch_rows":
        batch_rows = batch_rows[:2]
    elif invalid == "writes":
        writes = writes[:2]
    elif invalid == "lengths":
        lengths = lengths[:1]
    else:
        unit = 0
    with pytest.raises(ValueError):
        merge_sparse_arena_indices(
            topk,
            requests,
            batch_rows,
            active,
            lengths,
            history,
            fetched,
            writes,
            torch.empty((2, 2), dtype=torch.int32, device="cuda"),
            unit_tokens=unit,
        )


@_requires_cuda
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("count", [1, 3, 300])
def test_gather_reads_each_request_from_its_batch_row(strided, count):
    generator = torch.Generator().manual_seed(count)
    num_rows = max(2 * count, 4)
    values = torch.randint(0, 1 << 20, (num_rows,), generator=generator, dtype=torch.int32)
    rows = torch.randperm(num_rows, generator=generator)[:count].to(torch.int32)
    rows[::3] = -1
    columns = _cuda_tensor(torch.stack((values + 1, values), dim=1), strided)
    out = _cuda_tensor(torch.full((count,), 999, dtype=torch.int32), strided)
    result = gather_sparse_batch_rows(columns[:, 1], _cuda_tensor(rows, strided), out)
    assert result is out
    expected = torch.where(rows >= 0, values[rows.clamp(min=0).long()], -1)
    torch.testing.assert_close(out.cpu(), expected)
    allocated = gather_sparse_batch_rows(columns[:, 1], _cuda_tensor(rows))
    assert allocated.is_contiguous() and allocated.dtype == torch.int32
    torch.testing.assert_close(allocated.cpu(), expected)


@_requires_cuda
def test_gather_replays_with_new_rows_and_ignores_rows_past_the_batch():
    values = _cuda_tensor([10, 11, 12])
    rows = _cuda_tensor([2, 0])
    out = torch.empty(2, dtype=torch.int32, device="cuda")
    gather_sparse_batch_rows(values, rows, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gather_sparse_batch_rows(values, rows, out)
    for new_rows, expected in (([2, 0], [12, 10]), ([1, -1], [11, -1]), ([3, 2], [-1, 12])):
        rows.copy_(_cuda_tensor(new_rows))
        graph.replay()
        assert out.tolist() == expected


@_requires_cuda
@pytest.mark.parametrize("invalid", ["dtype", "ndim", "out"])
def test_gather_rejects_invalid_layout(invalid):
    values, rows, out = _cuda_tensor([1, 2]), _cuda_tensor([0, 1]), None
    if invalid == "dtype":
        values = values.to(torch.int64)
    elif invalid == "ndim":
        values = values.view(1, 2)
    else:
        out = torch.empty(3, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError):
        gather_sparse_batch_rows(values, rows, out)


def _union_reference(head_topk, requests, batch_rows, active, history, width, unit_tokens):
    num_rows = history.numel()
    out = torch.full((num_rows, width), -1, dtype=torch.int32)
    columns = torch.full(head_topk.shape, -1, dtype=torch.int32)
    for query, request in enumerate(requests.tolist()):
        if query >= active or not 0 <= request < active:
            continue
        row = batch_rows[request].item()
        if not 0 <= row < num_rows:
            continue
        history_units = history[row].item() // unit_tokens
        picks = head_topk[:, query, :]
        union = sorted({unit for unit in picks.flatten().tolist() if 0 <= unit < history_units})
        out[row, : min(width, len(union))] = torch.tensor(union[:width], dtype=torch.int32)
        for head in range(picks.shape[0]):
            for k, unit in enumerate(picks[head].tolist()):
                if 0 <= unit < history_units and union.index(unit) < width:
                    columns[head, query, k] = union.index(unit)
                elif unit == history_units:
                    columns[head, query, k] = SPARSE_IN_FLIGHT_COLUMN
    return out, columns


@_requires_cuda
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("active", [0, 2, 3])
def test_union_sorts_distinct_history_picks_into_batch_rows(strided, active):
    num_rows, num_heads, topk, unit_tokens = 5, 4, 6, 128
    history = torch.tensor([7 * 128 + 5, 0, 300, 12 * 128, 99], dtype=torch.int32)
    batch_rows = torch.tensor([3, 0, 2, -1, -1], dtype=torch.int32)
    # Heads overlap and repeat units; each names the in-flight block of its row too.
    head_topk = torch.full((num_heads, 3, topk), -1, dtype=torch.int32)
    head_topk[:, 0, :] = torch.tensor(
        [
            [0, 3, 5, 12, -1, -1],
            [3, 4, 11, 12, -1, -1],
            [1, 11, 12, -1, -1, -1],
            [0, 1, 2, 12, 12, -1],
        ]
    )
    head_topk[:, 1, :3] = torch.tensor([[0, 7, 6], [7, 2, 6], [6, -1, -1], [7, 1, 3]])
    head_topk[:, 2, :2] = torch.tensor([[0, 1], [2, 1], [0, 2], [1, 2]])
    requests = torch.tensor([0, 1, 2], dtype=torch.int32)
    width = num_heads * (topk - 1)
    out = torch.full((num_rows, width), 777, dtype=torch.int32, device="cuda")
    columns = torch.full(head_topk.shape, 777, dtype=torch.int32, device="cuda")
    select_sparse_history_unit_union(
        _cuda_tensor(head_topk, strided),
        _cuda_tensor(requests),
        _cuda_tensor(batch_rows),
        _cuda_tensor([active]),
        _cuda_tensor(history),
        out,
        columns,
        unit_tokens=unit_tokens,
    )
    expected_out, expected_columns = _union_reference(
        head_topk, requests, batch_rows, active, history, width, unit_tokens
    )
    assert torch.equal(out.cpu(), expected_out)
    assert torch.equal(columns.cpu(), expected_columns)


@_requires_cuda
def test_union_matches_reference_for_random_picks():
    generator = torch.Generator().manual_seed(0)
    num_rows, num_heads, topk, unit_tokens, queries = 16, 4, 16, 128, 11
    history = torch.randint(0, 64 * 128, (num_rows,), generator=generator, dtype=torch.int32)
    batch_rows = torch.full((num_rows,), -1, dtype=torch.int32)
    batch_rows[:queries] = torch.randperm(num_rows, generator=generator)[:queries].to(torch.int32)
    head_topk = torch.full((num_heads, queries, topk), -1, dtype=torch.int32)
    for query in range(queries):
        units = history[batch_rows[query]].item() // unit_tokens + 1
        for head in range(num_heads):
            picks = torch.randperm(units, generator=generator)[:topk].to(torch.int32)
            head_topk[head, query, : picks.numel()] = picks.sort().values
    requests = torch.arange(queries, dtype=torch.int32)
    width = num_heads * (topk - 1)
    out = torch.empty((num_rows, width), dtype=torch.int32, device="cuda")
    columns = torch.empty(head_topk.shape, dtype=torch.int32, device="cuda")
    select_sparse_history_unit_union(
        head_topk.cuda(),
        requests.cuda(),
        batch_rows.cuda(),
        _cuda_tensor([queries]),
        history.cuda(),
        out,
        columns,
        unit_tokens=unit_tokens,
    )
    expected_out, expected_columns = _union_reference(
        head_topk, requests, batch_rows, queries, history, width, unit_tokens
    )
    assert torch.equal(out.cpu(), expected_out)
    assert torch.equal(columns.cpu(), expected_columns)


@_requires_cuda
def test_union_replays_in_a_cuda_graph_with_new_picks_and_rows():
    num_rows, num_heads, topk = 4, 2, 4
    head_topk = torch.full((num_heads, 2, topk), -1, dtype=torch.int32, device="cuda")
    requests = torch.arange(2, dtype=torch.int32, device="cuda")
    batch_rows = torch.full((num_rows,), -1, dtype=torch.int32, device="cuda")
    active = torch.zeros(1, dtype=torch.int32, device="cuda")
    history = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    out = torch.empty((num_rows, num_heads * (topk - 1)), dtype=torch.int32, device="cuda")
    columns = torch.empty(head_topk.shape, dtype=torch.int32, device="cuda")
    args = (head_topk, requests, batch_rows, active, history, out, columns)
    select_sparse_history_unit_union(*args, unit_tokens=128)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select_sparse_history_unit_union(*args, unit_tokens=128)
    for step in range(3):
        picks = torch.tensor([[0, step, 4, -1], [step + 1, 4, -1, -1]], dtype=torch.int32)
        head_topk[:, 0, :].copy_(picks)
        head_topk[:, 1, :].copy_(picks.flip(0))
        batch_rows.copy_(torch.tensor([step % num_rows, (step + 1) % num_rows, -1, -1]))
        active.fill_(2)
        history.fill_(4 * 128 + 3)
        graph.replay()
        expected_out, expected_columns = _union_reference(
            head_topk.cpu(),
            requests.cpu(),
            batch_rows.cpu(),
            2,
            history.cpu(),
            out.shape[1],
            128,
        )
        assert torch.equal(out.cpu(), expected_out)
        assert torch.equal(columns.cpu(), expected_columns)


@_requires_cuda
def test_copy_pages_moves_each_page_between_any_two_addresses():
    source = torch.arange(3 * 4096, dtype=torch.int32, device="cuda").view(3, 4096)
    pinned = torch.zeros((2, 1024), dtype=torch.int32).pin_memory()
    target = torch.zeros((2, 4096), dtype=torch.int32, device="cuda")
    # Device to device, then device to mapped host memory and back.
    copy_pages(
        torch.tensor(
            [source[2].data_ptr(), source[0].data_ptr(), source[1].data_ptr()],
            dtype=torch.int64,
            device="cuda",
        ),
        torch.tensor(
            [target[0].data_ptr(), target[1].data_ptr(), pinned[1].data_ptr()],
            dtype=torch.int64,
            device="cuda",
        ),
        torch.tensor([4096 * 4, 2048 * 4, 1024 * 4], dtype=torch.int64, device="cuda"),
    )
    torch.cuda.synchronize()
    assert torch.equal(target[0], source[2])
    assert torch.equal(target[1, :2048], source[0, :2048])
    assert (target[1, 2048:] == 0).all()
    assert torch.equal(pinned[1], source[1, :1024].cpu())
    assert (pinned[0] == 0).all()


@_requires_cuda
def test_complete_sparse_tails_copies_only_rows_with_a_unit_in_flight():
    rows, page_words = 4, 64
    tails = [
        torch.arange(rows * page_words, dtype=torch.int64, device="cuda").view(rows, page_words)
        + 1000 * buffer
        for buffer in range(2)
    ]
    arenas = [torch.zeros((10, page_words), dtype=torch.int64, device="cuda") for _ in range(2)]
    writes = torch.tensor([[3, 7], [-1, -1], [0, 2], [-1, -1]], dtype=torch.int32, device="cuda")

    def array(values):
        return torch.tensor(values, dtype=torch.int64, device="cuda")

    page_bytes = page_words * 8
    complete_sparse_tails(
        writes,
        array([tail.data_ptr() for tail in tails]),
        array([page_bytes, page_bytes]),
        array([arena.data_ptr() for arena in arenas]),
        array([page_bytes, page_bytes]),
        array([0, 1]),
        array([page_words, page_words]),
    )
    torch.cuda.synchronize()
    for buffer in range(2):
        expected = torch.zeros_like(arenas[buffer])
        for row, cells in enumerate(writes.tolist()):
            if cells[buffer] >= 0:
                expected[cells[buffer]] = tails[buffer][row]
        assert torch.equal(arenas[buffer], expected)
