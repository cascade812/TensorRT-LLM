# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernels shared by sparse attention models that offload KV history.

A decode step's selection names units of a request's history; the KV cache
manager fetches them into per-request arena rows. These kernels mask a
selection to the history the Batch published, and assemble the arena rows the
attention reads from the fetch output and the in-flight unit's write row.

Model inputs follow the requests' input positions, while the native Batch keeps
each request in a fixed row, its slot, for as long as the request is scheduled.
The kernels read input-ordered tensors by position and Batch-owned tensors by
row, through a per-step map from each request's position to its Batch row.

Models whose KV heads select separately fetch, per request, the union of the
heads' history picks, since the native cache keeps one resident set per
request and layer.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# A pick's union column when it names the unit being written this step, which
# no fetch returns.
SPARSE_IN_FLIGHT_COLUMN = -2


def check_sparse_offload_tensor(
    tensor: torch.Tensor, name: str, ndim: int, device: torch.device
) -> None:
    """Require a CUDA int32 tensor of ``ndim`` dimensions on ``device``."""
    if tensor.dtype != torch.int32 or tensor.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D int32 tensor")
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must be on CUDA device {device}")


@triton.jit(do_not_specialize=["num_lengths"])
def _select_sparse_history_units_kernel(
    topk_ptr,
    request_indices_ptr,
    batch_rows_ptr,
    active_request_count_ptr,
    unit_lengths_ptr,
    history_lengths_ptr,
    out_ptr,
    topk_stride0,
    topk_stride1,
    request_indices_stride,
    batch_rows_stride,
    unit_lengths_stride,
    history_lengths_stride,
    out_stride0,
    out_stride1,
    NUM_ROWS: tl.constexpr,
    num_lengths,
    TOPK: tl.constexpr,
    UNIT_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query = tl.program_id(0)
    active_count = tl.load(active_request_count_ptr)
    request = tl.load(request_indices_ptr + query * request_indices_stride)
    active = (query < active_count) & (request >= 0) & (request < active_count)
    active = active & (request < NUM_ROWS) & (request < num_lengths)
    row = tl.load(batch_rows_ptr + request * batch_rows_stride, active, other=-1)
    active = active & (row >= 0) & (row < NUM_ROWS)
    length = tl.load(unit_lengths_ptr + request * unit_lengths_stride, active, other=0)
    history = tl.load(history_lengths_ptr + row * history_lengths_stride, active, other=0)
    k = tl.arange(0, BLOCK_K)
    unit = tl.load(topk_ptr + query * topk_stride0 + k * topk_stride1, k < TOPK, other=-1)
    valid = (unit >= 0) & (unit < length) & (unit < history // UNIT_TOKENS)
    # Each live query owns a distinct Batch row. Keep top-k positions, including
    # holes left by in-flight units, so fetch output has the same slot meaning.
    tl.store(
        out_ptr + row * out_stride0 + k * out_stride1,
        tl.where(valid, unit, -1),
        mask=active & (k < TOPK),
    )


def select_sparse_history_units(
    topk_indices: torch.Tensor,
    request_indices: torch.Tensor,
    batch_rows: torch.Tensor,
    active_request_count: torch.Tensor,
    unit_lengths: torch.Tensor,
    history_lengths: torch.Tensor,
    out: torch.Tensor,
    *,
    unit_tokens: int,
) -> None:
    """Mask non-history selections and scatter them into native Batch rows.

    All tensors are CUDA int32 on one device. ``topk_indices`` is [Q, K] in
    selection units, with distinct nonnegative selections per live row.
    ``request_indices`` [Q] gives each query's request by input position, one
    query per request. ``batch_rows`` [B] gives each request's native Batch row,
    or -1 for a position without a live request; live requests have distinct
    rows, in any order and with free rows between them. ``active_request_count``
    [1] excludes graph padding. ``unit_lengths`` [R] counts each request's
    selectable units by input position, and may cover fewer requests than the
    native Batch capacity B. ``history_lengths`` [B] is the Batch-owned,
    published history watermark in original tokens.

    ``out`` is caller-owned [B, K], disjoint from the inputs. Only ordinals below
    both the unit length and history_lengths // unit_tokens are fetched.
    Top-k order and padding positions are preserved; in-flight units and invalid
    selections become -1. Batch rows without a live query are overwritten with
    -1 on every call, including when a graph's Q is smaller than B. No tensor
    allocation, host read, or synchronization is performed.
    """
    device = out.device
    for tensor, name, ndim in (
        (topk_indices, "topk_indices", 2),
        (request_indices, "request_indices", 1),
        (batch_rows, "batch_rows", 1),
        (active_request_count, "active_request_count", 1),
        (unit_lengths, "unit_lengths", 1),
        (history_lengths, "history_lengths", 1),
        (out, "out", 2),
    ):
        check_sparse_offload_tensor(tensor, name, ndim, device)
    num_rows, width = out.shape
    num_queries, topk = topk_indices.shape
    if unit_tokens <= 0:
        raise ValueError("unit_tokens must be positive")
    if topk != width:
        raise ValueError("top-k width must match the configured sparse selection width")
    if request_indices.shape != (num_queries,) or active_request_count.shape != (1,):
        raise ValueError("request_indices must be [Q] and active_request_count must be [1]")
    if history_lengths.shape != (num_rows,) or batch_rows.shape != (num_rows,):
        raise ValueError("history_lengths and batch_rows must cover the native Batch capacity")
    if not num_queries <= unit_lengths.numel() <= num_rows:
        raise ValueError("Query and unit-length rows must fit the native Batch capacity")

    with torch.cuda.device(device):
        # Clear separately: padded queries can carry stale or invalid request IDs,
        # and a smaller graph may leave native Batch rows without any query.
        out.fill_(-1)
        if num_queries == 0 or topk == 0:
            return
        _select_sparse_history_units_kernel[(num_queries,)](
            topk_indices,
            request_indices,
            batch_rows,
            active_request_count,
            unit_lengths,
            history_lengths,
            out,
            *topk_indices.stride(),
            request_indices.stride(0),
            batch_rows.stride(0),
            unit_lengths.stride(0),
            history_lengths.stride(0),
            *out.stride(),
            NUM_ROWS=num_rows,
            num_lengths=unit_lengths.numel(),
            TOPK=topk,
            UNIT_TOKENS=unit_tokens,
            BLOCK_K=triton.next_power_of_2(topk),
        )


@triton.jit
def _select_sparse_history_unit_union_kernel(
    topk_ptr,
    request_indices_ptr,
    batch_rows_ptr,
    active_request_count_ptr,
    history_lengths_ptr,
    out_ptr,
    columns_ptr,
    topk_stride_head,
    topk_stride_query,
    topk_stride_k,
    request_indices_stride,
    batch_rows_stride,
    history_lengths_stride,
    out_stride0,
    out_stride1,
    columns_stride_head,
    columns_stride_query,
    columns_stride_k,
    NUM_ROWS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    UNIT_TOKENS: tl.constexpr,
    IN_FLIGHT: tl.constexpr,
    BLOCK_PICKS: tl.constexpr,
):
    query = tl.program_id(0)
    active_count = tl.load(active_request_count_ptr)
    request = tl.load(request_indices_ptr + query * request_indices_stride)
    live = (query < active_count) & (request >= 0) & (request < active_count)
    live = live & (request < NUM_ROWS)
    row = tl.load(batch_rows_ptr + request * batch_rows_stride, live, other=-1)
    live = live & (row >= 0) & (row < NUM_ROWS)
    history_units = (
        tl.load(history_lengths_ptr + row * history_lengths_stride, live, other=0) // UNIT_TOKENS
    )

    # Every head's picks, flattened head-major.
    pick = tl.arange(0, BLOCK_PICKS)
    in_range = pick < NUM_HEADS * TOPK
    head = pick // TOPK
    k = pick % TOPK
    unit = tl.load(
        topk_ptr + head * topk_stride_head + query * topk_stride_query + k * topk_stride_k,
        in_range,
        other=-1,
    )
    history = live & in_range & (unit >= 0) & (unit < history_units)

    # A pick is the first of its unit when no earlier pick names the same unit.
    # Its column is the number of distinct history units below it, so the union
    # comes out sorted.
    unit_i = unit[:, None]
    unit_j = unit[None, :]
    history_j = history[None, :]
    earlier_same = (unit_j == unit_i) & history_j & (pick[None, :] < pick[:, None])
    first = history & (tl.sum(earlier_same.to(tl.int32), axis=1) == 0)
    smaller_first = (unit_j < unit_i) & first[None, :]
    column = tl.sum(smaller_first.to(tl.int32), axis=1)
    fits = history & (column < WIDTH)

    tl.store(out_ptr + row * out_stride0 + column * out_stride1, unit, mask=first & fits)
    in_flight = live & in_range & (unit == history_units)
    tl.store(
        columns_ptr
        + head * columns_stride_head
        + query * columns_stride_query
        + k * columns_stride_k,
        tl.where(fits, column, tl.where(in_flight, IN_FLIGHT, -1)),
        mask=in_range,
    )


def select_sparse_history_unit_union(
    head_topk_indices: torch.Tensor,
    request_indices: torch.Tensor,
    batch_rows: torch.Tensor,
    active_request_count: torch.Tensor,
    history_lengths: torch.Tensor,
    out: torch.Tensor,
    pick_columns: torch.Tensor,
    *,
    unit_tokens: int,
) -> None:
    """Union each request's per-head history picks into one Batch-row selection.

    ``head_topk_indices`` is [H, Q, K]: each KV head's K unit picks for each
    query, -1 padded, possibly a strided view. The other inputs follow
    :func:`select_sparse_history_units`; one query per request.

    ``out`` [B, S] receives each live request's distinct history picks in
    ascending order at its Batch row, -1 padded; rows without a live query hold
    -1. ``pick_columns`` [H, Q, K] receives each pick's column in its row of
    ``out``, ``SPARSE_IN_FLIGHT_COLUMN`` for a pick of the unit being written,
    and -1 for padding, units past the history and padding queries. S must hold
    the largest union; a pick past it gets -1. No tensor allocation, host read,
    or synchronization is performed.
    """
    device = out.device
    for tensor, name, ndim in (
        (head_topk_indices, "head_topk_indices", 3),
        (request_indices, "request_indices", 1),
        (batch_rows, "batch_rows", 1),
        (active_request_count, "active_request_count", 1),
        (history_lengths, "history_lengths", 1),
        (out, "out", 2),
        (pick_columns, "pick_columns", 3),
    ):
        check_sparse_offload_tensor(tensor, name, ndim, device)
    num_rows, width = out.shape
    num_heads, num_queries, topk = head_topk_indices.shape
    if unit_tokens <= 0:
        raise ValueError("unit_tokens must be positive")
    if pick_columns.shape != head_topk_indices.shape:
        raise ValueError("pick_columns must match the per-head picks")
    if request_indices.shape != (num_queries,) or active_request_count.shape != (1,):
        raise ValueError("request_indices must be [Q] and active_request_count must be [1]")
    if history_lengths.shape != (num_rows,) or batch_rows.shape != (num_rows,):
        raise ValueError("history_lengths and batch_rows must cover the native Batch capacity")
    if num_queries > num_rows:
        raise ValueError("Query rows must fit the native Batch capacity")

    with torch.cuda.device(device):
        # Rows without a query this step, and the columns a union leaves free,
        # must read as padding for the fetch.
        out.fill_(-1)
        if num_queries == 0 or num_heads == 0 or topk == 0:
            return
        _select_sparse_history_unit_union_kernel[(num_queries,)](
            head_topk_indices,
            request_indices,
            batch_rows,
            active_request_count,
            history_lengths,
            out,
            pick_columns,
            *head_topk_indices.stride(),
            request_indices.stride(0),
            batch_rows.stride(0),
            history_lengths.stride(0),
            *out.stride(),
            *pick_columns.stride(),
            NUM_ROWS=num_rows,
            NUM_HEADS=num_heads,
            TOPK=topk,
            WIDTH=width,
            UNIT_TOKENS=unit_tokens,
            IN_FLIGHT=SPARSE_IN_FLIGHT_COLUMN,
            BLOCK_PICKS=triton.next_power_of_2(num_heads * topk),
        )


@triton.jit(do_not_specialize=["num_lengths"])
def _merge_sparse_arena_indices_kernel(
    topk_ptr,
    request_indices_ptr,
    batch_rows_ptr,
    active_request_count_ptr,
    unit_lengths_ptr,
    history_lengths_ptr,
    fetched_ptr,
    write_ptr,
    out_ptr,
    topk_stride0,
    topk_stride1,
    request_indices_stride,
    batch_rows_stride,
    unit_lengths_stride,
    history_lengths_stride,
    fetched_stride0,
    fetched_stride1,
    write_stride,
    out_stride0,
    out_stride1,
    NUM_ROWS: tl.constexpr,
    num_lengths,
    TOPK: tl.constexpr,
    UNIT_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query = tl.program_id(0)
    active_count = tl.load(active_request_count_ptr)
    request = tl.load(request_indices_ptr + query * request_indices_stride)
    live = (query < active_count) & (request >= 0) & (request < active_count)
    live = live & (request < NUM_ROWS) & (request < num_lengths)
    row = tl.load(batch_rows_ptr + request * batch_rows_stride, live, other=-1)
    live = live & (row >= 0) & (row < NUM_ROWS)
    length = tl.load(unit_lengths_ptr + request * unit_lengths_stride, live, other=0)
    history_units = (
        tl.load(history_lengths_ptr + row * history_lengths_stride, live, other=0) // UNIT_TOKENS
    )
    k = tl.arange(0, BLOCK_K)
    in_width = k < TOPK
    unit = tl.load(topk_ptr + query * topk_stride0 + k * topk_stride1, in_width, other=-1)
    valid = live & (unit >= 0) & (unit < length)
    fetched = tl.load(
        fetched_ptr + row * fetched_stride0 + k * fetched_stride1,
        valid & in_width & (unit < history_units),
        other=-1,
    )
    write = tl.load(write_ptr + row * write_stride, live, other=-1)
    tl.store(
        out_ptr + query * out_stride0 + k * out_stride1,
        tl.where(valid & (unit == history_units), write, fetched),
        mask=in_width,
    )


def merge_sparse_arena_indices(
    topk_indices: torch.Tensor,
    request_indices: torch.Tensor,
    batch_rows: torch.Tensor,
    active_request_count: torch.Tensor,
    unit_lengths: torch.Tensor,
    history_lengths: torch.Tensor,
    fetched_indices: torch.Tensor,
    write_indices: torch.Tensor,
    out: torch.Tensor,
    *,
    unit_tokens: int,
) -> torch.Tensor:
    """Return each query's selection as rows of one sparse buffer's arena.

    Inputs follow :func:`select_sparse_history_units`. ``fetched_indices`` is
    the fetch output of one buffer, [B, K] in Batch row order and top-k column
    order; ``write_indices`` [B] is that buffer's in-flight row by Batch row,
    written this step. Both may be strided views of the per-buffer outputs.

    ``out`` is [Q, K] in query order. A selected history unit takes its fetched
    row, the in-flight unit takes the write row, and padding queries, invalid
    selections and units past the unit length take -1. No tensor allocation,
    host read, or synchronization is performed.
    """
    device = out.device
    for tensor, name, ndim in (
        (topk_indices, "topk_indices", 2),
        (request_indices, "request_indices", 1),
        (batch_rows, "batch_rows", 1),
        (active_request_count, "active_request_count", 1),
        (unit_lengths, "unit_lengths", 1),
        (history_lengths, "history_lengths", 1),
        (fetched_indices, "fetched_indices", 2),
        (write_indices, "write_indices", 1),
        (out, "out", 2),
    ):
        check_sparse_offload_tensor(tensor, name, ndim, device)
    num_rows, width = fetched_indices.shape
    num_queries, topk = topk_indices.shape
    if unit_tokens <= 0:
        raise ValueError("unit_tokens must be positive")
    if topk != width or out.shape != (num_queries, topk):
        raise ValueError("Selections, fetch output and out must share the top-k width")
    if request_indices.shape != (num_queries,) or active_request_count.shape != (1,):
        raise ValueError("request_indices must be [Q] and active_request_count must be [1]")
    if (
        history_lengths.shape != (num_rows,)
        or write_indices.shape != (num_rows,)
        or batch_rows.shape != (num_rows,)
    ):
        raise ValueError(
            "history_lengths, write_indices and batch_rows must cover the native Batch"
        )
    if not num_queries <= unit_lengths.numel() <= num_rows:
        raise ValueError("Query and unit-length rows must fit the native Batch capacity")
    if num_queries == 0 or topk == 0:
        return out
    with torch.cuda.device(device):
        _merge_sparse_arena_indices_kernel[(num_queries,)](
            topk_indices,
            request_indices,
            batch_rows,
            active_request_count,
            unit_lengths,
            history_lengths,
            fetched_indices,
            write_indices,
            out,
            *topk_indices.stride(),
            request_indices.stride(0),
            batch_rows.stride(0),
            unit_lengths.stride(0),
            history_lengths.stride(0),
            *fetched_indices.stride(),
            write_indices.stride(0),
            *out.stride(),
            NUM_ROWS=num_rows,
            num_lengths=unit_lengths.numel(),
            TOPK=topk,
            UNIT_TOKENS=unit_tokens,
            BLOCK_K=triton.next_power_of_2(topk),
        )
    return out


@triton.jit(do_not_specialize=["num_rows", "num_out"])
def _gather_sparse_batch_rows_kernel(
    values_ptr,
    batch_rows_ptr,
    out_ptr,
    values_stride,
    batch_rows_stride,
    out_stride,
    num_rows,
    num_out,
    BLOCK: tl.constexpr,
):
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = position < num_out
    row = tl.load(batch_rows_ptr + position * batch_rows_stride, in_range, other=-1)
    valid = in_range & (row >= 0) & (row < num_rows)
    value = tl.load(values_ptr + row * values_stride, valid, other=-1)
    tl.store(out_ptr + position * out_stride, value, mask=in_range)


def gather_sparse_batch_rows(
    values: torch.Tensor,
    batch_rows: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Read a per-Batch-row value for each request, in input order.

    ``values`` [B] holds one int32 per native Batch row, such as one buffer's
    write rows; ``batch_rows`` [N] gives each request's Batch row, or -1 for a
    position without a live request, which reads -1. Both may be strided. ``out``
    [N] receives the result in place, also possibly strided, and is allocated
    when None. One kernel launch; no host read or synchronization.
    """
    device = values.device
    if out is None:
        out = torch.empty(batch_rows.shape, dtype=torch.int32, device=device)
    for tensor, name in ((values, "values"), (batch_rows, "batch_rows"), (out, "out")):
        check_sparse_offload_tensor(tensor, name, 1, device)
    if out.shape != batch_rows.shape:
        raise ValueError("out must have one entry per request")
    num_out = batch_rows.numel()
    if num_out == 0:
        return out
    block = 256
    with torch.cuda.device(device):
        _gather_sparse_batch_rows_kernel[(triton.cdiv(num_out, block),)](
            values,
            batch_rows,
            out,
            values.stride(0),
            batch_rows.stride(0),
            out.stride(0),
            values.numel(),
            num_out,
            BLOCK=block,
        )
    return out


@triton.jit
def _copy_pages_kernel(
    src_ptrs,
    dst_ptrs,
    num_words_ptr,
    BLOCK: tl.constexpr,
):
    page = tl.program_id(0)
    num_words = tl.load(num_words_ptr + page)
    src = tl.load(src_ptrs + page).to(tl.pointer_type(tl.int64))
    dst = tl.load(dst_ptrs + page).to(tl.pointer_type(tl.int64))
    for start in tl.range(tl.program_id(1) * BLOCK, num_words, tl.num_programs(1) * BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        in_page = offsets < num_words
        tl.store(dst + offsets, tl.load(src + offsets, mask=in_page), mask=in_page)


def copy_pages(
    src_addresses: torch.Tensor, dst_addresses: torch.Tensor, num_bytes: torch.Tensor
) -> None:
    """Copy page ``i`` of ``num_bytes[i]`` bytes from ``src_addresses[i]`` to ``dst_addresses[i]``.

    All three are CUDA int64 [N]. Addresses may name device-mapped host memory;
    each must be 8-byte aligned and each size a multiple of 8. Pages must not
    overlap. One kernel launch on the current stream.
    """
    device = src_addresses.device
    for tensor, name in (
        (src_addresses, "src_addresses"),
        (dst_addresses, "dst_addresses"),
        (num_bytes, "num_bytes"),
    ):
        if tensor.dtype != torch.int64 or tensor.ndim != 1 or tensor.device != device:
            raise ValueError(f"{name} must be a 1D int64 tensor on {device}")
    if not src_addresses.is_cuda:
        raise ValueError("Page copies take CUDA address arrays")
    if dst_addresses.shape != src_addresses.shape or num_bytes.shape != src_addresses.shape:
        raise ValueError("Every page needs a source, a destination and a size")
    num_pages = src_addresses.numel()
    if num_pages == 0:
        return
    with torch.cuda.device(device):
        _copy_pages_kernel[(num_pages, 4)](
            src_addresses,
            dst_addresses,
            num_bytes // 8,
            BLOCK=1024,
        )


@triton.jit
def _complete_sparse_tails_kernel(
    write_ptr,
    write_stride0,
    write_stride1,
    tail_ptrs,
    tail_row_bytes_ptr,
    arena_ptrs,
    cell_bytes_ptr,
    columns_ptr,
    num_words_ptr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    buffer = tl.program_id(1)
    column = tl.load(columns_ptr + buffer)
    cell = tl.load(write_ptr + row * write_stride0 + column * write_stride1)
    if cell >= 0:
        num_words = tl.load(num_words_ptr + buffer)
        src = (
            tl.load(tail_ptrs + buffer) + row.to(tl.int64) * tl.load(tail_row_bytes_ptr + buffer)
        ).to(tl.pointer_type(tl.int64))
        dst = (
            tl.load(arena_ptrs + buffer) + cell.to(tl.int64) * tl.load(cell_bytes_ptr + buffer)
        ).to(tl.pointer_type(tl.int64))
        for start in tl.range(tl.program_id(2) * BLOCK, num_words, tl.num_programs(2) * BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            in_page = offsets < num_words
            tl.store(dst + offsets, tl.load(src + offsets, mask=in_page), mask=in_page)


def complete_sparse_tails(
    write_indices: torch.Tensor,
    tail_addresses: torch.Tensor,
    tail_row_bytes: torch.Tensor,
    arena_addresses: torch.Tensor,
    cell_bytes: torch.Tensor,
    columns: torch.Tensor,
    num_words: torch.Tensor,
) -> None:
    """Copy each completed unit from its row's tail into the cell native named for it.

    ``write_indices`` is one layer's int32 [B, F] in-flight cells by Batch row,
    -1 where a row has no unit in flight. The other inputs are CUDA int64 [T],
    one entry per tail buffer of the layer: the address of row 0's tail page,
    the bytes between rows, the buffer's arena base, the bytes per arena cell,
    the buffer's column in ``write_indices`` and the 8-byte words to copy.
    Addresses and strides are multiples of 8. One kernel launch on the current
    stream; no allocation or host read, so it can be captured.
    """
    device = write_indices.device
    if write_indices.dtype != torch.int32 or write_indices.ndim != 2 or not write_indices.is_cuda:
        raise ValueError("write_indices must be a 2D CUDA int32 tensor")
    arrays = (tail_addresses, tail_row_bytes, arena_addresses, cell_bytes, columns, num_words)
    if any(
        array.dtype != torch.int64 or array.shape != tail_addresses.shape or array.device != device
        for array in arrays
    ):
        raise ValueError("Tail copy descriptors must be int64 [T] tensors on the write device")
    num_rows = write_indices.shape[0]
    num_buffers = tail_addresses.numel()
    if num_rows == 0 or num_buffers == 0:
        return
    with torch.cuda.device(device):
        _complete_sparse_tails_kernel[(num_rows, num_buffers, 4)](
            write_indices,
            *write_indices.stride(),
            tail_addresses,
            tail_row_bytes,
            arena_addresses,
            cell_bytes,
            columns,
            num_words,
            BLOCK=1024,
        )
