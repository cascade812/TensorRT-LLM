<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sparse KV offload contracts

This document proposes internal interfaces for retaining sparse KV history on
host and fetching selected entries into a bounded GPU cache. It is a foundation
for parallel implementation and owner review. It does not enable offload or
declare a model supported. Storage, kernel, scheduler, and model implementations
can be reviewed separately against these contracts.

The typed definitions are in
[`kv_offload_contracts.py`](../../../tensorrt_llm/_torch/attention/backends/sparse/kv_offload_contracts.py).
They use only standard-library runtime imports when loaded as a source module;
normal package imports still have the usual TensorRT-LLM dependencies. They do
not import draft KVCM bindings. These are internal descriptors and protocols,
not public LLM arguments or a replacement for the existing per-request `KvCache`.

## Ownership and implementation boundaries

| Owner                    | Produces or implements                                                      | Consumer                                       |
| ------------------------ | --------------------------------------------------------------------------- | ---------------------------------------------- |
| Model adapter            | `LayerBinding`, `EntryLayout`, `SelectionBatch`, `NewestEntries`            | Host adapter and cache provider                |
| KVCM adapter             | `HostStorageProvider`, `HostReadLease`, host `MemoryRequirements`           | Executor, source publication, capacity planner |
| GPU cache provider       | `CacheProvider`, `PreparedCache`, `HostSources`, cache `MemoryRequirements` | Model adapter, attention, capacity planner     |
| Scheduler/capacity owner | `ReservationPlanner`, `Reservation`                                         | Request admission and provider setup           |
| Executor integration     | `RequestLifecycle`, stream/completion adaptation, status handling           | Scheduler and model forward                    |

Providers own allocation formulas, byte layouts, address validity, and execution.
The planner owns admission policy and accounting. It must not reproduce kernel
workspace formulas or inspect a concrete provider's tensors. KVCM retains
logical pages and shared-prefix identity. The selected-entry cache owns its GPU
copies and reader protection, not a second request history.

Future adapters connect this contract to the existing sparse registry and
attention interfaces. This foundation change does not widen shared attention
or public configuration APIs.

## Identity, units, and selection

| Field                                            | Meaning                                                                                              |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `request_id`                                     | Nonnegative ID unique for a request lifetime, not its batch row; use a new ID when a slot is reused. |
| `LayerBinding.model_layer_id`                    | Model layer number. IndexShare followers still have their own KV binding.                            |
| `LayerBinding.cache_layer_id`                    | Unique KVCM layer entry, allowing indexer and attention KV to have distinct entries.                 |
| `LayerBinding.layer_group_id`                    | KVCM-assigned lifecycle group; never substitute a residency-group label or pool index.               |
| `RequestSpec.history_tokens` / `reserved_tokens` | Model token counts; the latter includes admitted generation capacity.                                |
| Selection positions / `valid_entries`            | Native stored entries: tokens for DSA, completed compressed entries for compressed layouts.          |
| Page index                                       | Native-entry position divided by `EntryLayout.entries_per_page`.                                     |
| `AllocationId`                                   | Allocation identity scoped to a particular provider/allocator, including its lifetime.               |

The model/KVCM adapter owns token-to-entry conversion. Incomplete compression
groups must not be advertised as completed entries. A constant compression
ratio is not assumed by the shared interfaces.

`SelectionBatch` preserves the model's selection, order, and duplicate columns.
It carries CUDA `int64[R]` request IDs, `int32[Q]` query-to-request rows,
`int32[Q, K]` logical positions, `int32[Q]` causal entry bounds, and an optional
`bool[Q, K]` mask. Inactive request/query rows use `-1`. Negative positions,
positions outside the causal bound, and masked columns are padding/invalid.
They produce output index `-1`. A logically valid entry that exceeds the
reserved history is an error. Missing host/GPU storage is also an error, never
a reason to change the logical selection.

All input buffers are borrowed. Shape, dtype, layout binding, device, and
contiguity are checked during preparation. Value-dependent checks run on GPU.
Callers preserve input storage and contents until consumers finish.

## Entry bytes and attention indices

`EntryLayout` describes host byte spans, including scale bytes and coalesced
layer offsets. Each component names its host pool, offset, stride, and entry
size. There is no implicit quantization or lossy conversion. An encoded cold
page that cannot provide these spans needs a supported decoded layout before
publication, or setup must fail.

`ResidentEntries.components` are contiguous CUDA byte buffers with shape
`[request_capacity, physical_capacity, component.size]`, in layout order.
An output index addresses the first two dimensions flattened together. This
mapping is independent of ordinary KVCM GPU page tables. The attention adapter
interprets these bytes in the model's format, including scales and residuals.

`CacheSpec.cache_capacity` counts regular LRU entries per request. Extra newest
or write slots are provider-owned storage included in its quote. The provider
validates supported capacity, index-width limits, and the maximum simultaneous
selection union before allocation. `top_k` is a per-query bound; multiple queries
can require a larger union for the same request. Model top-K is not reduced to
fit a budget. The first DSA integration can require one decode query per request;
unsupported multi-query bounds must be rejected explicitly.

## Host publication, growth, and writes

`HostStorageProvider.backup()` enqueues backup after writes on the request
stream. It preserves the GPU source through copy completion. `acquire()` returns
a lease with immutable version, entry coverage, mapped addresses, and a readiness
dependency. Acquiring a lease does not imply that the CPU can read its bytes yet.
`ready.wait_on(stream)` queues a dependency without a CPU wait.

`PreparedCache.sources` owns the fixed-address GPU source tables and the leases
published into them. Their GPU metadata is included in the cache provider's
quote. KVCM's host quote covers the actual pinned host allocations. Allocation
identities prevent charging shared storage twice.

Before publication, `bind_requests()` binds the full request-row vector, including
empty histories and inactive rows. Rebinding preserves unchanged rows; changed
rows must first retire their old sources and cache mappings.

Publication is incremental, outside graph capture:

1. Acquire the newly backed-up page's read lease.
1. Call `sources.publish()` with its request row and a fence covering previous
   consumers of changed table entries. Validate all CPU descriptors first.
1. On the publication stream, wait for backup readiness, update addresses and
   version, then publish readable entry coverage. Order replay after publication.
1. Ownership transfers when submission begins. Validation failure before that
   leaves ownership with the caller. Once submission begins, the provider owns
   the new leases even on failure, retaining old/new resources for cleanup.

`publish()` does not replace table tensor storage. Newly generated pages fit
within pre-reserved table capacity. Exceeding that capacity requires a new setup
and graph, with both old and new allocations budgeted during transition.

For a writable partial page with an existing host copy:

1. Withdraw its published source after previous consumers. The returned fence
   covers the table update and retired readers.
1. Invalidate the old host version before submitting new writes, ordered after
   that fence. Open read leases cause rejection. Committed data is immutable.
1. Write and back up the new coverage; acquire and publish a new version.

Withdrawal sets coverage to zero before subsequent GPU fetches. A cache hit
on an immutable completed entry can remain usable while its host backup is
unavailable. GPU replacement is legal only when a retained host copy covers the
entry and all protected readers have finished.

One layer's table update cannot invalidate a shared page's other readers. Host
addresses stay pinned through all leases, including leases that outlive request
closure. A version changes when contents are invalidated or storage is reused;
a raw address alone never establishes identity. Disk restoration occurs before
decode publication, with staging and completion included in the provider's
requirements. Selected-entry fetch does not perform disk I/O.

## Setup, capture, and completion

CPU setup creates reservations, pools, tables, events, prepared steps, and
compiled kernels before capture. Every simultaneously outstanding selection,
prefetch, or newest-entry operation has its own prepared workspace. Reusing a
workspace while its lease is active is an error.

The model supplies completed GPU bytes for `NewestEntries`; the cache does not
produce new KV or perform model scoring. A native compressed entry is published
only when complete. Reusing newest-entry storage waits for its host backup and
readers. Already published logical entries are immutable.

The captured per-layer sequence is:

```text
model writes and selection
  -> ensure_resident(prepared_step)
  -> attention consumes components/indices, honoring per-request device status
  -> release(prepared_step, after=attention_completion)
```

`ensure_resident()` and `release()` allocate no tensors, create no events, and
read no GPU values on the CPU. They run on the cache's configured mutation
stream. Attention on another stream waits on the result's readiness; release
waits on attention completion. Capture events are prepared and warmed beforehand.
Replays update buffer contents, not addresses.

CPU retirement uses completion fences tied to a particular submitted use. A
fence used to release storage or a reservation cannot be re-recorded to refer
to unrelated work while retirement is pending. Provider adapters bridge this
contract to CUDA events and existing KVCM event handling.

IndexShare shares logical selections. Replaying another layer's physical miss
plan is an optional provider optimization that additionally requires matching
slot/LRU state and protection. It is not guaranteed by these interfaces.

## Capacity and reservations

Each provider returns `MemoryRequirements`, a complete allocation set with
absolute byte sizes. Quoting validates bounds and makes no CUDA or pinned-memory
allocation. A quote may assign planning identities; repeated quotes for the same
live allocation preserve them through reservation and creation.

The owner includes allocator rounding or a documented conservative upper bound,
regular cache entries, newest/write storage, full-history maps, LRU/protection
state, source tables, all prepared workspaces, and setup peaks. Host requirements
cover history and admitted generation at actual page/codec allocation granularity.
Shared slabs must be reported consistently: do not separately charge both a slab
and its constituent pages.

Budgets are bytes per process-local GPU device and bytes for the host tier. They
cover owned allocations and reserved growth, not framework allocator reservations,
free-slot counts, or only cached KV payload. The scheduler separately accounts
for model weights, indexer/dense KV, prefill, attention workspaces, and transport
buffers. Existing attention workspace accounting must not be double-counted.
User configuration units/defaults remain separate work.

The planner contract is:

- `reserve()` atomically accepts all requirements or raises `MemoryError`.
- Identical allocation IDs agree on tier, device, and size. Conflicts raise
  `ValueError` without changing accounting. Diagnostic labels do not define identity.
- Shared host allocations are charged once across live claims. Request-local GPU
  copies have distinct IDs. A persistent batch cache is charged once for its full
  request capacity, not again for every occupied row.
- `replace()` consumes a complete new requirement set. On failure the old claim
  remains intact. Include old/new allocations together while they coexist;
  remove old charges only after their users finish.
- `release(after=completion)` retires one claim. Shared allocations and pending
  readers/copies remain charged. Repeated release cannot shorten protection.
  `collect()` is nonblocking.

Provider allocation fits an accepted quote; quoting, reservation, and creation
are serialized with the existing CPU resource-management path. If creation
fails after submitting work, retire the reservation after a fence covering that
work. `SubmissionError.completion` supplies that fence; the provider owns the
partial resources until retirement. Ordinary validation/capacity exceptions mean
no submission or ownership transfer. `HostStorageProvider.prepare()` binds an
accepted request reservation before backup/growth. A failed allocation does not
permit an unreserved smaller selection.

Accepting a byte quote is not sufficient to admit a request into an already
allocated pool. `HostStorageProvider.prepare()` must reserve its history and
admitted generation slots; `RequestLifecycle.prepare()` also reserves a GPU
request row. These operations can reject a request even when its byte quote fits.
Rollback a tentative byte claim on rejection before submission. Future growth
within an accepted reservation cannot consume capacity promised to another request.
Providers keep their own claims on slabs or prefix-cache allocations retained
after request closure; releasing the last request claim does not prove the
underlying allocation was freed.

This boundary lets the capacity owner implement admission with fake providers
whose reports vary independently of any HiSparse allocation formula.

## Request lifecycle and errors

`RequestLifecycle.prepare()` runs before a step and checks reserved growth. It
also resumes suspended requests. One request retains one native `KvCache`, with
separate ownership for host copies and selected GPU entries. Prefill-to-decode
integration backs up completed sparse history before releasing GPU pages.
Actual shrinking/releasing of large allocations is accounted separately from
releasing page locks; prefill retains its own capacity peak.

`commit()` preserves shared logical identity, immutable committed contents, and
pending-copy events. `suspend()` retains history/references while retiring
transient uses. `close()` retires that request's mappings and leases; other
requests may still own shared data. Returned completions govern resource and
reservation release. Before reusing a row, clear old mappings, withdraw sources,
finish old uses, and publish a new request ID.

Host validity, request identity, selection range, cache capacity, and active
readers are checked before publishing cache indices. Per-request fetch failure
preserves that row's existing payload/mapping; successful rows still acquire
leases and require release. A busy step retains its old lease and cannot be
consumed as a new result.

GPU status is mandatory: attention must not dereference a failed row's indices.
Integration must provide a status-aware consumer or GPU guard before graph
enablement. A CPU status check after forward reports failure before returning
outputs; it cannot make an unsafe captured attention launch safe. A `SubmissionError` requires cleanup using its completion even if the operation
returned no handle. Failure after model writes requires orderly cleanup and cannot trigger automatic retry of a
partially executed forward.

## Tests and parallel implementation

The CPU examples are in
[`test_kv_offload_contracts.py`](../../../tests/unittest/_torch/attention/sparse/test_kv_offload_contracts.py).
They include a reference reservation ledger and completion/read-lease doubles.
Tests cover sharing, exact budgets, failed growth, conflicting identities,
multiple devices, deferred release, and backup versus reader completion.
They run without a GPU, model weights, native bindings, or PyTorch:

```bash
python3 tests/unittest/_torch/attention/sparse/test_kv_offload_contracts.py
```

With pytest installed:

```bash
python3 -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/unittest/_torch/attention/sparse/test_kv_offload_contracts.py
```

These examples specify consumer expectations; they do not prove real-provider
asynchronous memory safety. Implementation PRs must add:

| Implementation   | Required provider checks                                                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Host adapter     | Pending backup, partial-page withdrawal/invalidation/republish, shared pages, host exhaustion, disk restore, and leases after close.                    |
| Cache provider   | Actual KV/scale copies, duplicate/padded selections, full replacement, newest entries, side-stream readers, and changed request IDs under graph replay. |
| Capacity planner | Exact/insufficient budgets, mixed prefill/decode, shared allocations, growth rollback, and deferred cleanup using provider reports.                     |
| Model/executor   | Selection/output parity, status-aware attention, prefill transition, prefix reuse, suspend/resume, and cleanup after partial writes.                    |

Storage/cache owners implement providers; the capacity owner implements
reservation/admission against these interfaces; model/executor owners supply
selection/layout adaptation and lifecycle wiring. The interface PR can land
first, giving the implementation PRs a common mainline foundation.

KVCM, attention, and scheduler owners should review identity, publication,
budget, and failure semantics before treating these as agreed contracts.
Boundary changes update definitions, guide, and examples together. Kernel launch
settings, allocator internals, and replacement optimizations stay provider choices.
