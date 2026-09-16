# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal contracts for selected-entry KV offload; no storage or kernel implementation.

Runtime imports use only the standard library. Tensor annotations describe borrowed
PyTorch buffers; they do not allocate buffers or import the native KVCM bindings.
See docs/source/developer-guide/sparse-kv-offload-contracts.md for ordering,
ownership, error handling, and the implementation acceptance criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import TYPE_CHECKING, Literal, NewType, Protocol

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "AllocationId",
    "CacheProvider",
    "CacheSpec",
    "Completion",
    "EntryComponent",
    "EntryLayout",
    "HostPageBinding",
    "HostPageRef",
    "HostReadLease",
    "HostSources",
    "HostSourceView",
    "HostSpan",
    "HostStorageProvider",
    "LayerBinding",
    "MemoryRequirement",
    "MemoryRequirements",
    "NewestEntries",
    "PreparedCache",
    "PreparedStep",
    "RequestLifecycle",
    "RequestSpec",
    "Reservation",
    "ReservationPlanner",
    "ResidentEntries",
    "ResidencyStatus",
    "SelectionBatch",
    "StreamHandle",
    "SubmissionError",
]

StreamHandle = NewType("StreamHandle", int)
"""Borrowed CUDA stream handle; adapters bridge torch.cuda.Stream.cuda_stream."""


@dataclass(frozen=True)
class LayerBinding:
    """Model layer, unique KVCM layer entry, and KVCM-assigned lifecycle group.

    A model layer can have several bindings, e.g. indexer and attention KV.
    layer_group_id is never a residency-group label or a pool-group index.
    """

    model_layer_id: int
    cache_layer_id: int
    layer_group_id: int


@dataclass(frozen=True)
class EntryComponent:
    """Native KV or scale byte span within one host page pool.

    Entry i occupies [offset + i * stride, offset + i * stride + size).
    Offsets include coalesced-layer offsets and the cold codec's layout.
    """

    name: str
    pool_index: int
    offset: int
    stride: int
    size: int


@dataclass(frozen=True)
class EntryLayout:
    """One layer's token or compressed-entry layout, including all scale bytes.

    entries_per_page counts native stored entries. The model adapter converts
    KVCM token coverage to this unit, including incomplete compressed groups.
    Host codecs must supply an addressable layout or decode before publication.
    """

    binding: LayerBinding
    entries_per_page: int
    host_slot_bytes: tuple[int, ...]
    components: tuple[EntryComponent, ...]

    def __post_init__(self) -> None:
        if self.entries_per_page <= 0 or not self.host_slot_bytes or not self.components:
            raise ValueError("A layout needs positive entry capacity, pools, and components")
        if min(self.host_slot_bytes) <= 0:
            raise ValueError("Host slot sizes must be positive byte counts")
        if len({c.name for c in self.components}) != len(self.components):
            raise ValueError("Component names must be unique")
        for component in self.components:
            if (
                not component.name
                or not 0 <= component.pool_index < len(self.host_slot_bytes)
                or component.offset < 0
                or not 0 < component.size <= component.stride
            ):
                raise ValueError("Invalid entry component")
            end = component.offset + (self.entries_per_page - 1) * component.stride + component.size
            if end > self.host_slot_bytes[component.pool_index]:
                raise ValueError("Entry component extends beyond its host page slot")


@dataclass(frozen=True)
class SelectionBatch:
    """Borrowed, contiguous CUDA tensors describing one layer's logical selection.

    request_ids: int64 [R], unique request lifetimes; -1 marks inactive rows.
    query_rows: int32 [Q], indexes request_ids; -1 marks padded queries.
    positions: int32 [Q, K], token or compressed-entry positions, never GPU indices.
    valid_lengths: int32 [Q], causal completed-entry bounds.
    valid_mask: optional bool [Q, K]. Negative positions are padding. Entries
        outside the causal bound or masked out are invalid. Order and duplicate
        columns are preserved. Host/GPU residency never changes logical validity.

    Storage and shapes stay fixed through capture/replay. Contents may change
    before a replay, ordered after the previous consumer. CPU code must not read
    tensor values in the per-layer path.
    """

    binding: LayerBinding
    request_ids: Tensor
    query_rows: Tensor
    positions: Tensor
    valid_lengths: Tensor
    valid_mask: Tensor | None = None


@dataclass(frozen=True)
class CacheSpec:
    """Setup bounds for one layer and one independently executing cache instance.

    cache_capacity is regular LRU entries per request, excluding extra write
    slots. Providers declare those slots in their requirements. top_k bounds
    columns per query; max_queries_per_request bounds the selected union.
    max_inflight_steps counts all simultaneous attention/prefetch/newest leases.
    """

    layout: EntryLayout
    request_capacity: int
    logical_capacity: int
    cache_capacity: int
    top_k: int
    max_queries_per_request: int = 1
    max_inflight_steps: int = 1

    def __post_init__(self) -> None:
        if (
            min(
                self.request_capacity,
                self.logical_capacity,
                self.cache_capacity,
                self.top_k,
                self.max_queries_per_request,
                self.max_inflight_steps,
            )
            <= 0
        ):
            raise ValueError("Cache dimensions and workspace counts must be positive")
        if self.cache_capacity > self.logical_capacity:
            raise ValueError("Cache capacity exceeds logical capacity")


class Completion(Protocol):
    """Completion of submitted work; implementations own the underlying event.

    wait_on queues a stream dependency without CPU synchronization. is_complete
    is a nonblocking CPU query for setup/retirement, never the captured hot path.
    A retirement fence must cover a specific submission, not an event that can
    be re-recorded before its retirement is consumed.
    """

    def wait_on(self, stream: StreamHandle) -> None: ...

    def is_complete(self) -> bool: ...


class SubmissionError(RuntimeError):
    """Provider failed after enqueueing work or adopting resources.

    The provider retains and retires affected storage/leases through completion.
    The caller must clean up the request and defer reservation release to that
    fence. Ordinary validation/capacity exceptions mean no work was submitted
    and ownership was not transferred. This is a CPU submission error, distinct
    from a per-request device ResidencyStatus.
    """

    def __init__(self, message: str, completion: Completion) -> None:
        super().__init__(message)
        self.completion = completion


@dataclass(frozen=True)
class AllocationId:
    """Provider-scoped identity of one allocation or reserved future allocation.

    namespace identifies an allocator/cache instance, not its implementation
    class. key is never reused while a reservation or deferred release exists.
    Aliases report the same ID. Request-local GPU copies report different IDs.
    """

    namespace: str
    key: int


@dataclass(frozen=True)
class MemoryRequirement:
    """Absolute reserved bytes for one allocation, including allocator rounding.

    device_index is a process-local CUDA ordinal for GPU, and -1 for host.
    purpose is a diagnostic label, not a budget domain or sharing key.
    """

    allocation_id: AllocationId
    memory: Literal["gpu", "host"]
    device_index: int
    size_bytes: int
    purpose: str

    def __post_init__(self) -> None:
        if self.memory not in ("gpu", "host") or self.size_bytes < 0:
            raise ValueError("Requirements need a valid memory tier and nonnegative bytes")
        if (self.memory == "host" and self.device_index != -1) or (
            self.memory == "gpu" and self.device_index < 0
        ):
            raise ValueError("Host uses device -1; GPU uses a nonnegative device ordinal")


@dataclass(frozen=True)
class MemoryRequirements:
    """Complete desired allocation set, not incremental bytes or free-page counts.

    The planner deduplicates by allocation_id across live reservations. Identical
    IDs must agree on memory, device, and size. Growth that replaces storage uses
    a new ID and reserves coexistence until old work finishes. Providers include
    payload, writes/newest, mappings, state, workspaces, and allocation peaks.
    Shared allocator slabs must be reported at one agreed accounting granularity.
    """

    allocations: tuple[MemoryRequirement, ...]


@dataclass(frozen=True)
class RequestSpec:
    """CPU request bounds in model token units, before admission or step growth.

    history_tokens includes reusable prefixes; reserved_tokens is the total
    admitted history plus generation bound, not the next decode step's length.
    The provider resolves actual shared allocations through its KVCM ownership.
    """

    request_id: int
    history_tokens: int
    reserved_tokens: int

    def __post_init__(self) -> None:
        if self.request_id < 0 or not 0 <= self.history_tokens <= self.reserved_tokens:
            raise ValueError("Invalid request lifetime or history/generation reservation")


class Reservation(Protocol):
    """One owner's accepted claim; physical storage is still owned by its provider."""

    @property
    def requirements(self) -> MemoryRequirements: ...

    def release(self, *, after: Completion | None = None) -> None:
        """Retire this claim once; keep shared/pending claims charged until safe."""
        ...


class ReservationPlanner(Protocol):
    """CPU admission accounting, serialized with provider planning/allocation.

    Insufficient budgets raise MemoryError; malformed/conflicting identities
    raise ValueError. Rejection leaves accounting unchanged. Pending releases
    remain charged. Budget domains are host and each individual GPU device.
    """

    def reserve(self, requirements: MemoryRequirements) -> Reservation: ...

    def replace(self, reservation: Reservation, requirements: MemoryRequirements) -> None:
        """Atomically update a live claim before work; retain old claims on failure.

        The caller includes still-live old allocations in requirements. Shrinking
        a claim is legal only after their consumers complete.
        """
        ...

    def collect(self) -> None:
        """Reclaim completed deferred releases without blocking the CPU."""
        ...


@dataclass(frozen=True)
class HostPageRef:
    """Request-relative page; page_index uses EntryLayout.entries_per_page units."""

    request_id: int
    binding: LayerBinding
    page_index: int


@dataclass(frozen=True)
class HostSpan:
    """One pinned pool span of a page, valid for its host-read lease.

    device_address is CUDA-mapped, not an arbitrary CPU pointer. size_bytes is
    the page span, not the allocation's charged size. allocation_id names that
    underlying allocation for shared-memory accounting.
    """

    allocation_id: AllocationId
    device_address: int
    size_bytes: int


class HostReadLease(Protocol):
    """Immutable page version and entry coverage protected through a completion.

    ready orders backup completion. valid_entries becomes readable after that
    dependency; it is not proof the copy has completed on the CPU. Spans follow
    EntryLayout's pool order. This lease may outlive request closure.
    """

    @property
    def page(self) -> HostPageRef: ...

    @property
    def version(self) -> int: ...

    @property
    def valid_entries(self) -> int: ...

    @property
    def spans(self) -> tuple[HostSpan, ...]: ...

    @property
    def ready(self) -> Completion: ...

    def release(self, *, after: Completion) -> None:
        """Idempotently retire the lease; retain addresses until after completes."""
        ...


class HostStorageProvider(Protocol):
    """CPU adapter over the existing request KVCM; operations run outside capture.

    Byte and entry layout conversions belong to this adapter and the model.
    Needed disk pages are restored before publication, with staging budgeted.
    """

    def requirements(self, request: RequestSpec) -> MemoryRequirements:
        """Quote history/generation storage without allocating or changing residency."""
        ...

    def prepare(self, request: RequestSpec, reservation: Reservation, stream: StreamHandle) -> None:
        """Accept the quote and reserve history/generation slots before submission.

        Byte-budget acceptance alone does not guarantee free slots in an already
        allocated pool. Reserve that suballocation capacity here or fail before
        writes; retain the reservation through the request's admitted growth.
        """
        ...

    def backup(self, page: HostPageRef, valid_entries: int, stream: StreamHandle) -> Completion:
        """Copy after GPU writes; keep the GPU source until backup completes."""
        ...

    def acquire(self, page: HostPageRef) -> HostReadLease:
        """Acquire a valid version; pending readiness is expressed by the lease."""
        ...

    def invalidate(self, page: HostPageRef, stream: StreamHandle, *, after: Completion) -> None:
        """Invalidate before writes; reject committed data or outstanding leases.

        Withdraw published sources first. Order future writes after retired
        readers through after; advance the page version before republishing.
        """
        ...

    def offload(self, page: HostPageRef, stream: StreamHandle) -> Completion:
        """Release a held GPU page using its retained copy, preserving completion."""
        ...


@dataclass(frozen=True)
class HostPageBinding:
    """Bind one acquired page to a preallocated request row; publication adopts it."""

    request_row: int
    lease: HostReadLease


@dataclass(frozen=True)
class HostSourceView:
    """Borrowed, fixed-address CUDA tables consumed by the cache.

    request_ids: int64 [R], request lifetimes; -1 for inactive rows.
    page_addresses: one int64 [R, P] tensor per host pool, CUDA-mapped page bases.
    versions: int64 [R, P]; updated on invalidation/reuse.
    valid_entries: int32 [R, P], readable prefix after publication's dependency.
        Zero means unavailable; no kernel may dereference that page's addresses.
    The owning HostSources retains all leases through submitted reads.
    """

    layout: EntryLayout
    request_ids: Tensor
    page_addresses: tuple[Tensor, ...]
    versions: Tensor
    valid_entries: Tensor


class HostSources(Protocol):
    """CPU publication owner for graph-stable tables, created before capture.

    publish/withdraw update tensor contents on stream after earlier consumers.
    They never replace tensor storage. Request-row reassignment first withdraws
    its old pages and retires the old cache mapping. Shape/identity errors fail
    before submission. Ownership transfers when submission begins; failures
    after that point raise SubmissionError and keep new leases provider-owned.
    """

    @property
    def view(self) -> HostSourceView: ...

    def bind_requests(
        self, request_ids: tuple[int, ...], stream: StreamHandle, *, after: Completion
    ) -> None:
        """Bind every row, including empty histories; -1 marks inactive rows.

        Preserve sources for unchanged IDs. Changed rows must have no published
        sources or pending uses. The executor also clears their old cache maps.
        """
        ...

    def publish(
        self, pages: tuple[HostPageBinding, ...], stream: StreamHandle, *, after: Completion
    ) -> None:
        """Add/replace sources, waiting on each lease's ready dependency on stream.

        Pending coverage is not visible before its backup. Retire replaced
        leases after prior reads. Other published pages remain valid.
        """
        ...

    def withdraw(
        self, pages: tuple[HostPageRef, ...], stream: StreamHandle, *, after: Completion
    ) -> Completion:
        """Clear availability, retire matching leases, and return their completion."""
        ...

    def close(self, *, after: Completion) -> Completion:
        """Retire all sources/tables and return completion; forbid publication."""
        ...


class ResidencyStatus(IntFlag):
    """Per-request device status; nonzero forbids attention reads for that row."""

    OK = 0
    MISSING_HOST = 1
    CAPACITY = 2
    BUSY = 4
    INVALID_SELECTION = 8
    STALE_REQUEST = 16


@dataclass(frozen=True)
class ResidentEntries:
    """Preallocated cache result, borrowed until the step's consumers finish.

    components: uint8 [R, physical_capacity, component.size] per layout component.
    indices: int32 [Q, K], addresses R/physical_capacity flattened; padding is -1.
    status: int32 [R], ResidencyStatus bits. Missing required data is an error,
        never an instruction to silently omit selected KV from attention.
    ready orders reads of all buffers. Component format/bytes match EntryLayout.
    """

    components: tuple[Tensor, ...]
    indices: Tensor
    status: Tensor
    ready: Completion


@dataclass(frozen=True)
class NewestEntries:
    """Completed immutable GPU entries that can be read before host backup.

    request_ids: int64 [R]; positions: int32 [R], -1 means no new entry.
    components: uint8 [R, component.size] in layout order. Producer writes precede
    ensure_resident on its stream. Replacing these bytes needs a valid host copy
    and completed readers. Published logical entries cannot be overwritten.
    """

    request_ids: Tensor
    positions: Tensor
    components: tuple[Tensor, ...]


class PreparedStep(Protocol):
    """Preallocated inputs, result, and one reusable read lease for a cache step."""

    @property
    def result(self) -> ResidentEntries: ...


class PreparedCache(Protocol):
    """One layer's cache; mutations use one configured stream.

    prepare runs before capture. ensure_resident/release enqueue only GPU work,
    with no tensor allocation or CPU tensor reads. Per-request failure preserves
    that row's payload/mapping and returns a nonzero status. Other rows may
    succeed and retain leases that still require release.
    """

    @property
    def sources(self) -> HostSources:
        """The cache's preallocated source tables; their bytes are in its quote."""
        ...

    def prepare(
        self,
        selection: SelectionBatch,
        *,
        newest: NewestEntries | None = None,
    ) -> PreparedStep: ...

    def ensure_resident(self, step: PreparedStep) -> ResidentEntries:
        """Fetch misses and acquire read protection; keep order and duplicates."""
        ...

    def release(self, step: PreparedStep, *, after: Completion) -> None:
        """Release successful rows after attention; safe for failed/inactive rows."""
        ...

    def close(self, *, after: Completion) -> Completion:
        """Outside capture, retire resources and return completion; forbid replay."""
        ...


class CacheProvider(Protocol):
    """Provider owns allocation formulas and setup; consumers own budget policy."""

    def requirements(self, spec: CacheSpec) -> MemoryRequirements:
        """Validate supported bounds and quote all storage without CUDA allocation.

        Include extra write slots, logical maps, every prepared workspace, host
        table metadata, and setup peaks. Unsupported bounds raise ValueError.
        Quotes use stable allocation IDs through reservation and creation.
        """
        ...

    def create(
        self, spec: CacheSpec, reservation: Reservation, stream: StreamHandle
    ) -> PreparedCache:
        """Allocate within the quote; SubmissionError carries partial-work completion."""
        ...


class RequestLifecycle(Protocol):
    """CPU integration over one per-request KvCache, outside layer execution.

    prepare checks growth/reservation and publishes metadata before a step.
    Failure before submission leaves prior usable state intact. Failure after
    writes poisons that request for cleanup; partial forwards are never retried
    automatically and cannot return successful outputs.
    """

    def prepare(self, request: RequestSpec, reservation: Reservation, stream: StreamHandle) -> None:
        """Reserve request rows/host slots and prepare admitted work before writes.

        Also handle prefill-to-decode backup/publication. Rejection requires the
        caller to roll back its tentative byte claim; partial submission instead
        raises SubmissionError with the completion needed for cleanup.
        """
        ...

    def commit(self, request_id: int, history_tokens: int, stream: StreamHandle) -> None:
        """Commit submitted writes without changing shared identity or copy events."""
        ...

    def suspend(self, request_id: int, *, after: Completion) -> Completion:
        """Keep logical history/shared references; return transient-use completion."""
        ...

    def close(self, request_id: int, *, after: Completion) -> Completion:
        """Retire mappings/leases; returned completion gates slot/budget reuse."""
        ...
