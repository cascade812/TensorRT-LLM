/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/stagingBuffer.h"
#include "kv_cache_manager_v2/storage/config.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// SparseLeaseIndex and SparseArenaView — the device-resident state of one request's sparse
// history cache for one life cycle, as the fetch kernel sees it: a slot per row, and the
// geometry that slot is read within.
//
// A reserved slot is a row of buffer-sized cells, and any layer's unit may occupy any cell;
// a cell holds `unitsPerPage` units when a selection unit is smaller than a buffer. An entry
// is identified by (layer, entry index) and resolved to a cell arithmetically, so nothing
// ties a layer to a fixed position.
//
// The metadata is laid out [layer][entry]. It is per LAYER, not per buffer: every sparse
// buffer of a layer selects the same units, so one resident set and one recency order serve
// them all, and the entry index the resolve produces addresses each buffer's own cell.
// ---------------------------------------------------------------------------
//! The arena slot a row's member holds, or kNoSparseLease when the row carries none.
//!
//! This is the whole of a row's published cache state: everything else a kernel needs is
//! either arithmetic from this index -- a lease's cells and its metadata both sit at a
//! fixed stride from the arena's bases, see "Why the cache is a flat array of cells" in
//! SPARSE_ATTENTION_DESIGN.md -- or a constant of the life cycle, which SparseArenaView
//! carries once per launch rather than once per row.
using SparseLeaseIndex = int32_t;
inline constexpr SparseLeaseIndex kNoSparseLease = -1;

//! One life cycle's arena, as a kernel addresses it. Fixed for the arena's lifetime, so a
//! launch takes it by value and no row carries a copy.
struct SparseArenaView
{
    //! Lease i's metadata begins at `metadataBase + i * leaseStride`: first its resident
    //! unit ordinal per entry, kBadBlockOrdinal where the entry is free, then at
    //! `lruOffset` its entry indices in recency order, most-recently-used first.
    //!
    //! The ordinals are dense rather than keyed: a fetch builds a hash over them in shared
    //! memory to answer "is this ordinal resident, and where".
    std::byte* metadataBase = nullptr;
    uint64_t leaseStride = 0;
    int32_t lruOffset = 0;

    //! [numBuffers] each buffer's index among those sharing its pool: where its cells sit
    //! inside that pool's arena region.
    int32_t const* bufferSlotInPool = nullptr;

    //! [numBuffers] the layer each buffer belongs to, i.e. which metadata row carries its
    //! residency and its work list. Non-decreasing: the buffers of a layer are consecutive,
    //! which is what lets a one-layer call name its buffers as a range rather than a set.
    int32_t const* bufferLayer = nullptr;

    //! Layers in this life cycle: the metadata's first dimension.
    int32_t numLayers = 0;
    //! Sparse buffers in this life cycle, one per (layer, role). Cells and page offsets are
    //! per buffer, so this is the dimension the copy walks.
    int32_t numBuffers = 0;
    int32_t entriesPerLayer = 0;

    [[nodiscard]] constexpr int32_t numEntries() const noexcept
    {
        return numLayers * entriesPerLayer;
    }
};

// ---------------------------------------------------------------------------
// SparseCacheInit — everything a request's cache state is built from, all of it
// fixed for the life of the reservation.
// ---------------------------------------------------------------------------
struct SparseCacheInit
{
    int32_t numLayers;
    int32_t numBuffers;
    int32_t entriesPerLayer;
    //! Cells one buffer gets, i.e. entriesPerLayer / unitsPerPage. A lease is this many cells
    //! per buffer, laid out buffer-major.
    int32_t cellsPerBuffer;
    //! [numBuffers] the layer each buffer belongs to, non-decreasing.
    std::vector<int32_t> bufferLayer;
    //! [numBuffers] each buffer's index among those sharing its pool.
    std::vector<int32_t> bufferSlotInPool;
    //! One entry per pool of the life cycle: cell size, and buffers drawing on it.
    std::vector<std::pair<uint64_t, int32_t>> pools;
};

// ---------------------------------------------------------------------------
// SparseCacheLease — one request's slot in every sparse arena, and so its cells and the
// metadata describing them.
//
// Taken when the cache joins a Batch, returned on removal, suspend() and close(). Move-only
// and RAII: the pool's free list is restored by the destructor, so a request dropped on any
// path -- including a poisoned teardown -- cannot leak its cells.
// ---------------------------------------------------------------------------
class SparseCellArena;
class SparseSlotPool;

class SparseCacheLease
{
public:
    SparseCacheLease() = default;
    ~SparseCacheLease();

    SparseCacheLease(SparseCacheLease&& other) noexcept;
    SparseCacheLease& operator=(SparseCacheLease&& other) noexcept;
    SparseCacheLease(SparseCacheLease const&) = delete;
    SparseCacheLease& operator=(SparseCacheLease const&) = delete;

    [[nodiscard]] bool empty() const noexcept
    {
        return mPool == nullptr;
    }

    //! The slot held, or kNoSparseLease once released. This is what a Batch publishes for
    //! the row, and it addresses the request's cells and metadata in every arena.
    [[nodiscard]] SparseLeaseIndex index() const noexcept
    {
        return mIndex.value_or(kNoSparseLease);
    }

    //! Return the slot early. Idempotent; the destructor does the same.
    void release() noexcept;

private:
    friend class SparseSlotPool;

    SparseSlotPool* mPool = nullptr;
    std::optional<int> mIndex;
};

// ---------------------------------------------------------------------------
// SparseSlotPool — the arena slots of a manager, as one free list.
//
// One slot serves a request in every sparse life cycle at once: the arenas are sized from
// the same `constraints`, so a slot that exists in one exists in all, and a request is
// seated in all of them or none. Holding the list here rather than per arena is what makes
// that an invariant instead of a coincidence of equal acquisition order.
// ---------------------------------------------------------------------------
class SparseSlotPool
{
public:
    SparseSlotPool() = default;

    explicit SparseSlotPool(int numSlots);

    //! Take a slot, or nullopt when every one is held. A caller sized from `constraints`
    //! cannot see nullopt, since a free row implies a free slot.
    [[nodiscard]] std::optional<SparseCacheLease> acquire();

    [[nodiscard]] int numSlots() const noexcept
    {
        return mNumSlots;
    }

private:
    friend class SparseCacheLease;

    void _release(int index) noexcept;

    int mNumSlots = 0;
    std::vector<int> mFree;
};

// ---------------------------------------------------------------------------
// SparseCellArena — one life cycle's GPU cells, and the metadata that describes them.
//
// A dedicated allocation rather than storage slots, so the life cycle's hot pool group can
// be host memory without the cache following it there. Every request of a life cycle needs
// the same number of cells, so this is a fixed-size-block free list.
//
// One block holds the cells of every lease followed by their metadata. Cells come first so
// that a cell index is relative to the arena's base, which is what the fetch reports and
// what get_sparse_cache_base_address returns for a sparse buffer.
// ---------------------------------------------------------------------------
class SparseCellArena
{
public:
    SparseCellArena() = default;

    //! @param shape     per-request metadata and cell geometry, fixed by configuration,
    //!                   including one entry per pool of the life cycle.
    //! @param maxLeases  concurrent requests this life cycle can serve.
    //!
    //! One region per pool, mirroring the hot pool group: pools partition the life cycle's
    //! buffers by size, and a cell is one buffer, so cells cannot be uniform across them.
    SparseCellArena(SparseCacheInit const& shape, int maxLeases);

    ~SparseCellArena() = default;
    SparseCellArena(SparseCellArena&&) noexcept = default;
    SparseCellArena& operator=(SparseCellArena&&) noexcept = default;
    SparseCellArena(SparseCellArena const&) = delete;
    SparseCellArena& operator=(SparseCellArena const&) = delete;

    [[nodiscard]] bool empty() const noexcept
    {
        return mMemory == nullptr;
    }

    //! Base of one pool's cell region: the base the fetch's indices for a buffer are relative to.
    [[nodiscard]] CUdeviceptr base(PoolIndex pool) const
    {
        return mPoolBase.at(pool);
    }

    [[nodiscard]] uint64_t bufferBytes(PoolIndex pool) const
    {
        return mShape.pools.at(static_cast<size_t>(pool.value())).first;
    }

    //! Cells one lease takes in this pool's region: its buffers times the cells each gets.
    [[nodiscard]] int cellsPerLease(PoolIndex pool) const
    {
        return mShape.pools.at(static_cast<size_t>(pool.value())).second * mShape.cellsPerBuffer;
    }

    //! Device bytes this arena holds: every pool's cell region, the state its leases share,
    //! and one metadata block per lease -- the whole allocation.
    [[nodiscard]] size_t numBytes() const noexcept
    {
        return mNumBytes;
    }

    //! Clear `slot`'s metadata on `stream`, which is what taking it means here: every
    //! ordinal becomes kBadBlockOrdinal and the recency order the identity permutation.
    //! The slot itself comes from the manager's SparseSlotPool, since one serves every
    //! life cycle.
    void initialiseLease(SparseLeaseIndex slot, CUstream stream);

    //! Copy a slot's resident ordinals back to the host. Test and debug use only;
    //! synchronises.
    [[nodiscard]] std::vector<int32_t> readResidentOrdinals(SparseLeaseIndex slot) const;

    //! [numBuffers] the layer each buffer belongs to, as every lease of this life cycle sees it.
    [[nodiscard]] int32_t const* bufferLayer() const noexcept
    {
        return mSharedLayer;
    }

    //! How a kernel addresses this arena's leases. Constant once constructed, so a launch
    //! carries one of these rather than one descriptor per row.
    [[nodiscard]] SparseArenaView view() const noexcept;

    [[nodiscard]] SparseCacheInit const& shape() const noexcept
    {
        return mShape;
    }

private:
    CudaUniqPtr<> mMemory;
    SparseCacheInit mShape;
    //! Slots this arena was sized for, which every SparseSlotPool index must fall inside.
    int mMaxLeases = 0;
    size_t mMetadataBytes = 0;
    size_t mNumBytes = 0;
    TypedVec<PoolIndex, CUdeviceptr> mPoolBase;
    std::byte* mMetadataBase = nullptr;

    //! Shared by every lease of the life cycle, so a resume copies none of them.
    int32_t* mSharedLayer = nullptr;
    int32_t* mSharedSlotInPool = nullptr;
    int32_t* mIdentity = nullptr;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
