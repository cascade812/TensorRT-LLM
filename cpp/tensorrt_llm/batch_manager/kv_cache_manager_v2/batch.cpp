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

#include "kv_cache_manager_v2/batch.h"

#include "kv_cache_manager_v2/batchedPageCopy.h"

#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/indexConversionKernels.h"
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/sparseCacheKernels.h"
#include "kv_cache_manager_v2/storageManager.h"
#include "kv_cache_manager_v2/utils/math.h"

#include "tensorrt_llm/common/cudaUtils.h"

#include <algorithm>
#include <numeric>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

//! Largest batch and longest sequence the declared constraints describe, or zero when none
//! were declared.
std::pair<int, int> deriveBounds(KVCacheManagerConfig const& config)
{
    int rows = 0;
    int capacity = 0;
    for (auto const& desc : config.constraints)
    {
        rows = std::max(rows, static_cast<int>(desc.kvCaches.size()));
        for (auto const& kvCache : desc.kvCaches)
        {
            capacity = std::max(capacity, kvCache.capacity);
        }
    }
    return {rows, capacity == 0 ? 0 : divUp(capacity, config.tokensPerBlock)};
}

} // namespace

Batch::Batch(KvCacheManager& manager, CUstream stream, std::vector<IndexTableSpec> indexTables,
    std::optional<int> maxBatchSize, std::optional<int> maxBlocksPerSeq, std::optional<PageIndexMode> mode,
    std::optional<int> maxTotalInputTokens)
    : mManager(&manager)
    , mCudaStream(stream)
    , mMaxTotalInputTokens(maxTotalInputTokens)
{
    if (maxTotalInputTokens.has_value() && *maxTotalInputTokens <= 0)
    {
        throw std::invalid_argument("Batch: maxTotalInputTokens must be positive when given");
    }
    auto const [derivedRows, derivedBlocks] = deriveBounds(manager.config());
    mMaxBatchSize = maxBatchSize.value_or(derivedRows);
    mMaxBlocksPerSeq = maxBlocksPerSeq.value_or(derivedBlocks);
    if (mMaxBatchSize <= 0 || mMaxBlocksPerSeq <= 0)
    {
        throw std::invalid_argument(
            "Batch needs a row and block bound; pass them explicitly when the manager declares no constraints");
    }

    mRows.assign(static_cast<size_t>(mMaxBatchSize), nullptr);

    // Prefer SHARED: one table per life cycle rather than one per layer. The manager
    // answers from configuration alone, so this cannot drift with what a member is doing.
    mIndexModes = TypedVec<LifeCycleId, PageIndexMode>(mManager->numLifeCycles(), PageIndexMode::PER_LAYER);
    mIndexModeNamed = mode.has_value();
    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        mIndexModes[lcId]
            = mode.value_or(manager.supportsIndexMode(PageIndexMode::SHARED, lcId) == std::optional<bool>(true)
                    ? PageIndexMode::SHARED
                    : PageIndexMode::PER_LAYER);
    }

    // Page tables and watermarks are published together, so a kernel cannot read one step's
    // table beside another step's lengths -- an out-of-range read rather than a stale
    // answer.
    auto const elements = _totalElements();
    mStaging = std::make_unique<StaticHostMem>(_totalBytes());
    auto* staging = reinterpret_cast<int32_t*>(mStaging->address());
    // A row that no member occupies must still read as "no such block" rather than as slot
    // zero, since the fetch has no other way to tell them apart.
    std::fill(staging, staging + _historyOffset(), kBadPageIndex.value());
    // Watermarks of a free row are zero: a request with nothing eligible.
    std::fill(staging + _historyOffset(), staging + elements, 0);

    // Views start default-constructed, so an unpublished batch reads as having no caches.
    auto* views = reinterpret_cast<SparseCacheView*>(reinterpret_cast<std::byte*>(staging) + _viewsOffsetBytes());
    auto const numViews = static_cast<size_t>(mManager->numLifeCycles().value()) * static_cast<size_t>(mMaxBatchSize);
    std::fill(views, views + numViews, SparseCacheView{});

    // The staging mirrors the device block exactly, so publishing it is one linear copy.
    // It also doubles what a Batch costs in device memory, which is the price of letting
    // the host-to-device crossing overlap the previous step; a caller that already holds a
    // buffer of this shape could lend it instead.
    mDevice = allocateDeviceMemory(_totalBytes());
    mDeviceStaging = allocateDeviceMemory(_totalBytes());
    auto const device = reinterpret_cast<CUdeviceptr>(mDevice.get());
    auto const deviceStaging = reinterpret_cast<CUdeviceptr>(mDeviceStaging.get());

    for (auto const base : {device, deviceStaging})
    {
        cuCheck(cuMemcpyHtoDAsync(base, staging, _totalBytes(), cudaStream()));
    }

    _allocateStepScratch();
    _allocateFlushRanges();

    // After the modes: a request's offsets depend on where its life cycle puts the base.
    _buildIndexTables(indexTables);
    _checkSparseRunsAreAddressable();
    _allocateIndexTables();

    // Last: a descriptor names where in the index tables a buffer's converted indices sit.
    _allocateCopyDesc();

    // Everything above is staged on the Batch's stream. A consumer reading this Batch from
    // another stream has nothing ordering it against that, so the state is made visible here.
    cuCheck(cuStreamSynchronize(cudaStream()));
}

void Batch::_allocateCopyDesc()
{
    // One descriptor per buffer, serving both directions, for every life cycle. Small
    // enough to share an allocation and a lifetime.
    std::vector<SparseCopyDesc> host;
    std::vector<size_t> offset(static_cast<size_t>(mManager->numLifeCycles().value()), 0);

    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        auto& scratch = mScratch[lcId];
        if (scratch.selectionWidth == 0)
        {
            continue;
        }
        auto const runs = mManager->_sparseCopyTemplates(lcId);
        auto const plan = mManager->_sparseBufferPlan(lcId);
        scratch.numBuffers = static_cast<int32_t>(plan.size());
        offset[static_cast<size_t>(lcId.value())] = host.size();

        for (auto const& run : runs)
        {
            // A run is consecutive buffers of one pool, which can span layers: a pool holds
            // every layer's buffer of one size. An index table spec covers one layer's
            // run of a pool (sparseIndexTableSpecs), so each buffer names its own table,
            // already offset to its entry, and the base that table's indices are relative to.
            for (int32_t i = 0; i < run.numBuffers; ++i)
            {
                auto const buffer = run.bufferBase + i;
                auto const& id = plan[static_cast<size_t>(buffer)].id;
                auto const table = sparseIndexRun(id.layerId, id.role);
                SparseCopyDesc desc;
                desc.hostBase = static_cast<CUdeviceptr>(table.base);
                desc.hostPageStride = run.hostPageStride;
                // Pre-offset to this buffer's entry, so the kernel indexes the row alone.
                desc.pageTable = reinterpret_cast<int32_t const*>(table.table);
                desc.pageTableStride = static_cast<uint32_t>(table.rowStride);
                // Not the row stride: a row now spans every entry of the table, while the
                // blocks this run may address are still one row's worth of one entry.
                desc.numBlocks = static_cast<uint32_t>(mMaxBlocksPerSeq);
                // The buffer's own slot in the pool, folded in so the kernel need not read
                // it through a view it loads per work item.
                desc.deviceBase = run.deviceBase
                    + static_cast<uint64_t>(
                          mManager->_sparseArena(lcId).shape().bufferSlotInPool[static_cast<size_t>(buffer)])
                        * run.cellsPerBuffer * run.bufferBytes;
                desc.leaseStride = static_cast<uint32_t>(run.cellsPerLease) * run.bufferBytes;
                desc.cellBytes = run.bufferBytes;
                desc.layout = run.layout;
                desc.width = run.width;

                auto const width = static_cast<uint64_t>(sparseCopyWidthBytes(desc.width));
                TLLM_CHECK_WITH_INFO(desc.hostBase % width == 0 && desc.deviceBase % width == 0,
                    "A sparse buffer's bases must be aligned to the %llu bytes its layout copies at a time; host "
                    "base %llu, device base %llu",
                    static_cast<unsigned long long>(width), static_cast<unsigned long long>(desc.hostBase),
                    static_cast<unsigned long long>(desc.deviceBase));
                host.push_back(desc);
            }
        }
    }

    if (host.empty())
    {
        return;
    }
    mCopyDesc = allocateDeviceMemory<SparseCopyDesc>(host.size());
    auto const* device = mCopyDesc.get();
    cuCheck(cuMemcpyHtoDAsync(
        reinterpret_cast<CUdeviceptr>(device), host.data(), sizeof(SparseCopyDesc) * host.size(), cudaStream()));
    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        auto& scratch = mScratch[lcId];
        if (scratch.selectionWidth != 0)
        {
            scratch.copyDesc = device + offset[static_cast<size_t>(lcId.value())];
        }
    }
    // `host` does not outlive this scope, so the copy out of it has to have landed.
    cuCheck(cuStreamSynchronize(cudaStream()));
}

int32_t Batch::_flushCapacity(LifeCycleId lifeCycle) const
{
    // A row advancing `t` tokens crosses at most `t / unitTokens + 1` units -- the `+1` for
    // the partial unit its watermark starts inside -- so the batch crosses at most
    // `maxTotalInputTokens / unitTokens + maxBatchSize`. The bound is an aggregate one, and
    // usable only because the list is compacted: indexed by row, every row's slice would
    // have to allow for that row taking the whole budget.
    int const perPage = mManager->sparseCacheUnitsPerPage(lifeCycle);
    int const unitTokens = perPage > 0 ? mManager->config().tokensPerBlock / perPage : 0;
    if (mMaxTotalInputTokens.has_value() && unitTokens > 0)
    {
        return *mMaxTotalInputTokens / unitTokens + mMaxBatchSize;
    }
    // Without a budget, a step is assumed to advance capacity by the in-flight headroom in
    // every row at once, which is what a decode step does.
    return mMaxBatchSize * mManager->_sparseCacheMaxInFlightUnits(lifeCycle);
}

void Batch::_allocateStepScratch()
{
    mScratch = TypedVec<LifeCycleId, SparseStepScratch>(mManager->numLifeCycles(), SparseStepScratch{});

    // Sized up front from configuration, so no step has to allocate. Laid out as one block
    // because a per-group allocation would be a dozen tiny ones serving the same lifetime.
    auto const rows = static_cast<size_t>(mMaxBatchSize);
    size_t total = 0;
    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        int const width = mManager->_sparseCacheSelectionWidth(lcId);
        if (mManager->sparseCacheMaxUnits(lcId) <= 0 || width <= 0)
        {
            continue;
        }
        auto& scratch = mScratch[lcId];
        scratch.selectionWidth = width;
        scratch.maxInFlightUnits = mManager->_sparseCacheMaxInFlightUnits(lcId);
        scratch.numLayers = mManager->_sparseCacheLayerCount(lcId);
        auto const layers = static_cast<size_t>(scratch.numLayers);
        // One list for the fetch, holding the worst case of every row missing its whole
        // selection; one per layer for the write-back, each holding the units the step's
        // token budget can carry across a watermark.
        scratch.fetchWork.capacity = static_cast<int32_t>(rows) * width;
        scratch.fetchWork.numLists = 1;
        scratch.flushWork.capacity = _flushCapacity(lcId);
        scratch.flushWork.numLists = scratch.numLayers;
        // A counter per layer even though the fetch's list is one: the list is reused from
        // layer to layer, because a layer's copy has consumed it before the next layer's
        // resolve overwrites it, but the counter has to read zero when that resolve starts.
        // Giving each layer its own lets the whole step's counters be cleared in one go,
        // rather than one clear per layer standing between every resolve and whatever ran
        // before it. Per-layer *lists* would cost the layer count times the list.
        mCounterElements += 2 * layers;
        auto const fetchEntries = 3 * static_cast<size_t>(scratch.fetchWork.capacity);
        auto const flushEntries = layers * 3 * static_cast<size_t>(scratch.flushWork.capacity);
        total += rows * (static_cast<size_t>(width) + 1 + static_cast<size_t>(scratch.maxInFlightUnits));
        total += fetchEntries + flushEntries + rows * layers;
    }
    if (total == 0)
    {
        return;
    }

    mStepScratch = allocateDeviceMemory<int32_t>(total + mCounterElements);
    // Its own allocation: the step's clear covers mCounters, and these have to survive it.
    mHitMiss = allocateDeviceMemory<SparseHitMiss>(1);
    cuCheck(cuMemsetD32Async(
        reinterpret_cast<CUdeviceptr>(mHitMiss.get()), 0U, sizeof(SparseHitMiss) / sizeof(uint32_t), cudaStream()));

    // The counters lead the block so that one clear covers every life cycle's.
    auto* counter = mStepScratch.get();
    mCounters = counter;
    auto* cursor = counter + mCounterElements;
    auto carve = [&cursor](size_t count)
    {
        auto* ptr = cursor;
        cursor += count;
        return ptr;
    };
    auto const carveCounters = [&counter](size_t count)
    {
        auto* ptr = counter;
        counter += count;
        return ptr;
    };
    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        auto& scratch = mScratch[lcId];
        if (scratch.selectionWidth == 0)
        {
            continue;
        }
        auto const width = static_cast<size_t>(scratch.selectionWidth);
        scratch.outSlot = carve(rows * width);
        scratch.missCount = carve(rows);
        scratch.inFlightSlot = carve(rows * static_cast<size_t>(scratch.maxInFlightUnits));

        auto const carveList = [&carve](SparseWorkList& work)
        {
            auto const entries = static_cast<size_t>(work.capacity) * static_cast<size_t>(work.numLists);
            work.row = carve(entries);
            work.ordinal = carve(entries);
            work.slot = carve(entries);
        };
        carveList(scratch.fetchWork);
        carveList(scratch.flushWork);
        auto const layerCount = static_cast<size_t>(scratch.numLayers);
        // The fetch takes one counter per layer against its single list, so a layer names
        // its own by offsetting this; the write-back's counters match its lists one to one.
        scratch.fetchWork.total = carveCounters(layerCount);
        scratch.flushWork.total = carveCounters(layerCount);
        scratch.hitMiss = mHitMiss.get();

        auto const layers = static_cast<size_t>(scratch.numLayers);
        scratch.flushCount = carve(rows * layers);
        // -1 so that "no write-back has run" stays distinguishable from "one ran and found
        // nothing to move", which is what lets a test tell the two apart.
        cuCheck(cuMemsetD32Async(
            reinterpret_cast<CUdeviceptr>(scratch.flushCount), 0xFFFFFFFFU, rows * layers, cudaStream()));
    }
}

SparseStepScratch const& Batch::sparseStepScratch(LifeCycleId lcId) const
{
    if (lcId >= mScratch.size())
    {
        throw std::out_of_range("Batch::sparseStepScratch: no such layer group");
    }
    return mScratch[lcId];
}

void Batch::setCudaStream(CUstream stream)
{
    if (stream == cudaStream())
    {
        return;
    }
    // The Batch's own queued work -- the staging copy, the index conversion, the counter
    // resets -- is ordered on the old stream alone, and a Batch with no members still has
    // it. Each member's move below syncs that member, which is a different question.
    CachedCudaEvent handover(reinterpret_cast<CudaStream>(cudaStream()));
    handover.waitInStream(reinterpret_cast<CudaStream>(stream));
    mCudaStream = stream;
    for (auto const& kvCache : mRows)
    {
        if (kvCache != nullptr)
        {
            // The private form: the public one refuses to move a member at all.
            kvCache->_setCudaStream(stream);
        }
    }
}

Batch::~Batch()
{
    // Both buffers are about to go, and transfers on the upload stream still name them.
    _drain();
    KVCM2_POISON_ON_EXCEPT(
        [this]
        {
            for (auto const& kvCache : mRows)
            {
                if (kvCache != nullptr)
                {
                    removeKvCache(*kvCache);
                }
            }
        });
}

size_t Batch::_rowElements() const noexcept
{
    return static_cast<size_t>(mMaxBatchSize) * static_cast<size_t>(mMaxBlocksPerSeq);
}

PageIndexMode Batch::indexMode(LifeCycleId lifeCycleId) const
{
    if (lifeCycleId.value() < 0 || lifeCycleId >= mManager->numLifeCycles())
    {
        throw std::out_of_range("Batch::indexMode: unknown layer group");
    }
    return mIndexModes[lifeCycleId];
}

void Batch::_buildIndexTables(std::vector<IndexTableSpec> const& specs)
{
    // The offset of a buffer relative to a chosen base, in page-index units. Mirrors the
    // runtime's kv_offset: a base difference divided by the page stride, which must divide
    // exactly or the two buffers do not sit a whole number of pages apart.
    auto pageOffset = [this](BufferId const& buffer, MemAddress base)
    {
        auto const bufferBase = mManager->getMemPoolBaseAddress(buffer.layerId, buffer.role, PageIndexMode::SHARED);
        auto const stride = static_cast<int64_t>(mManager->getPageStride(buffer.layerId, buffer.role));
        TLLM_CHECK_WITH_INFO(stride > 0, "Page stride must be positive to offset a buffer within its slot");
        auto const delta = static_cast<int64_t>(bufferBase) - static_cast<int64_t>(base);
        TLLM_CHECK_WITH_INFO(delta % stride == 0,
            "Buffers sharing an index table must sit a whole number of pages apart; they cannot share one");
        return static_cast<int>(delta / stride);
    };

    for (auto const& spec : specs)
    {
        TLLM_CHECK_WITH_INFO(!spec.buffers.empty(), "An index table spec names no buffers");

        auto const& first = spec.buffers.front();
        auto const lifeCycle = mManager->_bufferAttr(first.layerId, first.role).lifeCycleId;
        // Substitution replaces one layer's page, so every layer of the life cycle can no
        // longer read one table. SHARED is exactly the claim that they can.
        // Derived SHARED has to be provable from configuration, because nothing else would
        // check it. A named one is the caller's claim about this batch's members, which
        // publish() tests against each of them, so it is allowed to outrun what the manager
        // can prove about requests in general.
        bool const mayHoldScratch = !mManager->supportsIndexMode(PageIndexMode::SHARED, lifeCycle).has_value();
        TLLM_CHECK_WITH_INFO(!mayHoldScratch || mIndexModes[lifeCycle] == PageIndexMode::PER_LAYER || mIndexModeNamed,
            "A life cycle that can hold scratch slots needs PER_LAYER index tables; its layers stop sharing one "
            "set of converted indices as soon as a scratch slot is substituted. Name SHARED on the Batch to "
            "assert no member of it will hold scratch there -- as a decode-only batch does -- and publish() "
            "will hold you to it");
        auto const scale = mManager->getPageIndexScale(first.layerId, first.role);
        auto const mode = mIndexModes[lifeCycle];

        // Under SHARED the base carries the per-layer offset, so the spec's own offsets
        // are relative to its first buffer. Under PER_LAYER the base is the pool group's
        // and carries none, so every offset is absolute within the slot. Derived the same
        // way every reader derives it, so the offsets below and the base a caller is
        // handed later cannot disagree.
        auto const base = _specBase(spec);

        std::vector<int> offsets;
        offsets.reserve(spec.buffers.size());
        for (auto const& buffer : spec.buffers)
        {
            auto const& attr = mManager->_bufferAttr(buffer.layerId, buffer.role);
            // The reference converter emits `expansion` entries per block. Nothing in the
            // tree sets tokensPerBlockOverride, and every attention backend refuses an
            // expansion of anything but one, so this follows them rather than building a
            // path no caller can reach.
            TLLM_CHECK_WITH_INFO(attr.expansion == 1,
                "Index conversion does not implement page index expansion, so a buffer with a tokens-per-block "
                "override cannot have converted index tables");
            TLLM_CHECK_WITH_INFO(attr.lifeCycleId == lifeCycle,
                "An index table spec spans life cycles; their slot ids differ, so no one table can serve it");
            TLLM_CHECK_WITH_INFO(mManager->getPageIndexScale(buffer.layerId, buffer.role) == scale,
                "An index table spec spans page index scales; their converted indices differ, so no one table "
                "can serve it");
            offsets.push_back(pageOffset(buffer, base));
        }

        // Specs coincide when their entries do. Under PER_LAYER the table is per layer,
        // so specs for different layers never merge however alike their entries look.
        auto const sameTable = [&](IndexTable const& table)
        {
            if (table.lifeCycle != lifeCycle || table.scale != scale || table.concatOffsets != offsets)
            {
                return false;
            }
            return mode == PageIndexMode::SHARED || table.served.front().buffers.front().layerId == first.layerId;
        };

        auto found = std::find_if(mIndexTables.begin(), mIndexTables.end(), sameTable);
        if (found == mIndexTables.end())
        {
            mIndexTables.push_back(IndexTable{{}, lifeCycle, scale, offsets});
            found = std::prev(mIndexTables.end());
        }
        IndexTableId const tableId{static_cast<int>(std::distance(mIndexTables.begin(), found))};
        found->served.push_back(spec);

        for (size_t slot = 0; slot < spec.buffers.size(); ++slot)
        {
            auto const& buffer = spec.buffers[slot];
            BufferSlot const resolved{tableId, static_cast<int>(slot)};
            auto const [it, inserted] = mBufferSlots.emplace(buffer, resolved);
            // The base is not compared because it cannot differ here: two specs reach one
            // table only when their concatOffsets agree, and an offset is measured from
            // the spec's own base, so equal offsets at equal entries mean equal bases.
            TLLM_CHECK_WITH_INFO(inserted || (it->second.table == resolved.table && it->second.entry == resolved.entry),
                "A buffer appears in two index table specs that resolved differently. Specs may "
                "not overlap: name a buffer in one spec only, and read a subset of a table by "
                "slicing it at the entry indexTableSlot() gives that buffer");
        }
    }
}

void Batch::_allocateFlushRanges()
{
    mFlushedHistory.assign(static_cast<size_t>(mMaxBatchSize), 0);

    bool anySparse = false;
    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        anySparse = anySparse || mManager->sparseCacheMaxUnits(lcId) > 0;
    }
    if (!anySparse)
    {
        return;
    }

    auto const elements
        = 2 * static_cast<size_t>(mMaxBatchSize) * static_cast<size_t>(mManager->numLifeCycles().value());
    mFlushRanges = allocateDeviceMemory<int32_t>(elements);
    // An empty range asks for nothing, which is what a row that never flushes should read.
    cuCheck(cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(mFlushRanges.get()), 0, elements, cudaStream()));
    mFlushRangeStaging.assign(elements, 0);
}

int32_t const* Batch::flushFromUnits(LifeCycleId lifeCycle) const
{
    auto* base = reinterpret_cast<int32_t*>(mFlushRanges.get());
    return base + 2 * static_cast<size_t>(mMaxBatchSize) * static_cast<size_t>(lifeCycle.value());
}

int32_t const* Batch::flushToUnits(LifeCycleId lifeCycle) const
{
    return flushFromUnits(lifeCycle) + mMaxBatchSize;
}

int Batch::flushedHistory(int row) const
{
    if (row < 0 || row >= mMaxBatchSize)
    {
        throw std::out_of_range("Batch::flushedHistory: row is beyond max_batch_size");
    }
    return mFlushedHistory[static_cast<size_t>(row)];
}

void Batch::flushSparseUnits(std::vector<int> const& newHistoryLengths)
{
    if (newHistoryLengths.size() != mRows.size())
    {
        throw std::invalid_argument("Batch::flushSparseUnits: one history length per row");
    }
    if (!mFlushRanges)
    {
        // No sparse life cycle, so nothing is ever in flight and the record is vacuous.
        for (size_t row = 0; row < mRows.size(); ++row)
        {
            mFlushedHistory[row] = newHistoryLengths[row];
        }
        return;
    }

    // Every row is checked before any work is issued, so a move no reservation could cover
    // leaves the batch as it was.
    bool anyWork = false;
    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        int const unitsPerPage = mManager->sparseCacheUnitsPerPage(lcId);
        if (mManager->sparseCacheMaxUnits(lcId) <= 0 || unitsPerPage <= 0)
        {
            continue;
        }
        int const unitTokens = mManager->config().tokensPerBlock / unitsPerPage;
        auto const base = static_cast<size_t>(2 * mMaxBatchSize * lcId.value());

        // Bounded in aggregate rather than per row, which is what the compacted list allows
        // and what in-flight batching needs: one prefilling row may cross the whole step's
        // worth of units while the rest cross none.
        int64_t crossed = 0;

        for (size_t row = 0; row < mRows.size(); ++row)
        {
            auto const& kvCache = mRows[row];
            int fromUnit = 0;
            int toUnit = 0;
            if (kvCache != nullptr)
            {
                if (newHistoryLengths[row] < kvCache->historyLength())
                {
                    throw std::invalid_argument("Batch::flushSparseUnits: history length cannot be decreased");
                }
                // Clamped to the capacity this row starts from: a unit at or above it was
                // never in flight, so it has no cell and nothing to write back.
                fromUnit = kvCache->historyLength() / unitTokens;
                toUnit = std::min(newHistoryLengths[row], kvCache->capacity()) / unitTokens;
                toUnit = std::max(toUnit, fromUnit);

                crossed += toUnit - fromUnit;
                anyWork = anyWork || toUnit > fromUnit;
            }
            mFlushRangeStaging[base + row] = fromUnit;
            mFlushRangeStaging[base + static_cast<size_t>(mMaxBatchSize) + row] = toUnit;
        }

        if (crossed > mScratch[lcId].flushWork.capacity)
        {
            throw std::invalid_argument(
                "Batch::flushSparseUnits: the batch crossed more units than the step's token budget allows, so "
                "the write-back list could not name them all; construct the Batch with maxTotalInputTokens");
        }
    }

    for (size_t row = 0; row < mRows.size(); ++row)
    {
        mFlushedHistory[row] = newHistoryLengths[row];
    }
    if (!anyWork)
    {
        // Nothing crossed anywhere, so the ranges on the device are stale but unread.
        return;
    }

    auto* const upload = mUploadStream.handle();
    if (!mFlushRead.isClosed())
    {
        // The previous step's write-back read these same ranges; its reads must retire
        // before the copy below overwrites them.
        mFlushRead.waitInStream(reinterpret_cast<CudaStream>(upload));
        mFlushRead.close();
    }
    detail::copyFlushRangesToDevice(reinterpret_cast<CUdeviceptr>(mFlushRanges.get()), mFlushRangeStaging.data(),
        mFlushRangeStaging.size(), upload);
    auto staged = mUploadStream.recordEvent();
    staged.waitInStream(reinterpret_cast<CudaStream>(cudaStream()));

    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        if (mManager->sparseCacheMaxUnits(lcId) <= 0)
        {
            continue;
        }
        writeBackSparseUnits(lcId, cudaStream());
    }
    mFlushRead = CachedCudaEvent{reinterpret_cast<CudaStream>(cudaStream())};
}

void Batch::_checkSparseRunsAreAddressable() const
{
    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        auto const plan = mManager->_sparseBufferPlan(lcId);
        if (plan.empty())
        {
            continue;
        }
        for (size_t begin = 0; begin < plan.size();)
        {
            size_t end = begin;
            while (end < plan.size() && plan[end].poolIndex == plan[begin].poolIndex
                && plan[end].layerRow == plan[begin].layerRow)
            {
                ++end;
            }

            auto const slotOf = [this](SparseBufferPlan const& entry)
            {
                auto const found = mBufferSlots.find(entry.id);
                TLLM_CHECK_WITH_INFO(found != mBufferSlots.end(),
                    "A sparse buffer has no index table spec; the fetch reads converted indices, so every "
                    "sparse buffer needs one");
                return found->second;
            };

            auto const baseOf = [this](SparseBufferPlan const& entry)
            { return _specBase(_specOf(entry.id.layerId, entry.id.role, "Batch::_checkSparseRunsAreAddressable")); };

            auto const first = slotOf(plan[begin]);
            auto const firstBase = baseOf(plan[begin]);
            for (size_t i = begin + 1; i < end; ++i)
            {
                auto const slot = slotOf(plan[i]);
                // Both, because neither implies the other: a merged table serves several
                // specs, each with its own base, so one table is not one base pointer.
                TLLM_CHECK_WITH_INFO(slot.table == first.table && baseOf(plan[i]) == firstBase,
                    "Sparse buffers sharing a pool are copied in one launch, from one base pointer, so they must "
                    "share one index table spec");
                TLLM_CHECK_WITH_INFO(slot.entry == first.entry + static_cast<int>(i - begin),
                    "Sparse buffers sharing a pool must sit at consecutive entries of their request, in the order "
                    "the pool holds them");
            }
            begin = end;
        }
    }
}

void Batch::_allocateIndexTables()
{
    if (mIndexTables.empty())
    {
        return;
    }

    // Scratch-capable life cycles sort last so that one launch covers the plain tables and
    // one the substituting ones, rather than a kernel branching per table.
    auto const mayHoldScratch = [this](IndexTable const& table)
    { return !mManager->supportsIndexMode(PageIndexMode::SHARED, table.lifeCycle).has_value(); };

    std::vector<size_t> order(mIndexTables.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::ranges::stable_partition(order, [&](size_t index) { return !mayHoldScratch(mIndexTables[index]); });

    std::vector<IndexTable> sorted;
    std::vector<int> remap(mIndexTables.size());
    sorted.reserve(mIndexTables.size());
    for (size_t position = 0; position < order.size(); ++position)
    {
        remap[order[position]] = static_cast<int>(position);
        sorted.push_back(std::move(mIndexTables[order[position]]));
    }
    mIndexTables = std::move(sorted);
    for (auto& [key, slot] : mBufferSlots)
    {
        slot.table = IndexTableId{remap[static_cast<size_t>(slot.table.value())]};
    }
    mNumPlainTables = static_cast<int>(std::count_if(
        mIndexTables.begin(), mIndexTables.end(), [&](IndexTable const& table) { return !mayHoldScratch(table); }));

    // The most slots a request could need for one life cycle, which is the stride every
    // (row, life cycle) of the scratch staging gets. Over this batch's own tables rather
    // than every buffer the manager has: a batch that declared none stages nothing, which
    // is what leaving this zero says.
    //
    // Not floored at one. Zero is a real answer -- a step that cannot outrun the window
    // adds no block that is already stale -- and staging for it would allocate a slot no
    // request can reach, while giving a wrong bound the slack to go unnoticed.
    for (size_t index = static_cast<size_t>(mNumPlainTables); index < mIndexTables.size(); ++index)
    {
        auto const& first = mIndexTables[index].served.front().buffers.front();
        // The step bound only where the caller gave one: unset, mMaxTotalInputTokens
        // stands in for a decode step, which would under-count a batch that then prefills.
        auto const slots = mManager->_scratchSlotsFor(first, mMaxBlocksPerSeq, mMaxTotalInputTokens);
        mMaxScratchSlots = std::max(mMaxScratchSlots, slots);
    }

    // Scratch capability is the manager's answer for the life cycle, while the bound above
    // is this batch's; a batch naming SHARED over an undecided life cycle has the first
    // without the second. With no slot any member can reach there is nothing to substitute,
    // so those tables take the plain path -- which is also what keeps the scratch kernel
    // from running over staging that the bound just decided not to allocate.
    if (mMaxScratchSlots == 0)
    {
        mNumPlainTables = static_cast<int>(mIndexTables.size());
    }

    if (mMaxScratchSlots > 0)
    {
        auto const elements = _scratchElements();
        mScratchStaging = std::make_unique<StaticHostMem>(sizeof(int32_t) * elements);
        auto* host = reinterpret_cast<int32_t*>(mScratchStaging->address());
        // An empty range leaves every block on the plain path, which is what a member
        // holding no scratch should look like.
        std::fill(host, host + elements, 0);
        mScratchDevice = allocateDeviceMemory<int32_t>(elements);
        cuCheck(cuMemcpyHtoDAsync(
            reinterpret_cast<CUdeviceptr>(mScratchDevice.get()), host, sizeof(int32_t) * elements, cudaStream()));
    }

    auto const rowBlocks = static_cast<size_t>(mMaxBatchSize) * static_cast<size_t>(mMaxBlocksPerSeq);
    size_t elements = 0;
    for (auto& table : mIndexTables)
    {
        table.elementOffset = elements;
        elements += table.concatOffsets.size() * rowBlocks;
        mMaxTableEntries = std::max(mMaxTableEntries, static_cast<int>(table.concatOffsets.size()));
    }

    mIndexTableMemory = allocateDeviceMemory<int32_t>(elements);
    auto const memory = reinterpret_cast<CUdeviceptr>(mIndexTableMemory.get());
    // A table a publish has not reached yet must read as absent rather than as slot zero.
    cuCheck(cuMemsetD32Async(memory, static_cast<unsigned int>(kBadPageIndex.value()), elements, cudaStream()));

    // The descriptors and the offsets they point at share one allocation: the offsets are a
    // handful of ints per table and would otherwise be an allocation each.
    size_t offsetElements = 0;
    for (auto const& table : mIndexTables)
    {
        offsetElements += table.concatOffsets.size();
    }
    size_t const descBytes = sizeof(IndexTableDesc) * mIndexTables.size();
    size_t const offsetsBytes = sizeof(int32_t) * offsetElements;

    mIndexTableDescs = allocateDeviceMemory(descBytes + offsetsBytes);
    auto const descs = reinterpret_cast<CUdeviceptr>(mIndexTableDescs.get());

    std::vector<int32_t> offsetHost;
    offsetHost.reserve(offsetElements);
    std::vector<IndexTableDesc> descHost;
    descHost.reserve(mIndexTables.size());

    auto const offsetsBase = descs + descBytes;
    auto const rawBase = reinterpret_cast<CUdeviceptr>(mDeviceStaging.get());
    for (auto const& table : mIndexTables)
    {
        IndexTableDesc desc;
        // The raw slot ids of this table's life cycle, in the staging the upload filled.
        desc.raw = reinterpret_cast<int32_t const*>(
            rawBase + sizeof(int32_t) * _rowElements() * static_cast<size_t>(table.lifeCycle.value()));
        desc.converted = reinterpret_cast<int32_t*>(memory + sizeof(int32_t) * table.elementOffset);
        desc.offsets = reinterpret_cast<int32_t const*>(offsetsBase + sizeof(int32_t) * offsetHost.size());
        desc.numBuffers = static_cast<int32_t>(table.concatOffsets.size());
        desc.scale = table.scale;
        if (static_cast<int>(descHost.size()) >= mNumPlainTables)
        {
            auto const row = static_cast<size_t>(mMaxBatchSize);
            auto const lc = static_cast<size_t>(table.lifeCycle.value());
            auto const begs = reinterpret_cast<CUdeviceptr>(mScratchDevice.get());
            auto const groups = static_cast<size_t>(mManager->numLifeCycles().value());
            desc.scratchBegs = reinterpret_cast<int32_t const*>(begs + sizeof(int32_t) * lc * row);
            desc.scratchEnds = reinterpret_cast<int32_t const*>(begs + sizeof(int32_t) * (groups * row + lc * row));
            desc.scratchSlots = reinterpret_cast<int32_t const*>(
                begs + sizeof(int32_t) * (2 * groups * row + lc * row * static_cast<size_t>(mMaxScratchSlots)));
            desc.scratchSlotStride = mMaxScratchSlots;
            auto const& firstBuffer = table.served.front().buffers.front();
            desc.scratchPagesPerBlock
                = mManager->getPageIndexConverter(firstBuffer.layerId, firstBuffer.role).scratchPagesPerBlock;
        }
        descHost.push_back(desc);
        offsetHost.insert(offsetHost.end(), table.concatOffsets.begin(), table.concatOffsets.end());
    }

    cuCheck(cuMemcpyHtoDAsync(descs, descHost.data(), descBytes, cudaStream()));
    cuCheck(cuMemcpyHtoDAsync(offsetsBase, offsetHost.data(), offsetsBytes, cudaStream()));
    // Neither host vector outlives this scope, so the copies out of them have to have landed.
    cuCheck(cuStreamSynchronize(cudaStream()));
}

void Batch::_convertIndices(CUstream stream)
{
    if (mIndexTables.empty())
    {
        return;
    }
    auto const* const descs = reinterpret_cast<IndexTableDesc const*>(mIndexTableDescs.get());
    launchIndexConversion(descs, mNumPlainTables, mMaxTableEntries, mMaxBatchSize, mMaxBlocksPerSeq,
        static_cast<int32_t>(mMaxBlocksPerSeq), stream);
    launchIndexConversionWithScratch(descs + mNumPlainTables,
        static_cast<int32_t>(mIndexTables.size()) - mNumPlainTables, mMaxTableEntries, mMaxBatchSize, mMaxBlocksPerSeq,
        static_cast<int32_t>(mMaxBlocksPerSeq), stream);
}

CUdeviceptr Batch::indexTableAddress(IndexTableId tableId) const
{
    if (tableId.value() < 0 || tableId >= numIndexTables())
    {
        throw std::out_of_range("Batch::indexTableAddress: unknown table");
    }
    return reinterpret_cast<CUdeviceptr>(mIndexTableMemory.get())
        + sizeof(int32_t) * mIndexTables[static_cast<size_t>(tableId.value())].elementOffset;
}

ConvertedIndexTable Batch::convertedIndexTable(IndexTableId tableId) const
{
    if (tableId.value() < 0 || tableId >= numIndexTables())
    {
        throw std::out_of_range("Batch::convertedIndexTable: unknown table");
    }
    ConvertedIndexTable view;
    view.data = indexTableAddress(tableId);
    view.numRows = mMaxBatchSize;
    // A table's entries are its requests' buffers: the offsets it converts with, one each.
    view.numBuffers = static_cast<int32_t>(mIndexTables[static_cast<size_t>(tableId.value())].concatOffsets.size());
    view.blocksPerRow = mMaxBlocksPerSeq;
    return view;
}

Batch::SparseIndexRun Batch::sparseIndexRun(LayerId layerId, DataRole const& role) const
{
    auto const& slot = _bufferSlot(layerId, role, "Batch::sparseIndexRun");
    auto const tableId = slot.table;
    SparseIndexRun run;
    // Row-major in the entries, so a row costs the whole table's entries and an entry costs
    // one row of blocks.
    auto const numBuffers = static_cast<int>(mIndexTables[static_cast<size_t>(tableId.value())].concatOffsets.size());
    run.bufferStride = mMaxBlocksPerSeq;
    run.rowStride = numBuffers * mMaxBlocksPerSeq;
    run.table = indexTableAddress(tableId) + sizeof(int32_t) * static_cast<size_t>(slot.entry) * run.bufferStride;
    run.base = _specBase(_specOf(layerId, role, "Batch::sparseIndexRun"));
    return run;
}

size_t Batch::_scratchElements() const noexcept
{
    auto const rows = static_cast<size_t>(mMaxBatchSize);
    auto const groups = static_cast<size_t>(mManager->numLifeCycles().value());
    // [begs][ends][slots], each indexed by life cycle then row.
    return 2 * groups * rows + groups * rows * static_cast<size_t>(mMaxScratchSlots);
}

void Batch::_stageScratch()
{
    if (!mScratchStaging)
    {
        return;
    }
    auto const rows = static_cast<size_t>(mMaxBatchSize);
    auto const groups = static_cast<size_t>(mManager->numLifeCycles().value());
    auto* const begs = reinterpret_cast<int32_t*>(mScratchStaging->address());
    auto* const ends = begs + groups * rows;
    auto* const slots = ends + groups * rows;

    for (size_t row = 0; row < mRows.size(); ++row)
    {
        auto const& kvCache = mRows[row];
        for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
        {
            auto const flat = static_cast<size_t>(lcId.value()) * rows + row;
            auto* const rowSlots = slots + flat * static_cast<size_t>(mMaxScratchSlots);

            // A free row, a suspended member, or one holding no scratch here all leave an
            // empty range, which the kernel reads as "every block takes the plain path".
            auto const desc = kvCache != nullptr && kvCache->isActive() ? kvCache->getScratchDesc(lcId) : std::nullopt;
            if (!desc.has_value())
            {
                begs[flat] = 0;
                ends[flat] = 0;
                continue;
            }
            begs[flat] = desc->range.beg.value();
            ends[flat] = desc->range.end.value();
            // The slot check below is what keeps the copy in bounds. This one tests the
            // bound the sizing was derived from, against a range an actual request
            // reached -- so a mistake in that derivation is caught where it was made,
            // rather than only when some request's conversion happens to need the slot
            // the arithmetic lost.
            TLLM_CHECK_DEBUG_WITH_INFO(desc->range.end.value() - desc->range.beg.value()
                    <= mManager->_maxScratchBlocks(lcId, mMaxBlocksPerSeq, mMaxTotalInputTokens),
                "A request's scratch range outruns the blocks the batch was sized for");
            TLLM_CHECK_WITH_INFO(desc->slotIds.size() <= static_cast<size_t>(mMaxScratchSlots),
                "A request holds more scratch slots than the batch was sized for");
            std::copy(desc->slotIds.begin(), desc->slotIds.end(), rowSlots);
        }
    }
}

IndexTableId Batch::numIndexTables() const noexcept
{
    return IndexTableId{static_cast<int>(mIndexTables.size())};
}

std::vector<IndexTableSpec> const& Batch::indexTable(IndexTableId tableId) const
{
    if (tableId.value() < 0 || tableId >= numIndexTables())
    {
        throw std::out_of_range("Batch::indexTable: unknown table");
    }
    return mIndexTables[static_cast<size_t>(tableId.value())].served;
}

Batch::BufferSlot const& Batch::_bufferSlot(LayerId layerId, DataRole const& role, char const* api) const
{
    auto const found = mBufferSlots.find(BufferId{layerId, role});
    if (found == mBufferSlots.end())
    {
        throw std::out_of_range(std::string(api) + ": no index table spec named this buffer");
    }
    return found->second;
}

IndexTableId Batch::indexTableOf(IndexTableSpec const& spec) const
{
    for (size_t i = 0; i < mIndexTables.size(); ++i)
    {
        for (auto const& served : mIndexTables[i].served)
        {
            if (served.buffers == spec.buffers)
            {
                return IndexTableId{static_cast<int>(i)};
            }
        }
    }
    throw std::out_of_range("Batch::indexTableOf: no index table was built for this spec");
}

IndexTableSpec const& Batch::_specOf(LayerId layerId, DataRole const& role, char const* api) const
{
    auto const& slot = _bufferSlot(layerId, role, api);
    BufferId const wanted{layerId, role};
    for (auto const& served : mIndexTables[static_cast<size_t>(slot.table.value())].served)
    {
        if (std::ranges::find(served.buffers, wanted) != served.buffers.end())
        {
            return served;
        }
    }
    // Unreachable while construction records a slot only for a buffer a spec names.
    throw LogicError(std::string(api) + ": a buffer's table does not serve the spec that named it");
}

MemAddress Batch::_specBase(IndexTableSpec const& spec) const
{
    auto const& first = spec.buffers.front();
    auto const lifeCycle = mManager->_bufferAttr(first.layerId, first.role).lifeCycleId;
    return mManager->getMemPoolBaseAddress(first.layerId, first.role, mIndexModes[lifeCycle]);
}

PoolSpec Batch::poolSpecFor(IndexTableSpec const& spec) const
{
    // Membership first: a spec the Batch never built has no base, and saying so beats
    // handing back the pool of whichever buffer happens to be named.
    static_cast<void>(indexTableOf(spec));

    auto const& first = spec.buffers.front();
    PoolSpec out;
    out.base = _specBase(spec);
    out.pageStride = mManager->getPageStride(first.layerId, first.role);
    for (auto const& buffer : spec.buffers)
    {
        // One stride for the spec, because a reader has only one: the entries carry each
        // buffer's offset in pages, which is a count of this stride and of no other.
        TLLM_CHECK_WITH_INFO(mManager->getPageStride(buffer.layerId, buffer.role) == out.pageStride,
            "An index table spec spans page strides, so one base cannot address its buffers");
    }
    return out;
}

std::pair<IndexTableId, int> Batch::indexTableSlot(LayerId layerId, DataRole const& role) const
{
    auto const& slot = _bufferSlot(layerId, role, "Batch::indexTableSlot");
    return {slot.table, slot.entry};
}

MemAddress Batch::indexTableBase(LayerId layerId, DataRole const& role) const
{
    return _specBase(_specOf(layerId, role, "Batch::indexTableBase"));
}

void Batch::_drain() noexcept
{
    KVCM2_POISON_ON_EXCEPT(
        [this]
        {
            waitPublished();
            if (!mStagingRead.isClosed())
            {
                mStagingRead.synchronize();
                mStagingRead.close();
            }
        });
}

size_t Batch::_historyOffset() const noexcept
{
    return _rowElements() * static_cast<size_t>(mManager->numLifeCycles().value());
}

size_t Batch::_capacityOffset() const noexcept
{
    return _historyOffset() + static_cast<size_t>(mMaxBatchSize);
}

size_t Batch::_totalElements() const noexcept
{
    return _capacityOffset() + static_cast<size_t>(mMaxBatchSize);
}

size_t Batch::_viewsOffsetBytes() const noexcept
{
    // The views hold pointers, so the region starts at their alignment rather than wherever
    // the int32 arrays happen to end.
    size_t const bytes = sizeof(int32_t) * _totalElements();
    size_t const align = alignof(SparseCacheView);
    return (bytes + align - 1) / align * align;
}

size_t Batch::_totalBytes() const noexcept
{
    return _viewsOffsetBytes()
        + sizeof(SparseCacheView) * static_cast<size_t>(mManager->numLifeCycles().value())
        * static_cast<size_t>(mMaxBatchSize);
}

void Batch::_stageRowState()
{
    auto* bytes = reinterpret_cast<std::byte*>(mStaging->address());
    auto* base = reinterpret_cast<int32_t*>(bytes);
    auto* history = base + _historyOffset();
    auto* capacity = base + _capacityOffset();
    auto* views = reinterpret_cast<SparseCacheView*>(bytes + _viewsOffsetBytes());

    for (size_t row = 0; row < mRows.size(); ++row)
    {
        auto const& kvCache = mRows[row];
        history[row] = kvCache == nullptr ? 0 : kvCache->historyLength();
        capacity[row] = kvCache == nullptr ? 0 : kvCache->capacity();

        for (LifeCycleId lc{0}; lc < mManager->numLifeCycles(); ++lc)
        {
            auto& slot = views[static_cast<size_t>(lc.value()) * mRows.size() + row];
            // A free row, or a member holding no reservation for this group, leaves the
            // default view: null pointers the kernel reads as "nothing to resolve here".
            slot = kvCache == nullptr || lc >= kvCache->mSparseCache.size() ? SparseCacheView{}
                                                                            : kvCache->mSparseCache[lc].view();
        }
    }
}

void Batch::_seedRowWatermarks(int row, int historyLength, int capacity)
{
    auto* base = reinterpret_cast<int32_t*>(mStaging->address());
    auto const index = static_cast<size_t>(row);
    base[_historyOffset() + index] = historyLength;
    base[_capacityOffset() + index] = capacity;
}

int32_t* Batch::_stagingRow(LifeCycleId lifeCycleId, int row) const
{
    auto* base = reinterpret_cast<int32_t*>(mStaging->address());
    return base + static_cast<int64_t>(_rowElements()) * lifeCycleId.value()
        + static_cast<int64_t>(mMaxBlocksPerSeq) * row;
}

int Batch::addKvCache(KvCache& kvCache, std::optional<int> requestedRow)
{
    if (kvCache.mBatch != nullptr)
    {
        throw LogicError("A KvCache belongs to at most one Batch");
    }
    if (kvCache.beamWidth() != BeamIndex{1})
    {
        // Only beam 0 is mirrored below, so a wider request would publish one beam's pages
        // and have every beam read them. Refusing beats a wrong answer nothing would notice.
        throw LogicError("Batch does not support beam search yet; a member must have beam width 1");
    }
    if (!kvCache.isActive())
    {
        // Membership means the member is being stepped, which is what lets its arena lease,
        // its row and its active state be one thing rather than three that can disagree.
        throw LogicError("A KvCache must be resumed before it joins a Batch");
    }
    // Binding a row writes it, so the last publish must be done reading it.
    waitPublished();
    int row = 0;
    if (!requestedRow.has_value())
    {
        auto const free = std::find(mRows.begin(), mRows.end(), nullptr);
        if (free == mRows.end())
        {
            throw LogicError("Batch is full; it was sized for max_batch_size rows");
        }
        row = static_cast<int>(std::distance(mRows.begin(), free));
    }
    else
    {
        row = *requestedRow;
        if (row < 0 || row >= mMaxBatchSize)
        {
            throw std::out_of_range("Batch::addKvCache: row is beyond max_batch_size");
        }
        if (mRows[static_cast<size_t>(row)] != nullptr)
        {
            throw LogicError("Batch::addKvCache: row is already occupied");
        }
    }

    // Point every layer group's index table at this row. The bind copies the current
    // indices in and keeps writing there, so the row needs no refresh of its own
    // afterwards -- and keeps doing so until the member leaves, which is why a step that
    // only advances its members has nothing to rebind.
    for (LifeCycleId lc{0}; lc < mManager->numLifeCycles(); ++lc)
    {
        // Beam 0 only; addKvCache rejects anything wider.
        kvCache.setBasePageIndexBuf(kDefaultBeamIndex, lc, _stagingRow(lc, row), mMaxBlocksPerSeq);
    }

    _seedRowWatermarks(row, kvCache.historyLength(), kvCache.capacity());
    // A seating member brings no in-flight units: its lease is fresh, so everything its
    // history names already lives in the pages behind it. Seeding from the history is what
    // lets the row be reused -- the departing member leaves this equal to its own history,
    // and without this the next occupant would inherit that number.
    mFlushedHistory[static_cast<size_t>(row)] = kvCache.historyLength();

    // Joining is a stream change, so it carries the same cross-stream sync a direct move
    // would: work the member already enqueued has to retire before the Batch's stream
    // reaches anything that reads it.
    kvCache._setCudaStream(cudaStream());

    // The arena lease goes with membership: a member is what gets stepped, so this is
    // exactly when the GPU cache is needed. After the stream change, because acquiring one
    // clears its residency on that stream. Sized from the same `constraints` the rows come
    // from, so a free row always has a free lease and this cannot fail.
    for (LifeCycleId lc{0}; lc < mManager->numLifeCycles(); ++lc)
    {
        auto& arena = mManager->_sparseArena(lc);
        if (arena.empty() || !kvCache.mSparseCache[lc].empty())
        {
            continue;
        }
        auto lease = arena.acquire(cudaStream());
        TLLM_CHECK_WITH_INFO(lease.has_value(),
            "Batch::addKvCache: the sparse cache arena is full, which cannot happen while it is sized from the "
            "same `constraints` the Batch takes its rows from");
        kvCache.mSparseCache[lc] = std::move(*lease);
    }

    // Owning from here: the member cannot be destroyed under the Batch, so a caller may
    // keep only the Batch and get the reference back when it leaves.
    mRows[static_cast<size_t>(row)] = kvCache.shared_from_this();
    kvCache.mBatch = this;
    kvCache.mBatchRow = row;
    return row;
}

std::shared_ptr<KvCache> const& Batch::operator[](int row) const
{
    return mRows.at(static_cast<size_t>(row));
}

std::shared_ptr<KvCache> Batch::removeKvCache(int row)
{
    if (row < 0 || row >= mMaxBatchSize)
    {
        throw std::out_of_range("Batch::removeKvCache: row is beyond max_batch_size");
    }
    auto seated = mRows[static_cast<size_t>(row)];
    return seated ? removeKvCache(*seated) : nullptr;
}

std::shared_ptr<KvCache> Batch::removeKvCache(KvCache& kvCache)
{
    if (kvCache.mBatch != this)
    {
        throw LogicError("KvCache is not a member of this Batch");
    }
    waitPublished();
    int const row = *kvCache.mBatchRow;
    // Checked before anything is touched, so a rejected removal leaves the batch as it was.
    //
    // Flushing a history the member never takes up is pointless, so the two agreeing is what
    // a member that has written back everything it means to keep looks like. The leases
    // released below discard the GPU cache, and the next occupant seeds its own watermark
    // from its own history, so anything still pending here has no later chance to be noticed.
    if (std::ranges::any_of(kvCache.mSparseCache, [](SparseCacheLease const& lease) { return !lease.empty(); }))
    {
        TLLM_CHECK_WITH_INFO(kvCache.historyLength() == mFlushedHistory[static_cast<size_t>(row)],
            "A sparse KvCache leaving row %d has history %d against %d written back; call "
            "Batch::flushSparseUnits() with the current history before removing it",
            row, kvCache.historyLength(), mFlushedHistory[static_cast<size_t>(row)]);
    }
    // Held until the end: clearing the row below drops the Batch's reference, and the rest
    // of this runs against the member.
    auto leaving = std::move(mRows[static_cast<size_t>(row)]);
    // Membership ends first, so the unbinding below goes through the same setter a caller
    // would use. Nothing binds a seated member's buffer, including the Batch.
    kvCache.mBatch = nullptr;
    kvCache.mBatchRow = std::nullopt;
    for (LifeCycleId lc{0}; lc < mManager->numLifeCycles(); ++lc)
    {
        // Hands the indices back to the cache's own storage before the row is reused.
        kvCache.setBasePageIndexBuf(kDefaultBeamIndex, lc, nullptr, 0);
        auto* staged = _stagingRow(lc, row);
        std::fill(staged, staged + mMaxBlocksPerSeq, kBadPageIndex.value());
    }
    // Leaving returns the lease, which discards whatever the GPU cache held: a lease comes
    // back with every entry free. Anything worth keeping is written back by
    // flushSparseUnits() before this.
    for (auto& lease : kvCache.mSparseCache)
    {
        lease.release();
    }
    _seedRowWatermarks(row, 0, 0);
    mRows[static_cast<size_t>(row)] = nullptr;
    return leaving;
}

void Batch::move(int dstRow, int srcRow)
{
    if (dstRow < 0 || dstRow >= mMaxBatchSize || srcRow < 0 || srcRow >= mMaxBatchSize)
    {
        throw std::out_of_range("Batch::move: row is beyond max_batch_size");
    }
    // Checked before anything is touched, so a rejected move leaves the batch as it was.
    // dstRow == srcRow fails here too: a row cannot be occupied and empty at once.
    if (mRows[static_cast<size_t>(srcRow)] == nullptr)
    {
        throw LogicError("Batch::move: source row holds no member");
    }
    if (mRows[static_cast<size_t>(dstRow)] != nullptr)
    {
        throw LogicError("Batch::move: destination row is already occupied");
    }

    waitPublished();

    for (LifeCycleId lc{0}; lc < mManager->numLifeCycles(); ++lc)
    {
        auto* const from = _stagingRow(lc, srcRow);
        auto* const to = _stagingRow(lc, dstRow);
        // Swapped rather than copied so the vacated row ends up holding what the empty
        // destination held: every entry kBadPageIndex, which is what a free row must publish.
        std::swap_ranges(from, from + mMaxBlocksPerSeq, to);
    }
    std::swap(mRows[static_cast<size_t>(dstRow)], mRows[static_cast<size_t>(srcRow)]);
    std::swap(mFlushedHistory[static_cast<size_t>(dstRow)], mFlushedHistory[static_cast<size_t>(srcRow)]);

    auto& kvCache = *mRows[static_cast<size_t>(dstRow)];
    kvCache.mBatchRow = dstRow;
    for (LifeCycleId lc{0}; lc < mManager->numLifeCycles(); ++lc)
    {
        // The row's contents have already moved. Rebind its sole writer without copying from
        // the old row or ending membership -- mBatch and the arena lease are untouched.
        kvCache.mBasePageIndices[kDefaultBeamIndex][lc] = Span<int>{_stagingRow(lc, dstRow), mMaxBlocksPerSeq};
    }
    _seedRowWatermarks(dstRow, kvCache.historyLength(), kvCache.capacity());
    _seedRowWatermarks(srcRow, 0, 0);
}

void Batch::_detach(KvCache& kvCache) noexcept
{
    if (kvCache.mBatch != this || !kvCache.mBatchRow.has_value())
    {
        return;
    }
    // A seated member holds a reference from mRows, so its destructor cannot run while it
    // is one. Reaching here means a row was cleared without unbinding the member, leaving a
    // row that publishes from an object about to stop existing.
    KVCM2_CHECK_FATAL_WITH_INFO(
        false, "A KvCache was destroyed while seated in a Batch, so its row outlived the cache it names");
}

std::vector<uint8_t> Batch::resize(
    std::vector<std::optional<int>> const& capacities, std::vector<std::optional<int>> const& historyLengths)
{
    if (capacities.size() != mRows.size() || historyLengths.size() != mRows.size())
    {
        throw std::invalid_argument("Batch::resize: one capacity and history length per row");
    }

    // Every row is checked before any is touched, so a bad request leaves the batch as it
    // was rather than part-way through.
    for (size_t row = 0; row < mRows.size(); ++row)
    {
        auto const& kvCache = mRows[row];
        if (kvCache == nullptr || !historyLengths[row].has_value())
        {
            continue;
        }
        int const newHist = *historyLengths[row];
        if (newHist < kvCache->historyLength())
        {
            throw std::invalid_argument("Batch::resize: history length cannot be decreased");
        }
        // Against the capacity that already existed, not the one being asked for. Capacity
        // this call grants makes room for tokens written after it, so history can only
        // cover tokens whose space was there while they were being written.
        if (newHist > kvCache->capacity())
        {
            throw std::invalid_argument("Batch::resize: history length cannot exceed the capacity the row already had");
        }
    }

    // Before the first member is touched, since each resize writes its staging row.
    waitPublished();

    std::vector<uint8_t> results(mRows.size(), 0);
    for (size_t row = 0; row < mRows.size(); ++row)
    {
        auto const& kvCache = mRows[row];
        if (kvCache == nullptr)
        {
            continue;
        }
        if (kvCache->resize(capacities[row], historyLengths[row]))
        {
            results[row] = 1;
            continue;
        }
        // The growth failed and rolled itself back, but the history advance is still both
        // legal -- it was checked against the capacity this row kept -- and free: the stale
        // range and the scratch range are each monotone in history, so raising it can only
        // release slots. Taking it anyway is what lets the units below it be written back
        // and their cells released, which is the pressure that made the growth fail.
        if (capacities[row].has_value() && historyLengths[row].has_value())
        {
            kvCache->resize(std::nullopt, historyLengths[row]);
        }
    }
    // Once, after every member has settled: a partial failure leaves the rolled-back state,
    // which is what the table should describe.
    publish();
    return results;
}

void Batch::publish()
{
    // A step's work-list counters start here, all of them at once. Per layer and per
    // direction they are independent, so one clear at the head of a step is enough -- and
    // clearing them here rather than before each producer keeps a non-kernel node from
    // standing between every resolve and whatever ran before it, which is what a launch
    // dependency cannot reach across.
    //
    // On the Batch's stream, like the staging below it. A fetch issued on another stream
    // therefore has to be ordered after this call -- but it already did, because it reads
    // the page tables and watermarks this same call stages.
    if (mCounterElements > 0)
    {
        TLLM_CUDA_CHECK(cudaMemsetAsync(
            mCounters, 0, sizeof(int32_t) * mCounterElements, reinterpret_cast<cudaStream_t>(cudaStream())));
    }

    // Refusing rather than waiting, because by this point a wait would be too late to help.
    // Everything that writes the staged rows has to be separated from the copy reading them:
    // the members' own mutations, which happened before this call, and the watermarks and
    // views staged just below. Waiting here would cover only the second, and would do it
    // while quietly hiding that the first had already raced.
    TLLM_CHECK_WITH_INFO(mPublished.isClosed(),
        "Batch::publish: the previous publish is still reading the staged rows. Call waitPublished() before "
        "whatever wrote them -- a direct resize(), a commit(), a beam fork, an SWA recycle, anything that locks "
        "or unlocks a page -- and not merely before this call: moving the wait here silences this check while "
        "leaving that earlier write racing the copy. Batch's own mutators, resize() and move() among them, "
        "already wait.");

    // A SHARED life cycle gives every layer the same converted indices, which only holds
    // while no member substitutes scratch slots there. Prefill acquires those and decode
    // does not, so a derived mode would have been PER_LAYER and reaching here means the
    // phases were mixed after it was fixed. A named SHARED makes this the check that holds
    // the caller to the claim: it is what lets a decode-only batch take the smaller table
    // on a life cycle the manager cannot clear for requests in general.
    // Life cycle outermost so the members are reached only for one the manager left
    // undecided about the mode this batch gave it. PER_LAYER it always allows, and a
    // SHARED it answers `true` for no request can hold scratch in, so either way every
    // member would pass and the rows need not be walked.
    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        if (mManager->supportsIndexMode(mIndexModes[lcId], lcId) == std::optional<bool>(true))
        {
            continue;
        }
        for (auto const& kvCache : mRows)
        {
            TLLM_CHECK_WITH_INFO(kvCache == nullptr || kvCache->supportsIndexMode(mIndexModes[lcId], lcId),
                "Batch::publish: a member holds scratch slots in a SHARED life cycle; its layers no longer share "
                "one converted index table");
        }
    }

    // The members write their own page indices, but not their watermarks; collect those
    // now so the single transfer below carries a matched pair.
    _stageRowState();
    _stageScratch();

    auto const staging = reinterpret_cast<CUdeviceptr>(mDeviceStaging.get());
    auto* const upload = mUploadStream.handle();

    // The fill below overwrites what the previous step's copy read into the final tables.
    // Those run on different streams, so nothing but this event orders them.
    if (!mStagingRead.isClosed())
    {
        mStagingRead.waitInStream(reinterpret_cast<CudaStream>(upload));
        mStagingRead.close();
    }

    // One transfer for the whole table, rather than one per row whose locks moved. The
    // table is large but the copy is a single driver call on the upload stream, overlapping
    // compute; per-row transfers would trade that for a launch apiece, and the driver cost
    // of those dominates the bytes they save. Correctness does not depend on the choice --
    // a narrower refresh would be slower to decide, never wrong.
    cuCheck(cuMemcpyHtoDAsync(staging, reinterpret_cast<void*>(mStaging->address()), _totalBytes(), upload));
    if (mScratchStaging)
    {
        cuCheck(cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(mScratchDevice.get()),
            reinterpret_cast<void*>(mScratchStaging->address()), sizeof(int32_t) * _scratchElements(), upload));
    }
    mPublished = mUploadStream.recordEvent();

    // The staging now holds this step's tables, so carrying them across is all that is
    // left on the Batch's stream -- one copy, because the two layouts match.
    mPublished.waitInStream(reinterpret_cast<CudaStream>(cudaStream()));
    cuCheck(cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(mDevice.get()), staging, _totalBytes(), cudaStream()));
    _convertIndices(cudaStream());
    mStagingRead = CachedCudaEvent{reinterpret_cast<CudaStream>(cudaStream())};
}

void Batch::waitPublished()
{
    if (mPublished.isClosed())
    {
        return;
    }
    mPublished.synchronize();
    // Closing returns the event to the pool and makes every later call free.
    mPublished.close();
}

CoalescedPageTable Batch::coalescedPageTable(LifeCycleId lifeCycleId) const
{
    if (lifeCycleId.value() < 0 || lifeCycleId >= mManager->numLifeCycles())
    {
        throw std::out_of_range("Batch::coalescedPageTable: unknown layer group");
    }
    CoalescedPageTable view;
    view.data = _stagingRow(lifeCycleId, 0);
    view.numRows = mMaxBatchSize;
    view.blocksPerRow = mMaxBlocksPerSeq;
    return view;
}

GpuCoalescedPageTable Batch::gpuCoalescedPageTable(LifeCycleId lifeCycleId) const
{
    if (lifeCycleId.value() < 0 || lifeCycleId >= mManager->numLifeCycles())
    {
        throw std::out_of_range("Batch::gpuCoalescedPageTable: unknown layer group");
    }
    GpuCoalescedPageTable view;
    view.data = reinterpret_cast<CUdeviceptr>(mDevice.get())
        + sizeof(int32_t) * _rowElements() * static_cast<size_t>(lifeCycleId.value());
    view.numRows = mMaxBatchSize;
    view.blocksPerRow = mMaxBlocksPerSeq;
    return view;
}

CUdeviceptr Batch::sparseCacheViews(LifeCycleId lifeCycleId) const
{
    if (lifeCycleId.value() < 0 || lifeCycleId >= mManager->numLifeCycles())
    {
        throw std::out_of_range("Batch::sparseCacheViews: unknown layer group");
    }
    return reinterpret_cast<CUdeviceptr>(mDevice.get()) + _viewsOffsetBytes()
        + sizeof(SparseCacheView) * static_cast<size_t>(lifeCycleId.value()) * static_cast<size_t>(mMaxBatchSize);
}

WatermarkView Batch::watermarks() const
{
    auto const base = reinterpret_cast<CUdeviceptr>(mDevice.get());
    WatermarkView view;
    view.historyLength = base + sizeof(int32_t) * _historyOffset();
    view.capacity = base + sizeof(int32_t) * _capacityOffset();
    view.numRows = mMaxBatchSize;
    return view;
}

SparseHitMiss Batch::peekHitMissCounter() const
{
    if (!mHitMiss)
    {
        return {};
    }
    // A temporary stream, synced alone: the Batch's own stream carries the step, and
    // waiting on that would turn a statistics read into a pipeline stall.
    SparseHitMiss host;
    TemporaryCudaStream temp{{}};
    cuCheck(cuMemcpyDtoHAsync(&host, reinterpret_cast<CUdeviceptr>(mHitMiss.get()), sizeof(host), temp.get()));
    cuCheck(cuStreamSynchronize(temp.get()));
    return host;
}

void Batch::resetHitMissCounter()
{
    if (mHitMiss)
    {
        TLLM_CUDA_CHECK(
            cudaMemsetAsync(mHitMiss.get(), 0, sizeof(SparseHitMiss), reinterpret_cast<cudaStream_t>(cudaStream())));
    }
}

size_t Batch::sparseArenaBytes() const
{
    size_t total = 0;
    for (LifeCycleId lc{0}; lc < mManager->numLifeCycles(); ++lc)
    {
        total += mManager->_sparseArena(lc).numBytes();
    }
    return total;
}

int Batch::numKvCaches() const noexcept
{
    return static_cast<int>(
        std::count_if(mRows.begin(), mRows.end(), [](std::shared_ptr<KvCache> const& row) { return row != nullptr; }));
}

bool Batch::isContiguous() const noexcept
{
    // Nothing occupied past the first free row is the same thing as a prefix, and says so
    // without counting first.
    auto const firstFree = std::find(mRows.begin(), mRows.end(), nullptr);
    return std::none_of(firstFree, mRows.end(), [](std::shared_ptr<KvCache> const& row) { return row != nullptr; });
}

std::optional<int> Batch::rowOf(KvCache const& kvCache) const
{
    return kvCache.mBatch == this ? kvCache.mBatchRow : std::nullopt;
}

void Batch::prepareSparseWriteIndices(
    std::vector<BufferId> const& buffers, int32_t* out, std::optional<CUstream> streamOpt)
{
    CUstream const stream = streamOpt.value_or(cudaStream());
    if (out == nullptr)
    {
        throw std::invalid_argument("prepareSparseWriteIndices: out must be a device pointer");
    }

    auto const span = mManager->_sparseLayerSpan(buffers, "prepareSparseWriteIndices");
    LifeCycleId const lcId = span.lifeCycle;

    LayerGroupId const lgId{lcId.value()};
    auto const& scratch = sparseStepScratch(lgId);
    if (scratch.inFlightSlot == nullptr)
    {
        throw std::invalid_argument("prepareSparseWriteIndices: this layer group carries no sparse cache");
    }

    int const unitsPerPage = mManager->sparseCacheUnitsPerPage(lcId);
    auto const views = reinterpret_cast<SparseCacheView const*>(sparseCacheViews(lgId));
    auto const marks = watermarks();
    auto const numRows = maxBatchSize();

    // A resolve with no selection. It makes the in-flight unit resident -- reusing its entry
    // when a previous step already pinned it -- and does nothing else: an in-flight unit's
    // host page is not written until the watermark passes it, so there is nothing to copy.
    //
    // The selection and miss arrays are therefore left null, which the resolve tolerates only
    // because it records a miss against the copy list under `ordinal < historyUnits`, and an
    // in-flight ordinal never satisfies it. A change that made that write unconditional would
    // fault here rather than quietly copying stale bytes over what the model wrote.
    SparseResolveArgs resolve;
    resolve.views = views;
    resolve.numRows = numRows;
    resolve.layerInGroup = span.layerInGroup;
    resolve.numLayers = mManager->_sparseCacheLayerCount(lcId);
    resolve.entriesPerLayer = mManager->_sparseCacheSubSlotsPerBuffer(lcId) * unitsPerPage;
    resolve.selectionWidth = 0;
    resolve.historyTokens = reinterpret_cast<int32_t const*>(marks.historyLength);
    resolve.capacityTokens = reinterpret_cast<int32_t const*>(marks.capacity);
    resolve.unitTokens = mManager->config().tokensPerBlock / unitsPerPage;
    resolve.inFlightSlot = scratch.inFlightSlot;
    resolve.maxInFlightUnits = scratch.maxInFlightUnits;
    resolve.missCount = scratch.missCount;
    // No selection on this path, so there is nothing to resolve into a slot.
    resolve.selected = nullptr;
    resolve.outSlot = nullptr;
    launchSparseResolve(resolve, stream);

    // The in-flight list stands in for a selection: the resolve writes kNoSparseSlot past the
    // span, so a row with nothing in flight needs no separate signal.
    //
    // Shared the same way the fetch shares, and for the same reason: buffers whose arena
    // indices coincide are resolved once. The two paths have to agree, or a caller would
    // need one rule for reading the fetch's output and another for this one.
    auto const slots = sparseOutSlots(buffers);
    std::vector<bool> written(static_cast<size_t>(span.numBuffers), false);
    for (auto const& copy : mManager->_sparseCopyTemplates(lcId))
    {
        auto const runBase = std::max(copy.bufferBase, span.bufferBase);
        auto const runEnd = std::min(copy.bufferBase + copy.numBuffers, span.bufferBase + span.numBuffers);
        if (runBase >= runEnd)
        {
            continue;
        }
        int const firstSlot = slots[static_cast<size_t>(runBase - span.bufferBase)];
        if (written[static_cast<size_t>(firstSlot)])
        {
            continue;
        }
        for (int32_t buffer = runBase; buffer < runEnd; ++buffer)
        {
            written[static_cast<size_t>(slots[static_cast<size_t>(buffer - span.bufferBase)])] = true;
        }
        SparseOutIndexArgs indices;
        indices.views = views;
        indices.numRows = numRows;
        indices.selected = nullptr;
        indices.outSlot = scratch.inFlightSlot;
        // One entry per request: resize() bounds capacity - history to a single token, so at
        // most one unit is ever in flight. The scratch row is the reservation's headroom,
        // which is wider, hence the separate stride.
        indices.selectionWidth = 1;
        indices.outSlotStride = scratch.maxInFlightUnits;
        indices.out = out;
        // The row stride is the caller's: `out` keeps one entry per buffer, and an entry
        // no buffer maps to is never written.
        indices.outBuffers = span.numBuffers;
        indices.outBase = firstSlot;
        indices.bufferBase = runBase;
        indices.numBuffers = runEnd - runBase;
        indices.cellsPerBuffer = copy.cellsPerBuffer;
        indices.cellsPerLease = copy.cellsPerLease;
        indices.unitsPerPage = unitsPerPage;
        launchSparseOutIndices(indices, stream);
    }
}

void Batch::writeBackSparseUnits(LifeCycleId lifeCycle, CUstream stream)
{
    LayerGroupId const lgId{lifeCycle.value()};
    auto const views = reinterpret_cast<SparseCacheView const*>(sparseCacheViews(lgId));
    auto const numRows = maxBatchSize();

    // Which units each row crossed. One scan covers every (layer, row); a row whose
    // watermark did not move reports none.
    auto const& scratch = sparseStepScratch(lgId);
    SparseFlushListArgs listArgs;
    listArgs.views = views;
    listArgs.numRows = numRows;
    listArgs.numLayers = scratch.numLayers;
    listArgs.fromUnits = flushFromUnits(lifeCycle);
    listArgs.toUnits = flushToUnits(lifeCycle);
    listArgs.work = scratch.flushWork;
    listArgs.outCount = scratch.flushCount;
    launchSparseFlushList(listArgs, stream);

    // One launch for every pool and every layer: the descriptors the Batch built already
    // say where each pool's buffers and index tables are, and a flush list is per layer.
    SparseUnitCopyArgs copy;
    copy.views = views;
    copy.numRows = numRows;
    copy.copyDesc = scratch.copyDesc;
    copy.bufferBase = 0;
    copy.numBuffers = scratch.numBuffers;
    copy.direction = SparseCopyDirection::kWriteBack;
    copy.unitsPerPage = mManager->sparseCacheUnitsPerPage(lifeCycle);
    // The work list is every layer of this row, which is what a write-back crosses at once,
    // unlike a fetch, which resolves one layer across the whole batch.
    copy.work = scratch.flushWork;
    copy.bufferLayer = mManager->_sparseArena(lifeCycle).bufferLayer();
    copy.gridCtas = mManager->_sparseCopyGridCtas(scratch.flushWork.capacity);
    launchSparseUnitCopy(copy, stream);
}

std::vector<int> Batch::sparseOutSlots(std::vector<BufferId> const& buffers) const
{
    auto const span = mManager->_sparseLayerSpan(buffers, "sparseOutSlots");
    auto const runs = mManager->_sparseCopyTemplates(span.lifeCycle);
    auto const& shape = mManager->_sparseArena(span.lifeCycle).shape();

    // What decides whether two buffers sharing a dense table also share an arena index.
    // cellsPerBuffer is one value for the whole shape, so a lease spans the same cells in
    // two pools exactly when they hold the same number of buffers.
    auto geometry = [&](int32_t buffer)
    {
        for (auto const& run : runs)
        {
            if (buffer >= run.bufferBase && buffer < run.bufferBase + run.numBuffers)
            {
                return std::tuple<int, int, int>{static_cast<int>(run.cellsPerLease),
                    static_cast<int>(run.cellsPerBuffer), shape.bufferSlotInPool[static_cast<size_t>(buffer)]};
            }
        }
        throw LogicError("sparseOutSlots: a sparse buffer belongs to no pool run");
    };

    std::vector<std::pair<int, int>> tableEntries;
    std::vector<std::tuple<int, int, int>> geometries;
    std::vector<int> slots;
    slots.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i)
    {
        auto const [tableId, entry] = indexTableSlot(buffers[i].layerId, buffers[i].role);
        auto const key = std::make_pair(tableId.value(), entry);
        auto const geo = geometry(span.bufferBase + static_cast<int32_t>(i));
        auto const found = std::find(tableEntries.begin(), tableEntries.end(), key);
        if (found == tableEntries.end())
        {
            tableEntries.push_back(key);
            geometries.push_back(geo);
            slots.push_back(static_cast<int>(tableEntries.size()) - 1);
            continue;
        }
        auto const at = static_cast<size_t>(std::distance(tableEntries.begin(), found));
        // The dense tables merge on the hot pool's scale and concat offsets; the arena has
        // its own geometry and has to agree separately. It normally does, but the two are
        // not the same quantity -- a pool carrying only some of the group's layers has the
        // same scale and a different cellsPerLease -- and taking the dense answer there
        // would alias two different cells onto one slice.
        TLLM_CHECK_WITH_INFO(geometries[at] == geo,
            "sparseOutSlots: buffers share an index table entry but not their arena geometry, so one "
            "slice cannot serve both");
        slots.push_back(static_cast<int>(at));
    }
    return slots;
}

void Batch::fetchSparsePages(std::vector<BufferId> const& buffers, int32_t const* selected, int selectionWidth,
    int32_t* out, std::optional<CUstream> streamOpt)
{
    CUstream const stream = streamOpt.value_or(cudaStream());
    if (selected == nullptr || out == nullptr)
    {
        throw std::invalid_argument("fetchSparsePages: selected and out must be device pointers");
    }

    auto const span = mManager->_sparseLayerSpan(buffers, "fetchSparsePages");
    LifeCycleId const lcId = span.lifeCycle;
    auto const bufferBase = span.bufferBase;
    auto const numBuffers = span.numBuffers;
    auto const layerInGroup = span.layerInGroup;

    auto const& scratch = sparseStepScratch(LayerGroupId{lcId.value()});
    if (scratch.outSlot == nullptr)
    {
        throw std::invalid_argument("fetchSparsePages: this layer group carries no sparse cache");
    }
    if (selectionWidth != scratch.selectionWidth)
    {
        throw std::invalid_argument("fetchSparsePages: selectionWidth must match the configured numSelectedBlocks");
    }

    int const unitsPerPage = mManager->sparseCacheUnitsPerPage(lcId);
    LayerGroupId const lgId{lcId.value()};
    auto const views = reinterpret_cast<SparseCacheView const*>(sparseCacheViews(lgId));
    auto const marks = watermarks();
    auto const table = gpuCoalescedPageTable(lgId);
    auto const numRows = maxBatchSize();

    SparseResolveArgs resolve;
    resolve.views = views;
    resolve.numRows = numRows;
    resolve.layerInGroup = layerInGroup;
    resolve.numLayers = mManager->_sparseCacheLayerCount(lcId);
    resolve.entriesPerLayer = mManager->_sparseCacheSubSlotsPerBuffer(lcId) * unitsPerPage;
    resolve.selected = selected;
    resolve.selectionWidth = selectionWidth;
    resolve.historyTokens = reinterpret_cast<int32_t const*>(marks.historyLength);
    resolve.capacityTokens = reinterpret_cast<int32_t const*>(marks.capacity);
    resolve.unitTokens = mManager->config().tokensPerBlock / unitsPerPage;
    resolve.outSlot = scratch.outSlot;
    resolve.inFlightSlot = scratch.inFlightSlot;
    resolve.maxInFlightUnits = scratch.maxInFlightUnits;
    // The list is one, reused from layer to layer; the counter is this layer's own, so a
    // step can clear every layer's at once instead of one clear before each resolve.
    auto layerWork = scratch.fetchWork;
    layerWork.total += layerInGroup;
    resolve.work = layerWork;
    resolve.missCount = scratch.missCount;
    // Every row adds its own hits and misses as it resolves them.
    resolve.hitMiss = scratch.hitMiss;
    launchSparseResolve(resolve, stream);

    // The misses this layer produced, applied to each of its buffers. One list row per
    // batch row, so the inner index is zero once workRowBase names the layer resolved.
    //
    // One launch for every pool of this layer. The descriptors the Batch built already
    // clip each pool to the layer's buffers and name its index table, so nothing here
    // depends on which layer is being resolved beyond choosing the row.
    SparseUnitCopyArgs copy;
    copy.views = views;
    copy.numRows = numRows;
    // The same descriptors as the write-back: a run never spans layers, so it already
    // names this layer's index table. Only the buffer range below differs.
    copy.copyDesc = scratch.copyDesc;
    copy.bufferBase = bufferBase;
    copy.numBuffers = numBuffers;
    copy.direction = SparseCopyDirection::kFetch;
    copy.unitsPerPage = unitsPerPage;
    copy.work = layerWork;
    copy.bufferLayer = mManager->_sparseArena(lcId).bufferLayer();
    copy.gridCtas = mManager->_sparseCopyGridCtas(scratch.fetchWork.capacity);
    launchSparseUnitCopy(copy, stream);

    // The index table the attention kernel reads is derived from the resolve's slots alone:
    // it neither reads a cell nor waits for one to be filled. The copy releases its launch
    // dependency as soon as it is addressed, so this follows it on the stream without
    // waiting for it to finish.
    auto const plan = mManager->_sparseBufferPlan(lcId);
    // Buffers whose indices coincide share a slice, so a run that only repeats slices an
    // earlier one already wrote is not launched at all. A layer's data and scale pools
    // mirror, which is exactly that case.
    auto const slots = sparseOutSlots(buffers);
    std::vector<bool> written(static_cast<size_t>(numBuffers), false);
    for (auto const& pool : mManager->_sparseCopyTemplates(lcId))
    {
        auto const runBase = std::max(pool.bufferBase, bufferBase);
        auto const runEnd = std::min(pool.bufferBase + pool.numBuffers, bufferBase + numBuffers);
        if (runBase >= runEnd)
        {
            continue;
        }
        // A run's buffers are consecutive entries of one table, so their slices are
        // consecutive too and the first one says where the run writes.
        int const firstSlot = slots[static_cast<size_t>(runBase - bufferBase)];
        if (written[static_cast<size_t>(firstSlot)])
        {
            continue;
        }
        for (int32_t buffer = runBase; buffer < runEnd; ++buffer)
        {
            written[static_cast<size_t>(slots[static_cast<size_t>(buffer - bufferBase)])] = true;
        }
        SparseOutIndexArgs indices;
        indices.views = views;
        indices.numRows = numRows;
        indices.selected = selected;
        indices.outSlot = scratch.outSlot;
        indices.selectionWidth = selectionWidth;
        indices.out = out;
        // The row stride is the caller's, not the slice count: `out` keeps one entry per
        // buffer, and a slice no buffer maps to is simply never written.
        indices.outBuffers = numBuffers;
        indices.outBase = firstSlot;
        indices.bufferBase = runBase;
        indices.numBuffers = runEnd - runBase;
        indices.cellsPerBuffer = pool.cellsPerBuffer;
        indices.cellsPerLease = pool.cellsPerLease;
        indices.unitsPerPage = unitsPerPage;
        launchSparseOutIndices(indices, stream);
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
