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
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/sparseCacheKernels.h"
#include "kv_cache_manager_v2/stagingBuffer.h"
#include "kv_cache_manager_v2/utils/cudaEvent.h"
#include "kv_cache_manager_v2/utils/generator.h"
#include "kv_cache_manager_v2/utils/staticHostMem.h"

#include <cuda.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

class KvCache;
class KvCacheManager;

//! One layer group's page table as the device sees it. Always packed: the Batch allocates
//! it, so a row is exactly `blocksPerRow` entries.
struct GpuCoalescedPageTable
{
    CUdeviceptr data = 0;
    int32_t numRows = 0;
    int32_t blocksPerRow = 0;
};

//! One index table's converted page indices as the device sees them, laid out
//! [numRows][numBuffers][blocksPerRow]. A buffer reads the entry `index_table_slot` gives
//! it.
//!
//! Row-major in the entries so a request's buffers sit together: an attention kernel takes
//! a key and value page list as one pair -- XQA's is
//! [batchSize][beamWidth][2][maxNbPagesPerSeq] -- and this is that shape, so a table can be
//! handed over as it stands instead of being repacked every step.
struct ConvertedIndexTable
{
    CUdeviceptr data = 0;
    int32_t numRows = 0;
    //! One per buffer of the requests this table serves; they differ only by the offset
    //! baked into them.
    int32_t numBuffers = 0;
    int32_t blocksPerRow = 0;
};

//! What a converted index means: the address it is relative to, and what it counts.
//!
//! A property of the spec rather than of the table it landed on. Specs whose converted
//! entries coincide are merged onto one table, and reading that one set of entries through
//! different bases is exactly what keeps a quantised layer's data and its block scales
//! apart. So a table has as many of these as it serves specs, and only the spec picks one.
struct PoolSpec
{
    //! What a converted index is added to.
    MemAddress base = 0;
    //! Bytes a converted index counts, which is the pool's page stride.
    size_t pageStride = 0;
};

//! One lifecycle's page table as the host writes it, in the Batch's pinned staging.
struct CoalescedPageTable
{
    //! Points into the pinned staging block, which publish() uploads asynchronously. A host
    //! write here races that copy and is in any case overwritten by the next resize().
    int32_t const* data = nullptr;
    int32_t numRows = 0;
    int32_t blocksPerRow = 0;
};

//! Identifies one converted index table of a Batch.
using IndexTableId = StrongIndex<int, struct IndexTableIdTag>;

//! Device scratch one lifecycle's step works through: what a fetch resolves into before
//! the copy and the index scatter read it back, and what a write-back lists before its own
//! copy reads it.
//!
//! Batch-owned: both consumers are one launch spanning every row, and every extent is a
//! configuration constant, so one allocation serves every step and the step allocates
//! nothing -- which is what lets a CUDA graph capture it.
struct SparseStepScratch
{
    int32_t* outSlot = nullptr;      //!< [numRows][selectionWidth]
    int32_t* missCount = nullptr;    //!< [numRows]
    int32_t* inFlightSlot = nullptr; //!< [numRows][maxInFlightUnits]
    int32_t selectionWidth = 0;
    int32_t maxInFlightUnits = 0;

    //! The misses of one layer, compacted across rows. A fetch resolves one layer at a
    //! time, so a single list serves the whole batch.
    SparseWorkList fetchWork;

    //! The write-back's counterpart. A write-back crosses every layer of a row at once and
    //! the copy that consumes it walks buffers, so it takes one list per layer rather than
    //! the one the fetch folds away.
    SparseWorkList flushWork;
    int32_t* flushCount = nullptr; //!< [numRows][numLayers]
    int32_t numLayers = 0;

    //! The Batch's cumulative fetch statistics, the same pointer for every life cycle.
    //! Carried here so a fetch reaches it the way it reaches the rest of its arguments.
    SparseHitMiss* hitMiss = nullptr;

    //! What differs between the pools of a copy, on the device for the kernel to index by
    //! `blockIdx.z`. Built and uploaded once with the Batch: every field comes from the
    //! buffer plan and the index tables, both fixed at construction, so a captured graph
    //! replays a copy with no host write at all.
    //!
    //! [numBuffers] one copy descriptor per BufferId, serving both directions. BufferIds
    //! are ordered (layer, pool), so one already names its own layer's index-table entry
    //! and needs no layer dimension; the fetch and the write-back then want the same
    //! descriptors and differ only in which of them they launch over.
    SparseCopyDesc const* copyDesc = nullptr;
    int32_t numBuffers = 0;
};

//! Per-row watermarks, in tokens. Everything a step needs about how far a request has got
//! derives from these two, at whatever granularity the consumer works in:
//!
//!     num_blocks     = historyLength / tokens_per_block     (valid page table entries)
//!     history_units  = historyLength / unit_tokens          (selectable units)
//!     capacity_units = capacity / unit_tokens               (the in-flight span's end)
//!
//! Publishing the source rather than the three derived counts keeps one truth; the divisors
//! are per-layer-group constants a kernel already has.
//!
//! A free row reads zero, so it describes a request with nothing eligible.
struct WatermarkView
{
    CUdeviceptr historyLength = 0;
    CUdeviceptr capacity = 0;
    int32_t numRows = 0;
};

// ---------------------------------------------------------------------------
// Batch — the KvCaches stepped together, and their device-side mirror.
//
// Owns the row order and the per-layer-group page tables the sparse fetch reads. Rows are
// slots, not positions in a compacted list: removing a member leaves a hole rather than
// shifting later rows, so array shapes and addresses survive a CUDA graph capture.
//
// Membership is exclusive. A KvCache detaches itself on destruction, so a dropped request
// cannot leave a dangling row.
// ---------------------------------------------------------------------------
class Batch
{
public:
    //! @param indexTables what consumers need addressable together. Requests whose
    //! converted indices coincide are merged onto one table, which is how a quantised
    //! layer's data and block scales come to share one.
    //!
    //! @param maxBatchSize    rows, i.e. the most requests that can be stepped together.
    //! @param maxBlocksPerSeq columns, i.e. the longest sequence a row can describe.
    //!
    //! Both bound device arrays whose shape must not change between steps, so neither can
    //! be grown later. Left unset they derive from the manager's `constraints`, which
    //! throws when none were declared.
    //!
    //! @param mode page index mode, applied to every life cycle. Left unset, each life
    //! cycle takes the smaller table -- SHARED where the manager says it is always
    //! available, PER_LAYER otherwise.
    //!
    //! Set, it overrides that answer rather than being checked against it, and SHARED may
    //! be named where KvCacheManager::supportsIndexMode() returns nullopt. That answer
    //! covers every request the life cycle could carry, so a sliding window leaves it
    //! undecided; naming SHARED is the caller's narrower claim that no member of *this*
    //! batch will hold scratch slots there. A decode-only batch is the case it exists for:
    //! scratch covers the blocks a prefill chunk adds, and a one-token request adds none.
    //!
    //! The claim is validated, not trusted: every publish() checks each member against
    //! KvCache::supportsIndexMode(SHARED, lc) and throws if one holds scratch after all.
    //! Seat only members that keep it -- the throw arrives mid-step, with the batch already
    //! built around the smaller table.
    //!
    //! The mode is fixed here because a converted table's shape is baked into a captured
    //! graph. It cannot follow a member's scratch state, which varies per step.
    //!
    //! @param stream the stream every member runs on: a member's own work and the Batch's
    //! transfers have to be ordered against each other. setCudaStream() moves the Batch and
    //! its members together, which is the only way it changes. The caller keeps it -- a
    //! Batch neither creates nor destroys it, and it must outlive the Batch.
    //!
    //! @param maxTotalInputTokens tokens one step may ingest across every row, as the
    //! runtime's own batching bound gives it. It sizes the write-back's work list: a row
    //! advancing `t` tokens crosses at most `t / unitTokens + 1` units, so the batch
    //! crosses at most `maxTotalInputTokens / unitTokens + maxBatchSize` of them.
    //!
    //! Left unset it falls back to `maxBatchSize * maxInFlightUnits`, which is
    //! `maxBatchSize` -- a decode step advances capacity one token per row, so one row
    //! crosses at most one unit boundary. Pass it once a step can ingest more than that,
    //! which is what chunked prefill under in-flight batching does.
    explicit Batch(KvCacheManager& manager, CUstream stream, std::vector<IndexTableSpec> indexTables,
        std::optional<int> maxBatchSize = std::nullopt, std::optional<int> maxBlocksPerSeq = std::nullopt,
        std::optional<PageIndexMode> mode = std::nullopt, std::optional<int> maxTotalInputTokens = std::nullopt);

    //! The stream the Batch and every member run on.
    [[nodiscard]] CUstream cudaStream() const noexcept
    {
        return mCudaStream;
    }

    //! Moves the Batch and every member onto `stream`.
    //!
    //! The new stream is made to wait on what the old one already has queued, so the
    //! caller need not synchronize: the device arrays and the staging buffer are the same
    //! objects either side of the move, and work that names them stays ordered.
    //!
    //! A KvCache's own stream setter refuses while it is a member.
    //!
    //! Like the constructor, this takes no ownership: the caller's stream must outlive the
    //! Batch, or be replaced here before it is destroyed.
    void setCudaStream(CUstream stream);

    //! Page index mode of one life cycle, as chosen at construction.
    [[nodiscard]] PageIndexMode indexMode(LayerGroupId layerGroupId) const;

    // ---- Converted index tables --------------------------------------------

    [[nodiscard]] IndexTableId numIndexTables() const noexcept;

    //! Every request one table satisfies. A request's buffer order is its own mapping into
    //! the buffer dimension, so the first buffer of each sits at entry zero.
    [[nodiscard]] std::vector<IndexTableSpec> const& indexTable(IndexTableId tableId) const;

    //! Which table serves this spec. The spec must be one the Batch was built with.
    //!
    //! Equivalent specs resolve to the same id -- a layer's data and its block scales name
    //! one table between them. The spec's buffers occupy entries [0, buffers.size()) in its
    //! own order.
    [[nodiscard]] IndexTableId indexTableOf(IndexTableSpec const& spec) const;

    //! How to turn this spec's converted indices into addresses.
    //!
    //! Keyed by the spec because the base is: a merged table serves several specs and each
    //! reads its entries through its own pool. The spec must be one the Batch was built
    //! with, and its buffers must share a page stride -- they sit in one pool, so one base
    //! serves them all.
    //!
    //! The result is stable for the Batch's lifetime, so a captured graph may bake it.
    [[nodiscard]] PoolSpec poolSpecFor(IndexTableSpec const& spec) const;

    //! One table's converted indices: int32[max_batch][entries][blocks].
    //!
    //! A row carries every entry before the next row starts, which is what puts a spec's
    //! key and value planes side by side. So one entry's plane is strided, not contiguous:
    //! it begins at `entry * blocks` elements in and repeats every `entries * blocks`.
    //!
    //! The address is stable for the Batch's lifetime, so a captured graph may bake it.
    [[nodiscard]] ConvertedIndexTable convertedIndexTable(IndexTableId tableId) const;

    ~Batch();

    Batch(Batch const&) = delete;
    Batch& operator=(Batch const&) = delete;

    //! Seat a cache in `row`, or in the lowest free row when none is named.
    //!
    //! Seat a request only once it is ready to be stepped. Seating takes the sparse arena
    //! lease, so a request that holds pages without being stepped -- a disaggregated one
    //! still receiving its KV, or one sitting out a step -- costs the arena nothing while it
    //! stays out. Nothing here checks it: an early seat costs arena pressure, not a throw.
    //!
    //! Throws when the row is taken or out of range, the batch is full, the cache already
    //! belongs to a batch, or its beam width exceeds one -- only beam 0 is mirrored, so a
    //! wider request would have every beam read beam 0's pages.
    int addKvCache(KvCache& kvCache, std::optional<int> row = std::nullopt);

    //! Free the cache's row, leaving a hole, and hand back the reference the Batch held.
    //! Throws when it is not a member.
    //!
    //! Returns the arena lease, which discards whatever the GPU cache held: a lease comes back
    //! with every entry free. Call flushSparseUnits() first for anything worth keeping.
    std::shared_ptr<KvCache> removeKvCache(KvCache& kvCache);

    //! The same, by the row addKvCache returned. Null when nothing is seated there.
    std::shared_ptr<KvCache> removeKvCache(int row);

    //! Move the member at `srcRow` to the empty `dstRow`, leaving `srcRow` free.
    //!
    //! Membership does not lapse: the arena lease, the GPU cache contents behind it and the
    //! stream binding all survive, which is what separates this from removeKvCache() followed
    //! by addKvCache(). That pair would hand the lease back -- a lease returns with every
    //! entry free -- and so would discard residency the member is still entitled to.
    //!
    //! Throws when either row is out of range, when `srcRow` holds nothing, or when `dstRow`
    //! is occupied -- including `dstRow == srcRow`, which is both at once.
    //!
    //! Waits for the previous staging upload before touching either row. Device tables retain
    //! their addresses; the move reaches them on the next publish.
    void move(int dstRow, int srcRow);

    //! Resize every member, then publish once.
    //!
    //! Waits on the previous publish before touching any member. Per-request operations with
    //! no Batch equivalent -- commit(), a beam fork, an SWA recycle -- still need an explicit
    //! waitPublished().
    //!
    //! One result per row, in row order; false for a free row. A member that fails rolls
    //! itself back, so a batch can partially fail and the table published afterwards
    //! describes the post-rollback state.
    //!
    //! `capacities` and `historyLengths` are indexed by row, so both must be maxBatchSize
    //! long. A nullopt leaves that dimension alone, matching KvCache::resize.
    //! Rejects, before touching any member, a history length that falls or that outruns the
    //! capacity its row already had: capacity granted by this call makes room for tokens
    //! written after it, so history can only cover tokens whose space was already there.
    //!
    //! A row whose growth fails still takes its history advance, which is legal by that same
    //! check and needs no slots -- both the stale range and the scratch range are monotone
    //! in history, so raising it only releases. Its result is still 0: the row is not at the
    //! state that was asked for.
    std::vector<uint8_t> resize(
        std::vector<std::optional<int>> const& capacities, std::vector<std::optional<int>> const& historyLengths);

    //! Copy the staged rows to the device, enqueuing on the Batch's stream.
    //!
    //! The staging rows are written by the members themselves, so they are always current;
    //! what this adds is a stream-ordered snapshot. Without it the host would be mutating
    //! the table a previously enqueued fetch is still reading.
    //!
    //! Two stages. The host-to-device fill runs on an upload stream of this Batch's own, so
    //! it can overlap the previous step; the Batch's stream then carries one linear
    //! device-to-device copy into the tables the fetch reads. Everything expensive is in the
    //! first stage, and only the second has to wait for anything.
    //!
    //! **The second stage must be ordered after the kernels that read the previous
    //! publish.** Those live in a captured graph, so the tables cannot be double-buffered
    //! out of the hazard -- the capture baked their addresses, which is why the staging
    //! sits upstream of them instead. The Batch's stream being the one the graph replays on
    //! is what provides that ordering. A torn publish is not a stale answer: the cache
    //! views carry pointers, so it can pair one request's residency with another's lease.
    //!
    //! The copy is asynchronous and reads pinned memory, so it is still in flight when this
    //! returns. **Every later host write to a member's page indices must be separated from
    //! it by waitPublished()** -- a KvCache::resize(), a commit(), a beam fork, an SWA
    //! recycle, anything that locks or unlocks a page. Writing sooner races the copy and
    //! publishes a torn table. This Batch's own mutators are not in that list: resize(),
    //! move(), addKvCache() and removeKvCache() each wait for themselves.
    //!
    //! Throws when the previous publish has not been waited on, which is how that rule is
    //! enforced: this call stages the watermarks and views itself, so it would be writing
    //! into the same hazard. It refuses rather than waiting because a wait here covers only
    //! its own write -- a member mutated before the call has already raced by then, and
    //! waiting would hide it.
    void publish();

    //! Write back the units each row's watermark is about to pass, given the history the
    //! caller is about to set. One per row; a row may repeat its current history, which
    //! asks for nothing.
    //!
    //! **Call this before advancing history, not after.** A block must be in its host page
    //! before it can be selected, and selectable means below the watermark; resize() refuses
    //! a history this has not covered.
    //!
    //! Bounded by the capacity each row started from, not by how far history travels: only
    //! a unit that was in flight has a cell holding its one copy, so a resumed request
    //! moving history across a whole prompt writes back nothing. A move wider than the
    //! in-flight headroom is refused, because the per-row flush list is sized for one.
    //!
    //! Reads the index table as of the last publish, which covers the blocks in range for
    //! the same reason -- they were allocated by an earlier step.
    void flushSparseUnits(std::vector<int> const& newHistoryLengths);

    //! Where each of `buffers` reads its indices in fetchSparsePages()' output.
    //!
    //! Buffers whose converted indices coincide share a slice. A layer's data and scale
    //! pools mirror -- equal buffer counts, so a lease spans the same cells in both -- and
    //! a selection then resolves to one index serving each, the caller applying the base
    //! of the pool it is reading. Buffers that do not coincide keep slices of their own: a
    //! single scale shared by key and value is its own index table, so it gets one.
    //!
    //! One entry per buffer, in the order given. A slice no buffer maps to is never
    //! written and holds whatever was there: read a buffer's indices at its own slot and
    //! nowhere else.
    [[nodiscard]] std::vector<int> sparseOutSlots(std::vector<BufferId> const& buffers) const;

    //! Resolve a batch's selections against each buffer's per-request history cache.
    //!
    //! Device-driven and stream-ordered throughout: the selection is read from device
    //! memory, so the host neither inspects it nor decides anything, and the call enqueues
    //! onto `stream` and returns. Every shape is static across steps, so a CUDA graph can
    //! capture it.
    //!
    //! `buffers` must be exactly one layer's sparse buffers. One layer, not one layer
    //! group, because a layer's selection is computed from the previous layer's output, so
    //! a step's selections are not all known at its start. The whole layer, not a subset,
    //! because the resolve marks entries resident on behalf of every buffer at once:
    //! filling only some would leave the rest stale while the metadata claimed otherwise.
    //!
    //! `selected` is [maxBatchSize][selectionWidth] unit ordinals, negative where padded,
    //! and must name only units below the history watermark -- a unit at or above it has
    //! not been written back, so fetching it would copy stale bytes. That is a wrong
    //! answer rather than a memory error, and the call cannot detect it, so it is the
    //! selector's responsibility.
    //!
    //! `out` is [maxBatchSize][buffers.size()][selectionWidth], written in full, in the
    //! expanded index space getPageIndexConverter() produces: entry k is the index holding
    //! `selected[row][k]`, or kBadPageIndex where that slot carries no selection.
    //!
    //! Relative to getSparseCacheBaseAddress() for the buffer, not getMemPoolBaseAddress()
    //! and not a cache level: the pool names the buffer's pages, which are host memory for a
    //! sparse life cycle, while the cells a kernel reads live in the arena.
    //!
    //! Compact, in the selection's own order, rather than a table indexed by unit ordinal,
    //! because that is the shape sparse attention kernels take. See
    //! SPARSE_ATTENTION_DESIGN.md's "Fetch".
    //!
    //! `stream` defaults to the Batch's own, which is the only correct choice while the
    //! layers are fetched in order: a layer's selection comes from the one before it. Name
    //! a stream only to run ahead of that -- a model that fetches several layers early can
    //! do so on its own -- and then order it against the Batch's stream yourself, because
    //! the resolve's scratch and counters are Batch state the step clears on that stream.
    void fetchSparsePages(std::vector<BufferId> const& buffers, int32_t const* selected, int selectionWidth,
        int32_t* out, std::optional<CUstream> stream = std::nullopt);

    //! Give the in-flight unit of each request a cell, and report where to write it.
    //!
    //! The unit between the two watermarks is the one the model is about to produce. It has
    //! no host copy yet -- a unit reaches its page only when resize() carries the watermark
    //! past it -- so nothing is fetched here: the call assigns a cell and leaves its bytes
    //! to the model.
    //!
    //! Needs only the watermarks, not the selection, so it runs before fetchSparsePages.
    //!
    //! Idempotent while the watermarks and the reservation hold: the resolve probes
    //! residency before it takes a victim, so a later call -- including the fetch's own
    //! resolve -- finds the entry and reuses it rather than assigning a second. The entry
    //! is pinned for as long as its unit is in flight, so no fetch can evict it.
    //!
    //! `buffers` is one layer's sparse buffers, exactly as for fetchSparsePages.
    //!
    //! `out` is [maxBatchSize][buffers.size()], written in full, in the same expanded index
    //! space fetchSparsePages produces and relative to the same per-pool arena region:
    //! kBadPageIndex where the row holds no reservation or has nothing in flight.
    //! `stream` defaults to the Batch's own, on the same terms as fetchSparsePages.
    void prepareSparseWriteIndices(
        std::vector<BufferId> const& buffers, int32_t* out, std::optional<CUstream> stream = std::nullopt);

    //! The history each row has been written back to. A row that has never needed a flush
    //! reads its own history, so a caller may always compare against it.
    [[nodiscard]] int flushedHistory(int row) const;

    //! Block until the last publish() has finished reading the staging rows.
    //!
    //! A no-op when nothing is in flight, so a caller may use it freely at the top of a step.
    void waitPublished();

    //! One layer group's page table on the host, as a publish will upload it. Read-only:
    //! the entries are the members' to write, through their page locks.
    [[nodiscard]] CoalescedPageTable coalescedPageTable(LayerGroupId layerGroupId) const;

    //! Per-row history length and capacity, published by the same transfer as the page
    //! tables. They have to travel together: the lengths bound which entries of a row are
    //! valid, so a kernel pairing one step's table with another step's lengths would read
    //! past the end of a row that has since been reused.
    [[nodiscard]] WatermarkView watermarks() const;

    //! Row of a member; nullopt when this cache is not one.
    [[nodiscard]] std::optional<int> rowOf(KvCache const& kvCache) const;

    //! Members seated, which is the rows occupied.
    [[nodiscard]] int numKvCaches() const noexcept;

    //! Cumulative fetch statistics, in selection units, since the Batch was built or the
    //! counters were last reset.
    //!
    //! A hit is a selected unit the GPU cache already held; a miss is one the fetch had to
    //! copy from its host page. Padded selection entries and in-flight units are neither,
    //! so they reach neither figure -- which is why this comes from the resolve rather than
    //! from a count the caller could keep.
    //!
    //! Reads back on a temporary stream and syncs only that, so the counters may be a few
    //! launches stale. That does not move a ratio; a caller wanting the two to agree
    //! exactly synchronises the device first and calls this after.
    [[nodiscard]] SparseHitMiss peekHitMissCounter() const;

    //! Zero the cumulative pair. Enqueued on the Batch's stream, so it orders against the
    //! fetches around it rather than tearing one.
    void resetHitMissCounter();

    //! Device bytes every life cycle's sparse arena holds, zero without sparse offload.
    //!
    //! The arenas sit outside the cache tiers' quota, so a caller budgeting device memory
    //! must account for them separately.
    [[nodiscard]] size_t sparseArenaBytes() const;

    //! Whether the occupied rows are exactly [0, numKvCaches()).
    //!
    //! A Batch does not require it. Removing a member leaves its row free and the next join
    //! takes the lowest free one, so a hole is legal and transient.
    [[nodiscard]] bool isContiguous() const noexcept;

    [[nodiscard]] int maxBatchSize() const noexcept
    {
        return mMaxBatchSize;
    }

    [[nodiscard]] int maxBlocksPerSeq() const noexcept
    {
        return mMaxBlocksPerSeq;
    }

    //! The member seated in `row`, or null when that row is free.
    //!
    //! The inverse of rowOf(). Throws when `row` is out of range, like removeKvCache(int):
    //! a free row is an ordinary answer, so null means empty and never means "no such row".
    //!
    //! The reference stays valid for the Batch's lifetime -- the row vector is sized once at
    //! construction -- but what it holds changes as members are seated and removed, so it is
    //! a view of the slot rather than of a member.
    [[nodiscard]] std::shared_ptr<KvCache> const& operator[](int row) const;

private:
    //! Reads the private step state below in order to report it.
    friend class KvCacheIntrospection;

    //! Where one table's converted indices live.
    [[nodiscard]] CUdeviceptr indexTableAddress(IndexTableId tableId) const;

    //! How a sparse copy addresses the run of buffers beginning at (layerId, role).
    //!
    //! The run's buffers occupy consecutive entries of one request, which construction
    //! checked, so naming the first is enough: the rest follow by BufferId.
    struct SparseIndexRun
    {
        //! The run's first buffer, for row zero.
        CUdeviceptr table = 0;
        //! What a converted index is added to. Shared by the whole run.
        MemAddress base = 0;
        //! Elements between consecutive buffers, and between consecutive rows.
        int bufferStride = 0;
        int rowStride = 0;
    };

    [[nodiscard]] SparseIndexRun sparseIndexRun(LayerId layerId, DataRole const& role) const;

    //! Which table serves this buffer, and its entry in that table's buffer dimension.
    [[nodiscard]] std::pair<IndexTableId, int> indexTableSlot(LayerId layerId, DataRole const& role) const;

    //! What a converted index for this buffer is added to.
    [[nodiscard]] MemAddress indexTableBase(LayerId layerId, DataRole const& role) const;

    //! Write back every row's crossed units for one life cycle, from the ranges the Batch
    //! staged. One scan over (layer, row), then one copy per pool covering the whole batch.
    void writeBackSparseUnits(LifeCycleId lifeCycle, CUstream stream);

    //! One layer group's page table on the device, which the index conversion reads.
    [[nodiscard]] GpuCoalescedPageTable gpuCoalescedPageTable(LayerGroupId layerGroupId) const;

    //! Per-row sparse cache state for one layer group, as the fetch kernel sees it.
    //!
    //! A device array of SparseCacheView, indexed by row. A free row, or a member whose
    //! cache is not reserved, reads as a default-constructed view -- null pointers and zero
    //! counts -- which the kernel skips.
    //!
    //! The views hold pointers into each request's own allocation, so what is mirrored is
    //! the handle rather than the memory: the reservation stays with the request. They move
    //! whenever a member resumes or suspends, which is why they are restaged on every
    //! publish alongside the watermarks.
    [[nodiscard]] CUdeviceptr sparseCacheViews(LayerGroupId layerGroupId) const;

    //! Scratch for one layer group's fetch. Null pointers when the group carries no sparse
    //! cache, which is what makes "nothing to fetch here" cheap to detect.
    [[nodiscard]] SparseStepScratch const& sparseStepScratch(LayerGroupId layerGroupId) const;

    //! The unit ordinals bounding what each row's history watermark crossed in the last
    //! flushSparseUnits(), for one life cycle: row `r` crossed [from[r], to[r]).
    //!
    //! Two parallel arrays of maxBatchSize rather than an array of pairs: the scan reads
    //! one row per thread and takes the bounds as separate pointers, so each stays
    //! coalesced. A struct per row would stride both reads by two.
    //!
    //! Ordinals are in this life cycle's units -- tokensPerBlock / unitsPerPage -- so one
    //! token count quantises differently per life cycle. That is why these are per life
    //! cycle where flushedHistory(), which is in tokens, is not.
    //!
    //! Device memory, written by flushSparseUnits() and read by the scan it launches.
    //! Stale unless that flush found work: a step where no row crossed a boundary skips
    //! the upload, which is safe only because nothing reads these then.
    [[nodiscard]] int32_t const* flushFromUnits(LifeCycleId lifeCycle) const;

    //! @see flushFromUnits
    [[nodiscard]] int32_t const* flushToUnits(LifeCycleId lifeCycle) const;

    friend class KvCache;

    //! Called by a member being destroyed. Unreachable while seated -- the Batch holds a
    //! reference, so the destructor cannot run -- and kept as the assertion of that.
    void _detach(KvCache& kvCache) noexcept;

    //! Whether the last publish() is still reading the staging rows.
    //!
    //! For a seated member to check before writing its page indices, which live in those rows.
    [[nodiscard]] bool _publishInFlight() const noexcept
    {
        return !mPublished.isClosed();
    }

    //! Host-side page-index row for one (layer group, row), inside the pinned staging block.
    [[nodiscard]] int32_t* _stagingRow(LifeCycleId lifeCycleId, int row) const;
    //! A (life cycle, row)'s slice of the scratch slot staging. Only valid once the
    //! staging exists, i.e. when mMaxScratchSlots is positive.
    [[nodiscard]] int32_t* _scratchSlotRow(LifeCycleId lifeCycleId, int row) const;

    //! Block until every transfer this Batch issued has retired, so its buffers can go.
    void _drain() noexcept;

    [[nodiscard]] size_t _rowElements() const noexcept;

    //! Element offset of the watermark regions within the staged block.
    [[nodiscard]] size_t _historyOffset() const noexcept;
    [[nodiscard]] size_t _capacityOffset() const noexcept;
    [[nodiscard]] size_t _totalElements() const noexcept;

    //! Seed a row's staged watermarks, so a row reads as its occupant rather than as
    //! whatever the last one left.
    void _seedRowWatermarks(int row, int historyLength, int capacity);

    //! Refresh the staged watermarks and cache views from the members. KvCache reaches the
    //! staging only through setBasePageIndexBuf, for its page indices.
    void _stageRowState();

    //! A buffer's layout: it yields how many int32 elements it needs and resumes with
    //! the memory, so one allocation can serve several of them.
    using LayoutGenerator = Generator<size_t, int32_t*>;

    //! Yields the elements one life cycle's sparse scratch needs, then carves the slice
    //! it is handed. Knows nothing of the counters, which are the driver's.
    LayoutGenerator _layoutStepScratchForLifecycle(LifeCycleId lcId);

    //! Sums what the life cycles need, yields that plus the counter block, and hands each
    //! life cycle its slice. Drives the leaves above while being driven itself.
    LayoutGenerator _layoutStepScratch();

    //! Entries one layer's write-back list must hold, from the step's token budget.
    [[nodiscard]] int32_t _flushCapacity(LifeCycleId lifeCycle) const;

    //! Fill the per-buffer copy descriptors and upload them. Runs once, after the index
    //! tables exist, because a descriptor names where in them a buffer's indices sit.
    void _allocateCopyDesc();

    //! Device room for the per-row unit ranges a write-back reads:
    //! [life cycle][2][maxBatchSize], as [from][to]. Allocated with the fetch scratch,
    //! which has the same lifetime.
    //! Yields the elements the write-back ranges need, then clears the slice.
    LayoutGenerator _layoutFlushRanges();

    //! Points into mDeviceArena; null when no life cycle holds a sparse cache.
    int32_t* mFlushRanges = nullptr;
    //! Staged host-side, then handed to a copy that consumes it before returning.
    std::vector<int32_t> mFlushRangeStaging;
    //! Per row, the history its units have been written back to.
    std::vector<int> mFlushedHistory;
    //! Recorded after the write-back, so the next call's staging can wait for its reads.
    CachedCudaEvent mFlushRead = CachedCudaEvent::makeNull();

    //! Byte offset of the cache view region within the staged block.
    [[nodiscard]] size_t _viewsOffsetBytes() const noexcept;
    //! Bytes of one staged block: the page table, the row state and the views. A device
    //! block is the same size, being a copy of it.
    [[nodiscard]] size_t _totalBytes() const noexcept;

    //! Outlives the Batch.
    KvCacheManager* const mManager;
    CUstream mCudaStream = nullptr;
    int mMaxBatchSize = 0;
    int mMaxBlocksPerSeq = 0;
    std::optional<int> mMaxTotalInputTokens;

    //! A seated member cannot be destroyed under the Batch, so a caller may keep only the
    //! Batch. KvCache points back with a raw `mBatch`, so there is no cycle.
    std::vector<std::shared_ptr<KvCache>> mRows;

    //! Cumulative fetch statistics, across every life cycle and step. Outside mStepScratch
    //! because the step's clear must not touch it.
    CudaUniqPtr<SparseHitMiss> mHitMiss;

    //! Device storage behind every life cycle's per-BufferId copy descriptors.
    CudaUniqPtr<SparseCopyDesc> mCopyDesc;

    //! Every life cycle's work-list counters. A step clears them once, in publish(); a
    //! producer then raises its own and the copy that consumes the list reads it.
    //!
    //! Not owning: they lead `mStepScratch`, which is what makes one clear reach all of
    //! them, so that allocation frees them.
    int32_t* mCounters = nullptr;
    size_t mCounterElements = 0;

    //! Per life cycle, fixed at construction. Only scratch-slot substitution can make one
    //! layer's indices differ from another's in the same life cycle, so a life cycle that
    //! cannot hold scratch takes SHARED and the single smaller table that implies.
    TypedVec<LifeCycleId, PageIndexMode> mIndexModes;

    //! Whether the caller named the mode rather than letting it be derived. publish()
    //! validates a named SHARED against each member, every step.
    bool mIndexModeNamed = false;

    //! One converted index table: the specs it serves, and what a consumer needs to read
    //! it. `concatOffsets` is in page-index units, parallel to a request's buffers.
    struct IndexTable
    {
        std::vector<IndexTableSpec> served;
        LifeCycleId lifeCycle{0};
        int scale = 1;
        std::vector<int> concatOffsets;
        //! Element offset into mIndexTableMemory.
        size_t elementOffset = 0;
    };

    std::vector<IndexTable> mIndexTables;

    //! Where one buffer's indices are.
    struct BufferSlot
    {
        IndexTableId table{0};
        int entry = 0;
    };

    std::map<BufferId, BufferSlot> mBufferSlots;

    //! The slot naming this buffer. Throws naming `api` when no index table spec has it.
    [[nodiscard]] BufferSlot const& _bufferSlot(LayerId layerId, DataRole const& role, char const* api) const;

    //! The spec that named this buffer. Exactly one does: specs may not overlap.
    [[nodiscard]] IndexTableSpec const& _specOf(LayerId layerId, DataRole const& role, char const* api) const;

    //! What a converted index of `spec` is added to: its first buffer's pool base, under
    //! the index mode this Batch chose. Every buffer of the spec shares it.
    [[nodiscard]] MemAddress _specBase(IndexTableSpec const& spec) const;

    //! Validate the requests, merge those whose converted indices coincide, and record
    //! where each buffer landed.
    void _buildIndexTables(std::vector<IndexTableSpec> const& specs);

    //! Refuse a sparse life cycle whose buffers no single launch can address.
    //!
    //! A sparse copy walks one (layer, pool) run of buffers per launch, from one base
    //! pointer, so that run has to sit at consecutive entries of one request. Splitting it
    //! across requests would need two bases in one launch; declaring only part of it would
    //! leave the rest with no converted indices at all. Checked here because a run that
    //! cannot be addressed produces wrong page addresses at step time rather than an error.
    void _checkSparseRunsAreAddressable() const;

    //! Allocate the tables and the device-side descriptors the conversion reads.
    void _allocateIndexTables();

    //! Stage each member's scratch block range, for the life cycles that can hold scratch.
    void _stageScratch();

    //! Elements of the scratch staging.
    [[nodiscard]] size_t _scratchElements() const noexcept;

    //! Host and device sides of the scratch metadata, laid out [begs][ends][slots].
    //! Empty when no table's life cycle can hold scratch.
    StaticHostMem mScratchStaging;
    CudaUniqPtr<int32_t> mScratchDevice;
    //! Slot ids a request may hold for one life cycle, which bounds the staging.
    int mMaxScratchSlots = 0;
    //! Tables whose life cycle can hold scratch sort last, so one launch covers each kind.
    int mNumPlainTables = 0;

    //! Convert this step's raw slot ids into every table, on `stream`.
    void _convertIndices(CUstream stream);

    //! All converted tables, carved by IndexTable::elementOffset.
    CudaUniqPtr<int32_t> mIndexTableMemory;
    //! One IndexTableDesc per table, and the offsets they point at.
    CudaUniqPtr<> mIndexTableDescs;
    int mMaxTableEntries = 0;

    //! Pinned so the members' index writes land here with no extra copy, and so the
    //! publish is a single asynchronous transfer per layer group.
    StaticHostMem mStaging;

    //! What the fetch reads. Its address is baked into a captured graph, so it is written
    //! only on the caller's stream, after the replay that read the previous publish.
    CudaUniqPtr<> mDevice;

    //! A device-side copy of the same layout, filled from the host ahead of time.
    //!
    //! Nothing captured reads it, so its fill may overlap the previous step.
    CudaUniqPtr<> mDeviceStaging;

    //! Carries the host-to-device fill, off the caller's stream so it can run early.
    CachedCudaStream mUploadStream;

    //! One allocation holding every layer group's fetch scratch, carved per group.
    //! One device block split between the layout generators. The pointers they hand out
    //! point into it, so it outlives every one of them.
    CudaUniqPtr<int32_t> mDeviceArena;
    TypedVec<LifeCycleId, SparseStepScratch> mScratch;

    //! Recorded after the host-to-device fill, so the host can tell when the staged rows
    //! are free to write again. Null when no publish is in flight.
    CachedCudaEvent mPublished = CachedCudaEvent::makeNull();

    //! Recorded after the device staging has been read into the final tables. The next
    //! fill must wait on it, since that fill runs on another stream and would otherwise
    //! overwrite the staging while this step's copy is still reading it.
    CachedCudaEvent mStagingRead = CachedCudaEvent::makeNull();
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
