/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "dsaKvCacheOffload.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <algorithm>
#include <cub/cub.cuh>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr std::int32_t kWarpSize = 32;
constexpr std::int32_t kThreadsPerBlock = 256;
constexpr std::int32_t kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr std::int32_t kGatherGridLimit = 16;
constexpr std::int32_t kMirrorGridLimit = 256;
constexpr std::int32_t kPrototypeLayers = 3;

__device__ __forceinline__ std::int64_t getHostRow(std::int32_t globalIndex, std::int32_t strideFactor,
    std::int32_t tokensPerBlock, std::int32_t layerOffset, std::int64_t numHostRows)
{
    std::int32_t const block = globalIndex / strideFactor;
    std::int32_t const withinBlock = globalIndex % strideFactor - layerOffset * tokensPerBlock;
    std::int64_t const hostRow = static_cast<std::int64_t>(block) * tokensPerBlock + withinBlock;
    if (withinBlock < 0 || withinBlock >= tokensPerBlock || hostRow >= numHostRows)
    {
        return -1;
    }
    return hostRow;
}

__global__ void dsaKvCacheOffloadMirrorKernel(std::uint8_t const* __restrict__ sourcePool,
    std::int32_t const* __restrict__ globalIndices, std::uint8_t* __restrict__ hostPool,
    std::int32_t* __restrict__ hostVersions, std::int64_t numRows, std::int32_t hostRowGrains,
    std::int32_t layerGrains, std::int32_t layerInGroup, std::int32_t strideFactor,
    std::int32_t tokensPerBlock, std::int32_t layerOffset, std::int64_t numHostRows)
{
    auto const warp = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarpSize;
    auto const lane = threadIdx.x % kWarpSize;
    auto const numWarps = static_cast<std::int64_t>(gridDim.x) * kWarpsPerBlock;
    auto const* source = reinterpret_cast<uint4 const*>(sourcePool);
    auto* destination = reinterpret_cast<uint4*>(hostPool);

    for (std::int64_t row = warp; row < numRows; row += numWarps)
    {
        std::int32_t const globalIndex = globalIndices[row];
        if (globalIndex < 0)
        {
            continue;
        }
        std::int64_t const hostRow = getHostRow(globalIndex, strideFactor, tokensPerBlock, layerOffset, numHostRows);
        if (hostRow < 0)
        {
            continue;
        }
        auto const* sourceRow = source + static_cast<std::int64_t>(globalIndex) * layerGrains;
        auto* destinationRow
            = destination + hostRow * hostRowGrains + static_cast<std::int64_t>(layerInGroup) * layerGrains;
        for (std::int32_t grain = lane; grain < layerGrains; grain += kWarpSize)
        {
            destinationRow[grain] = sourceRow[grain];
        }
        // Every lane must publish its host writes before lane 0 publishes the
        // row version; a fence in lane 0 alone would not order other lanes.
        __threadfence_system();
        __syncwarp();
        if (lane == 0)
        {
            atomicAdd(hostVersions + hostRow * kPrototypeLayers + layerInGroup, 1);
        }
    }
}

__global__ void dsaKvCacheOffloadGatherKernel(std::uint8_t const* __restrict__ hostPool,
    std::int32_t const* __restrict__ globalIndices, std::uint8_t* __restrict__ output, std::int64_t numRows,
    std::int32_t hostRowGrains, std::int32_t layerGrains, std::int32_t layerInGroup, std::int32_t strideFactor,
    std::int32_t tokensPerBlock, std::int32_t layerOffset, std::int64_t numHostRows)
{
    auto const warp = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarpSize;
    auto const lane = threadIdx.x % kWarpSize;
    auto const numWarps = static_cast<std::int64_t>(gridDim.x) * kWarpsPerBlock;
    auto const* source = reinterpret_cast<uint4 const*>(hostPool);
    auto* destination = reinterpret_cast<uint4*>(output);

    for (std::int64_t row = warp; row < numRows; row += numWarps)
    {
        std::int32_t const globalIndex = globalIndices[row];
        auto* destinationRow = destination + row * layerGrains;
        std::int64_t const hostRow
            = globalIndex < 0 ? -1 : getHostRow(globalIndex, strideFactor, tokensPerBlock, layerOffset, numHostRows);
        if (hostRow < 0)
        {
            for (std::int32_t grain = lane; grain < layerGrains; grain += kWarpSize)
            {
                destinationRow[grain] = uint4{};
            }
            continue;
        }
        auto const* sourceRow
            = source + hostRow * hostRowGrains + static_cast<std::int64_t>(layerInGroup) * layerGrains;
        for (std::int32_t grain = lane; grain < layerGrains; grain += kWarpSize)
        {
            destinationRow[grain] = sourceRow[grain];
        }
    }
}

__global__ void prepareDsaKvCacheOffloadEpochKernel(std::int32_t* epoch, std::int32_t* freeCounts,
    std::int32_t* allocationCounts, std::int32_t numRequests)
{
    if (threadIdx.x == 0)
    {
        ++epoch[0];
    }
    for (std::int32_t request = threadIdx.x; request < numRequests; request += blockDim.x)
    {
        freeCounts[request] = 0;
        allocationCounts[request] = 0;
    }
}

__global__ void dsaKvCacheOffloadFindHitsKernel(std::int32_t const* __restrict__ globalIndices,
    std::int32_t const* __restrict__ hostVersions, std::int32_t const* __restrict__ cacheKeys,
    std::int32_t const* __restrict__ cacheVersions, std::int32_t* __restrict__ rowToSlot,
    std::int32_t const* __restrict__ epoch, std::int32_t* __restrict__ slotEpochs,
    std::int32_t* __restrict__ outputSlots, std::int64_t numRows, std::int64_t numHostRows,
    std::int32_t maxRequests, std::int32_t rowsPerRequest, std::int32_t slotsPerRequest,
    std::int32_t layerInGroup, std::int32_t strideFactor, std::int32_t tokensPerBlock,
    std::int32_t layerOffset)
{
    auto const thread = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const numThreads = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    std::int32_t const currentEpoch = epoch[0];
    std::int32_t const totalSlots = maxRequests * slotsPerRequest;

    for (std::int64_t row = thread; row < numRows; row += numThreads)
    {
        std::int32_t const request = static_cast<std::int32_t>(row / rowsPerRequest);
        std::int32_t const globalIndex = globalIndices[row];
        auto const resolvedHostRow = globalIndex < 0
            ? -1
            : getHostRow(globalIndex, strideFactor, tokensPerBlock, layerOffset, numHostRows);
        std::int32_t const hostRow = static_cast<std::int32_t>(resolvedHostRow);
        if (hostRow < 0 || request >= maxRequests)
        {
            outputSlots[row] = -1;
            continue;
        }

        std::int32_t const sourceSlot = atomicAdd(rowToSlot + hostRow, 0);
        std::int32_t const requestBegin = request * slotsPerRequest;
        std::int32_t const requestEnd = requestBegin + slotsPerRequest;
        std::int32_t const currentVersion = hostVersions[hostRow * kPrototypeLayers + layerInGroup];
        bool const hit = sourceSlot >= requestBegin && sourceSlot < requestEnd && sourceSlot < totalSlots
            && cacheKeys[sourceSlot] == hostRow && cacheVersions[sourceSlot] == currentVersion;
        if (hit)
        {
            atomicExch(slotEpochs + sourceSlot, currentEpoch);
            outputSlots[row] = sourceSlot;
        }
        else
        {
            if (sourceSlot >= 0 && sourceSlot < totalSlots)
            {
                atomicCAS(rowToSlot + hostRow, sourceSlot, -1);
            }
            outputSlots[row] = -1;
        }
    }
}

__global__ void dsaKvCacheOffloadBuildFreeSlotsKernel(std::int32_t const* __restrict__ epoch,
    std::int32_t const* __restrict__ slotEpochs, std::int32_t* __restrict__ freeSlots,
    std::int32_t* __restrict__ freeCounts, std::int32_t slotsPerRequest)
{
    using BlockScan = cub::BlockScan<std::int32_t, kThreadsPerBlock, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename BlockScan::TempStorage scanStorage;
    __shared__ std::int32_t requestFreeCount;

    std::int32_t const request = blockIdx.x;
    auto const requestBegin = static_cast<std::int64_t>(request) * slotsPerRequest;
    std::int32_t const currentEpoch = epoch[0];
    if (threadIdx.x == 0)
    {
        requestFreeCount = 0;
    }
    __syncthreads();

    for (std::int32_t slotBase = 0; slotBase < slotsPerRequest; slotBase += kThreadsPerBlock)
    {
        std::int32_t const slotInRequest = slotBase + threadIdx.x;
        bool const isFree = slotInRequest < slotsPerRequest
            && slotEpochs[requestBegin + slotInRequest] != currentEpoch;
        std::int32_t freeIndex;
        std::int32_t blockFreeCount;
        BlockScan(scanStorage).ExclusiveSum(static_cast<std::int32_t>(isFree), freeIndex, blockFreeCount);
        std::int32_t const blockOffset = requestFreeCount;
        if (isFree)
        {
            freeSlots[requestBegin + blockOffset + freeIndex]
                = static_cast<std::int32_t>(requestBegin + slotInRequest);
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            requestFreeCount = blockOffset + blockFreeCount;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        freeCounts[request] = requestFreeCount;
    }
}

__global__ void dsaKvCacheOffloadFetchMissesKernel(std::uint8_t const* __restrict__ hostPool,
    std::int32_t const* __restrict__ globalIndices, std::int32_t const* __restrict__ hostVersions,
    std::int32_t* __restrict__ cacheKeys, std::int32_t* __restrict__ cacheVersions,
    std::int32_t* __restrict__ rowToSlot, std::int32_t const* __restrict__ epoch,
    std::int32_t* __restrict__ slotEpochs, std::int32_t const* __restrict__ freeSlots,
    std::int32_t const* __restrict__ freeCounts, std::int32_t* __restrict__ allocationCounts,
    std::uint8_t* __restrict__ cacheValues, std::int32_t* __restrict__ outputSlots,
    std::int32_t* __restrict__ missCount, std::int64_t numRows,
    std::int64_t numHostRows, std::int32_t maxRequests, std::int32_t rowsPerRequest,
    std::int32_t slotsPerRequest, std::int32_t hostRowGrains, std::int32_t layerGrains,
    std::int32_t layerInGroup, std::int32_t strideFactor, std::int32_t tokensPerBlock,
    std::int32_t layerOffset)
{
    auto const warp = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarpSize;
    auto const lane = threadIdx.x % kWarpSize;
    auto const numWarps = static_cast<std::int64_t>(gridDim.x) * kWarpsPerBlock;
    auto const* hostValues = reinterpret_cast<uint4 const*>(hostPool);
    auto* deviceValues = reinterpret_cast<uint4*>(cacheValues);
    std::int32_t const currentEpoch = epoch[0];

    for (std::int64_t row = warp; row < numRows; row += numWarps)
    {
        if (outputSlots[row] >= 0)
        {
            continue;
        }

        std::int32_t const request = static_cast<std::int32_t>(row / rowsPerRequest);
        std::int32_t hostRow = -1;
        std::int32_t currentVersion = -1;
        std::int32_t destinationSlot = -1;
        if (lane == 0)
        {
            std::int32_t const globalIndex = globalIndices[row];
            auto const resolvedHostRow = globalIndex < 0
                ? -1
                : getHostRow(globalIndex, strideFactor, tokensPerBlock, layerOffset, numHostRows);
            hostRow = static_cast<std::int32_t>(resolvedHostRow);
            if (hostRow >= 0 && request < maxRequests)
            {
                currentVersion = hostVersions[hostRow * kPrototypeLayers + layerInGroup];
                std::int32_t const freeIndex = atomicAdd(allocationCounts + request, 1);
                if (freeIndex < freeCounts[request])
                {
                    destinationSlot = freeSlots[static_cast<std::int64_t>(request) * slotsPerRequest + freeIndex];
                    slotEpochs[destinationSlot] = currentEpoch;
                }

                if (destinationSlot >= 0)
                {
                    std::int32_t const oldKey = cacheKeys[destinationSlot];
                    if (oldKey >= 0 && oldKey < numHostRows)
                    {
                        atomicCAS(rowToSlot + oldKey, destinationSlot, -1);
                    }
                }
            }
        }

        hostRow = __shfl_sync(0xffffffffU, hostRow, 0);
        currentVersion = __shfl_sync(0xffffffffU, currentVersion, 0);
        destinationSlot = __shfl_sync(0xffffffffU, destinationSlot, 0);
        if (hostRow < 0 || destinationSlot < 0)
        {
            continue;
        }

        auto const* sourceRow = hostValues + static_cast<std::int64_t>(hostRow) * hostRowGrains
            + static_cast<std::int64_t>(layerInGroup) * layerGrains;
        auto* destinationRow = deviceValues + static_cast<std::int64_t>(destinationSlot) * layerGrains;
        for (std::int32_t grain = lane; grain < layerGrains; grain += kWarpSize)
        {
            destinationRow[grain] = sourceRow[grain];
        }
        // The consumer waits for this stream's completion event. The warp sync
        // only ensures lane 0 publishes slot metadata after every lane writes
        // its portion of the row; no per-row device-wide fence is required.
        __syncwarp();

        if (lane == 0)
        {
            cacheKeys[destinationSlot] = hostRow;
            cacheVersions[destinationSlot] = currentVersion;
            outputSlots[row] = destinationSlot;
            atomicExch(rowToSlot + hostRow, destinationSlot);
            atomicAdd(missCount, 1);
        }
    }
}

} // namespace

void invokeDsaKvCacheOffloadMirror(std::uint8_t const* sourcePool, std::int32_t const* globalIndices,
    std::uint8_t* hostPool, std::int32_t* hostVersions, std::int64_t numRows, std::int64_t numHostRows,
    std::int32_t hostRowBytes, std::int32_t layerBytes, std::int32_t layerInGroup, std::int32_t strideFactor,
    std::int32_t tokensPerBlock, std::int32_t layerOffset, cudaStream_t stream)
{
    if (numRows == 0)
    {
        return;
    }
    constexpr std::int32_t kVectorBytes = sizeof(uint4);
    TLLM_CHECK_WITH_INFO(hostRowBytes % kVectorBytes == 0, "host row bytes must be 16-byte aligned");
    TLLM_CHECK_WITH_INFO(layerBytes % kVectorBytes == 0, "layer bytes must be 16-byte aligned");
    auto const blocks = std::min<std::int64_t>((numRows + kWarpsPerBlock - 1) / kWarpsPerBlock, kMirrorGridLimit);
    dsaKvCacheOffloadMirrorKernel<<<static_cast<std::uint32_t>(blocks), kThreadsPerBlock, 0, stream>>>(sourcePool,
        globalIndices, hostPool, hostVersions, numRows, hostRowBytes / kVectorBytes, layerBytes / kVectorBytes,
        layerInGroup, strideFactor, tokensPerBlock, layerOffset, numHostRows);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void invokeDsaKvCacheOffloadGather(std::uint8_t const* hostPool, std::int32_t const* globalIndices,
    std::uint8_t* output, std::int64_t numRows, std::int64_t numHostRows, std::int32_t hostRowBytes,
    std::int32_t layerBytes, std::int32_t layerInGroup, std::int32_t strideFactor, std::int32_t tokensPerBlock,
    std::int32_t layerOffset, cudaStream_t stream)
{
    if (numRows == 0)
    {
        return;
    }
    constexpr std::int32_t kVectorBytes = sizeof(uint4);
    TLLM_CHECK_WITH_INFO(hostRowBytes % kVectorBytes == 0, "host row bytes must be 16-byte aligned");
    TLLM_CHECK_WITH_INFO(layerBytes % kVectorBytes == 0, "layer bytes must be 16-byte aligned");
    auto const blocks = std::min<std::int64_t>((numRows + kWarpsPerBlock - 1) / kWarpsPerBlock, kGatherGridLimit);
    dsaKvCacheOffloadGatherKernel<<<static_cast<std::uint32_t>(blocks), kThreadsPerBlock, 0, stream>>>(hostPool,
        globalIndices, output, numRows, hostRowBytes / kVectorBytes, layerBytes / kVectorBytes, layerInGroup,
        strideFactor, tokensPerBlock, layerOffset, numHostRows);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void invokeDsaKvCacheOffloadIncrementalGather(std::uint8_t const* hostPool,
    std::int32_t const* globalIndices, std::int32_t const* hostVersions, std::int32_t* cacheKeys,
    std::int32_t* cacheVersions, std::int32_t* rowToSlot, std::int32_t* epoch, std::int32_t* slotEpochs,
    std::int32_t* freeSlots, std::int32_t* freeCounts, std::int32_t* allocationCounts,
    std::uint8_t* cacheValues, std::int32_t* outputSlots, std::int32_t* missCount, std::int64_t numRows,
    std::int64_t numHostRows, std::int32_t maxRequests, std::int32_t rowsPerRequest,
    std::int32_t slotsPerRequest, std::int32_t hostRowBytes, std::int32_t layerBytes,
    std::int32_t layerInGroup, std::int32_t strideFactor, std::int32_t tokensPerBlock,
    std::int32_t layerOffset, cudaStream_t stream)
{
    TLLM_CUDA_CHECK(cudaMemsetAsync(missCount, 0, sizeof(std::int32_t), stream));
    if (numRows == 0)
    {
        return;
    }
    constexpr std::int32_t kVectorBytes = sizeof(uint4);
    TLLM_CHECK_WITH_INFO(hostRowBytes % kVectorBytes == 0, "host row bytes must be 16-byte aligned");
    TLLM_CHECK_WITH_INFO(layerBytes % kVectorBytes == 0, "layer bytes must be 16-byte aligned");
    std::int32_t const numRequests = static_cast<std::int32_t>(numRows / rowsPerRequest);
    prepareDsaKvCacheOffloadEpochKernel<<<1, kThreadsPerBlock, 0, stream>>>(
        epoch, freeCounts, allocationCounts, numRequests);
    auto const hitBlocks = (numRows + kThreadsPerBlock - 1) / kThreadsPerBlock;
    dsaKvCacheOffloadFindHitsKernel<<<static_cast<std::uint32_t>(hitBlocks), kThreadsPerBlock, 0, stream>>>(
        globalIndices, hostVersions, cacheKeys, cacheVersions, rowToSlot, epoch, slotEpochs, outputSlots, numRows,
        numHostRows, maxRequests, rowsPerRequest, slotsPerRequest, layerInGroup, strideFactor, tokensPerBlock,
        layerOffset);
    TLLM_CUDA_CHECK(cudaGetLastError());
    dsaKvCacheOffloadBuildFreeSlotsKernel<<<numRequests, kThreadsPerBlock, 0, stream>>>(
        epoch, slotEpochs, freeSlots, freeCounts, slotsPerRequest);
    TLLM_CUDA_CHECK(cudaGetLastError());
    auto const missBlocks
        = std::min<std::int64_t>((numRows + kWarpsPerBlock - 1) / kWarpsPerBlock, kGatherGridLimit);
    dsaKvCacheOffloadFetchMissesKernel<<<static_cast<std::uint32_t>(missBlocks), kThreadsPerBlock, 0, stream>>>(
        hostPool, globalIndices, hostVersions, cacheKeys, cacheVersions, rowToSlot, epoch, slotEpochs, freeSlots,
        freeCounts, allocationCounts, cacheValues, outputSlots, missCount, numRows, numHostRows, maxRequests,
        rowsPerRequest, slotsPerRequest, hostRowBytes / kVectorBytes, layerBytes / kVectorBytes, layerInGroup,
        strideFactor, tokensPerBlock, layerOffset);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END
