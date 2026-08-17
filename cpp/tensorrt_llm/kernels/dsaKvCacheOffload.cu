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
    std::int32_t const* __restrict__ globalIndices, std::uint8_t* __restrict__ hostPool, std::int64_t numRows,
    std::int32_t hostRowGrains, std::int32_t layerGrains, std::int32_t layerInGroup, std::int32_t strideFactor,
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

} // namespace

void invokeDsaKvCacheOffloadMirror(std::uint8_t const* sourcePool, std::int32_t const* globalIndices,
    std::uint8_t* hostPool, std::int64_t numRows, std::int64_t numHostRows, std::int32_t hostRowBytes,
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
    auto const blocks = std::min<std::int64_t>((numRows + kWarpsPerBlock - 1) / kWarpsPerBlock, kMirrorGridLimit);
    dsaKvCacheOffloadMirrorKernel<<<static_cast<std::uint32_t>(blocks), kThreadsPerBlock, 0, stream>>>(sourcePool,
        globalIndices, hostPool, numRows, hostRowBytes / kVectorBytes, layerBytes / kVectorBytes, layerInGroup,
        strideFactor, tokensPerBlock, layerOffset, numHostRows);
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

} // namespace kernels

TRTLLM_NAMESPACE_END
