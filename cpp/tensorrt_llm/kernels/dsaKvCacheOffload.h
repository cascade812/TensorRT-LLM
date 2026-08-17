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

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void invokeDsaKvCacheOffloadMirror(std::uint8_t const* sourcePool, std::int32_t const* globalIndices,
    std::uint8_t* hostPool, std::int32_t* hostVersions, std::int64_t numRows, std::int64_t numHostRows,
    std::int32_t hostRowBytes, std::int32_t layerBytes, std::int32_t layerInGroup, std::int32_t strideFactor,
    std::int32_t tokensPerBlock, std::int32_t layerOffset, cudaStream_t stream);

void invokeDsaKvCacheOffloadGather(std::uint8_t const* hostPool, std::int32_t const* globalIndices,
    std::uint8_t* output, std::int64_t numRows, std::int64_t numHostRows, std::int32_t hostRowBytes,
    std::int32_t layerBytes, std::int32_t layerInGroup, std::int32_t strideFactor, std::int32_t tokensPerBlock,
    std::int32_t layerOffset, cudaStream_t stream);

void invokeDsaKvCacheOffloadIncrementalGather(std::uint8_t const* hostPool,
    std::int32_t const* globalIndices, std::int32_t const* hostVersions, std::int32_t* cacheKeys,
    std::int32_t* cacheVersions, std::int32_t* rowToSlot, std::int32_t* epoch, std::int32_t* slotEpochs,
    std::int32_t* freeSlots, std::int32_t* freeCounts, std::int32_t* allocationCounts,
    std::uint8_t* cacheValues, std::int32_t* outputSlots, std::int32_t* missCount, std::int64_t numRows,
    std::int64_t numHostRows, std::int32_t maxRequests, std::int32_t rowsPerRequest,
    std::int32_t slotsPerRequest, std::int32_t hostRowBytes, std::int32_t layerBytes,
    std::int32_t layerInGroup, std::int32_t strideFactor, std::int32_t tokensPerBlock,
    std::int32_t layerOffset, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
