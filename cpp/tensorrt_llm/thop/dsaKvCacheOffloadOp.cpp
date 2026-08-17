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

#include "tensorrt_llm/kernels/dsaKvCacheOffload.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <c10/cuda/CUDAGuard.h>

#include <limits>

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr std::int64_t kPrototypeLayers = 3;
constexpr std::int64_t kWorkingSetCapacityMultiplier = 2;

void validateCommon(th::Tensor const& hostPool, th::Tensor const& globalIndices)
{
    TORCH_CHECK(hostPool.device().is_cpu() && hostPool.is_pinned(), "host_pool must be a pinned CPU tensor");
    TORCH_CHECK(hostPool.scalar_type() == th::kUInt8 && hostPool.dim() == 2 && hostPool.is_contiguous(),
        "host_pool must be a contiguous rank-2 uint8 tensor");
    TORCH_CHECK(globalIndices.is_cuda() && globalIndices.scalar_type() == th::kInt32 && globalIndices.is_contiguous(),
        "global_indices must be a contiguous CUDA int32 tensor");
    TORCH_CHECK(hostPool.size(1) % kPrototypeLayers == 0,
        "host_pool row width must be divisible by the three prototype layers");
    TORCH_CHECK(hostPool.size(0) <= std::numeric_limits<std::int32_t>::max(),
        "host_pool has too many rows for the prototype's int32 working-set keys");
}

void validateHostVersions(th::Tensor const& hostVersions, th::Tensor const& hostPool,
    th::Tensor const& globalIndices)
{
    TORCH_CHECK(hostVersions.is_cuda() && hostVersions.scalar_type() == th::kInt32 && hostVersions.dim() == 2
            && hostVersions.is_contiguous(),
        "host_versions must be a contiguous rank-2 CUDA int32 tensor");
    TORCH_CHECK(hostVersions.device() == globalIndices.device(),
        "host_versions and global_indices must be on the same CUDA device");
    TORCH_CHECK(hostVersions.size(0) == hostPool.size(0) && hostVersions.size(1) == kPrototypeLayers,
        "host_versions must have shape [host_pool rows, 3]");
}

void validateCudaInt32(th::Tensor const& tensor, th::Device const& device, char const* name)
{
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == device && tensor.scalar_type() == th::kInt32
            && tensor.is_contiguous(),
        name, " must be a contiguous CUDA int32 tensor on the global_indices device");
}

} // namespace

void dsaKvCacheOffloadMirror(th::Tensor const& sourcePool, th::Tensor const& globalIndices, th::Tensor& hostPool,
    th::Tensor& hostVersions, std::int64_t strideFactor, std::int64_t tokensPerBlock, std::int64_t layerOffset,
    std::int64_t layerInGroup)
{
    validateCommon(hostPool, globalIndices);
    validateHostVersions(hostVersions, hostPool, globalIndices);
    TORCH_CHECK(sourcePool.is_cuda() && sourcePool.device() == globalIndices.device() && sourcePool.element_size() == 1
            && sourcePool.is_contiguous(),
        "source_pool must be a contiguous one-byte CUDA tensor on the global_indices device");
    TORCH_CHECK(layerInGroup >= 0 && layerInGroup < kPrototypeLayers, "layer_in_group must be in [0, 3)");
    TORCH_CHECK(strideFactor > 0 && tokensPerBlock > 0, "stride_factor and tokens_per_block must be positive");

    std::int64_t const hostRowBytes = hostPool.size(1);
    std::int64_t const layerBytes = hostRowBytes / kPrototypeLayers;
    TORCH_CHECK(sourcePool.numel() % layerBytes == 0, "source_pool size must be a multiple of layer bytes");

    c10::cuda::CUDAGuard const deviceGuard(globalIndices.device());
    auto const stream = at::cuda::getCurrentCUDAStream(globalIndices.get_device()).stream();
    tk::invokeDsaKvCacheOffloadMirror(reinterpret_cast<std::uint8_t const*>(sourcePool.data_ptr()),
        globalIndices.data_ptr<std::int32_t>(), hostPool.data_ptr<std::uint8_t>(),
        hostVersions.data_ptr<std::int32_t>(), globalIndices.numel(), hostPool.size(0),
        static_cast<std::int32_t>(hostRowBytes), static_cast<std::int32_t>(layerBytes),
        static_cast<std::int32_t>(layerInGroup), static_cast<std::int32_t>(strideFactor),
        static_cast<std::int32_t>(tokensPerBlock), static_cast<std::int32_t>(layerOffset), stream);
}

void dsaKvCacheOffloadGather(th::Tensor const& hostPool, th::Tensor const& globalIndices, th::Tensor& output,
    std::int64_t strideFactor, std::int64_t tokensPerBlock, std::int64_t layerOffset,
    std::int64_t layerInGroup)
{
    validateCommon(hostPool, globalIndices);
    TORCH_CHECK(output.is_cuda() && output.device() == globalIndices.device() && output.scalar_type() == th::kUInt8
            && output.dim() == 2 && output.is_contiguous(),
        "output must be a contiguous rank-2 CUDA uint8 tensor on the global_indices device");
    TORCH_CHECK(layerInGroup >= 0 && layerInGroup < kPrototypeLayers, "layer_in_group must be in [0, 3)");
    std::int64_t const layerBytes = hostPool.size(1) / kPrototypeLayers;
    TORCH_CHECK(output.size(0) >= globalIndices.numel() && output.size(1) == layerBytes,
        "output must have at least one layer-width row per global index");
    TORCH_CHECK(strideFactor > 0 && tokensPerBlock > 0, "stride_factor and tokens_per_block must be positive");

    c10::cuda::CUDAGuard const deviceGuard(globalIndices.device());
    auto const stream = at::cuda::getCurrentCUDAStream(globalIndices.get_device()).stream();
    tk::invokeDsaKvCacheOffloadGather(hostPool.data_ptr<std::uint8_t>(), globalIndices.data_ptr<std::int32_t>(),
        output.data_ptr<std::uint8_t>(), globalIndices.numel(), hostPool.size(0),
        static_cast<std::int32_t>(hostPool.size(1)), static_cast<std::int32_t>(layerBytes),
        static_cast<std::int32_t>(layerInGroup), static_cast<std::int32_t>(strideFactor),
        static_cast<std::int32_t>(tokensPerBlock), static_cast<std::int32_t>(layerOffset), stream);
}

void dsaKvCacheOffloadIncrementalGather(th::Tensor const& hostPool, th::Tensor const& globalIndices,
    th::Tensor const& hostVersions, th::Tensor& cacheKeys, th::Tensor& cacheVersions, th::Tensor& rowToSlot,
    th::Tensor& epoch, th::Tensor& slotEpochs, th::Tensor& freeSlots, th::Tensor& freeCounts,
    th::Tensor& allocationCounts, th::Tensor& cacheValues, th::Tensor& outputSlots, th::Tensor& missCount,
    std::int64_t rowsPerRequest, std::int64_t strideFactor, std::int64_t tokensPerBlock,
    std::int64_t layerOffset, std::int64_t layerInGroup)
{
    validateCommon(hostPool, globalIndices);
    validateHostVersions(hostVersions, hostPool, globalIndices);
    validateCudaInt32(cacheKeys, globalIndices.device(), "cache_keys");
    validateCudaInt32(cacheVersions, globalIndices.device(), "cache_versions");
    validateCudaInt32(rowToSlot, globalIndices.device(), "row_to_slot");
    validateCudaInt32(epoch, globalIndices.device(), "epoch");
    validateCudaInt32(slotEpochs, globalIndices.device(), "slot_epochs");
    validateCudaInt32(freeSlots, globalIndices.device(), "free_slots");
    validateCudaInt32(freeCounts, globalIndices.device(), "free_counts");
    validateCudaInt32(allocationCounts, globalIndices.device(), "allocation_counts");
    validateCudaInt32(outputSlots, globalIndices.device(), "output_slots");
    validateCudaInt32(missCount, globalIndices.device(), "miss_count");
    TORCH_CHECK(cacheValues.is_cuda() && cacheValues.device() == globalIndices.device()
            && cacheValues.scalar_type() == th::kUInt8 && cacheValues.is_contiguous(),
        "cache_values must be a contiguous CUDA uint8 tensor on the global_indices device");
    TORCH_CHECK(layerInGroup >= 0 && layerInGroup < kPrototypeLayers, "layer_in_group must be in [0, 3)");
    TORCH_CHECK(rowsPerRequest > 0 && rowsPerRequest <= std::numeric_limits<std::int32_t>::max(),
        "rows_per_request must be a positive int32 value");
    TORCH_CHECK(strideFactor > 0 && tokensPerBlock > 0, "stride_factor and tokens_per_block must be positive");
    TORCH_CHECK(globalIndices.numel() % rowsPerRequest == 0,
        "global_indices rows must be divisible by rows_per_request");

    TORCH_CHECK(cacheKeys.dim() == 2, "cache_keys must be rank 2");
    std::int64_t const maxRequests = cacheKeys.size(0);
    std::int64_t const slotsPerRequest = cacheKeys.size(1);
    std::int64_t const layerBytes = hostPool.size(1) / kPrototypeLayers;
    TORCH_CHECK(slotsPerRequest >= kWorkingSetCapacityMultiplier * rowsPerRequest,
        "cache_keys must provide at least 2 * rows_per_request slots per request");
    TORCH_CHECK(cacheVersions.sizes() == cacheKeys.sizes(), "cache_versions must match cache_keys");
    TORCH_CHECK(slotEpochs.sizes() == cacheKeys.sizes(), "slot_epochs must match cache_keys");
    TORCH_CHECK(freeSlots.sizes() == cacheKeys.sizes(), "free_slots must match cache_keys");
    TORCH_CHECK(freeCounts.dim() == 1 && freeCounts.size(0) == maxRequests,
        "free_counts must have shape [max_requests]");
    TORCH_CHECK(allocationCounts.dim() == 1 && allocationCounts.size(0) == maxRequests,
        "allocation_counts must have shape [max_requests]");
    TORCH_CHECK(cacheValues.dim() == 3 && cacheValues.size(0) == maxRequests
            && cacheValues.size(1) == slotsPerRequest && cacheValues.size(2) == layerBytes,
        "cache_values must have shape [max_requests, slots_per_request, layer_bytes]");
    TORCH_CHECK(rowToSlot.dim() == 1 && rowToSlot.size(0) == hostPool.size(0),
        "row_to_slot must have one entry per host-pool row");
    TORCH_CHECK(epoch.numel() == 1 && missCount.numel() == 1, "epoch and miss_count must be scalar tensors");
    TORCH_CHECK(outputSlots.numel() >= globalIndices.numel(),
        "output_slots must have at least one entry per global index");
    TORCH_CHECK(maxRequests <= std::numeric_limits<std::int32_t>::max(), "max_requests exceeds int32");
    TORCH_CHECK(slotsPerRequest <= std::numeric_limits<std::int32_t>::max(),
        "slots_per_request exceeds int32");
    TORCH_CHECK(globalIndices.numel() / rowsPerRequest <= maxRequests,
        "global_indices contains more requests than the working set");

    c10::cuda::CUDAGuard const deviceGuard(globalIndices.device());
    auto const stream = at::cuda::getCurrentCUDAStream(globalIndices.get_device()).stream();
    tk::invokeDsaKvCacheOffloadIncrementalGather(hostPool.data_ptr<std::uint8_t>(),
        globalIndices.data_ptr<std::int32_t>(), hostVersions.data_ptr<std::int32_t>(),
        cacheKeys.data_ptr<std::int32_t>(), cacheVersions.data_ptr<std::int32_t>(),
        rowToSlot.data_ptr<std::int32_t>(), epoch.data_ptr<std::int32_t>(),
        slotEpochs.data_ptr<std::int32_t>(), freeSlots.data_ptr<std::int32_t>(),
        freeCounts.data_ptr<std::int32_t>(), allocationCounts.data_ptr<std::int32_t>(),
        cacheValues.data_ptr<std::uint8_t>(), outputSlots.data_ptr<std::int32_t>(),
        missCount.data_ptr<std::int32_t>(), globalIndices.numel(), hostPool.size(0),
        static_cast<std::int32_t>(maxRequests), static_cast<std::int32_t>(rowsPerRequest),
        static_cast<std::int32_t>(slotsPerRequest), static_cast<std::int32_t>(hostPool.size(1)),
        static_cast<std::int32_t>(layerBytes),
        static_cast<std::int32_t>(layerInGroup), static_cast<std::int32_t>(strideFactor),
        static_cast<std::int32_t>(tokensPerBlock), static_cast<std::int32_t>(layerOffset), stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "dsa_kv_cache_offload_mirror(Tensor source_pool, Tensor global_indices, Tensor(a!) host_pool, "
        "Tensor(b!) host_versions, int stride_factor, int tokens_per_block, int layer_offset, "
        "int layer_in_group) -> ()");
    m.def(
        "dsa_kv_cache_offload_gather(Tensor host_pool, Tensor global_indices, Tensor(a!) output, "
        "int stride_factor, int tokens_per_block, int layer_offset, int layer_in_group) -> ()");
    m.def(
        "dsa_kv_cache_offload_incremental_gather(Tensor host_pool, Tensor global_indices, Tensor host_versions, "
        "Tensor(a!) cache_keys, Tensor(b!) cache_versions, Tensor(c!) row_to_slot, Tensor(d!) epoch, "
        "Tensor(e!) slot_epochs, Tensor(f!) free_slots, Tensor(g!) free_counts, "
        "Tensor(h!) allocation_counts, Tensor(i!) cache_values, Tensor(j!) output_slots, "
        "Tensor(k!) miss_count, int rows_per_request, "
        "int stride_factor, int tokens_per_block, int layer_offset, int layer_in_group) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("dsa_kv_cache_offload_mirror", &tensorrt_llm::torch_ext::dsaKvCacheOffloadMirror);
    m.impl("dsa_kv_cache_offload_gather", &tensorrt_llm::torch_ext::dsaKvCacheOffloadGather);
    m.impl("dsa_kv_cache_offload_incremental_gather",
        &tensorrt_llm::torch_ext::dsaKvCacheOffloadIncrementalGather);
}
