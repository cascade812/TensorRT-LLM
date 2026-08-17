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

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr std::int64_t kPrototypeLayers = 3;

void validateCommon(th::Tensor const& hostPool, th::Tensor const& globalIndices)
{
    TORCH_CHECK(hostPool.device().is_cpu() && hostPool.is_pinned(), "host_pool must be a pinned CPU tensor");
    TORCH_CHECK(hostPool.scalar_type() == th::kUInt8 && hostPool.dim() == 2 && hostPool.is_contiguous(),
        "host_pool must be a contiguous rank-2 uint8 tensor");
    TORCH_CHECK(globalIndices.is_cuda() && globalIndices.scalar_type() == th::kInt32 && globalIndices.is_contiguous(),
        "global_indices must be a contiguous CUDA int32 tensor");
    TORCH_CHECK(hostPool.size(1) % kPrototypeLayers == 0,
        "host_pool row width must be divisible by the three prototype layers");
}

} // namespace

void dsaKvCacheOffloadMirror(th::Tensor const& sourcePool, th::Tensor const& globalIndices, th::Tensor& hostPool,
    std::int64_t strideFactor, std::int64_t tokensPerBlock, std::int64_t layerOffset, std::int64_t layerInGroup)
{
    validateCommon(hostPool, globalIndices);
    TORCH_CHECK(sourcePool.is_cuda() && sourcePool.element_size() == 1 && sourcePool.is_contiguous(),
        "source_pool must be a contiguous CUDA tensor with one-byte elements");
    TORCH_CHECK(layerInGroup >= 0 && layerInGroup < kPrototypeLayers, "layer_in_group must be in [0, 3)");
    TORCH_CHECK(strideFactor > 0 && tokensPerBlock > 0, "stride_factor and tokens_per_block must be positive");

    std::int64_t const hostRowBytes = hostPool.size(1);
    std::int64_t const layerBytes = hostRowBytes / kPrototypeLayers;
    TORCH_CHECK(sourcePool.numel() % layerBytes == 0, "source_pool size must be a multiple of layer bytes");

    c10::cuda::CUDAGuard const deviceGuard(globalIndices.device());
    auto const stream = at::cuda::getCurrentCUDAStream(globalIndices.get_device()).stream();
    tk::invokeDsaKvCacheOffloadMirror(reinterpret_cast<std::uint8_t const*>(sourcePool.data_ptr()),
        globalIndices.data_ptr<std::int32_t>(), hostPool.data_ptr<std::uint8_t>(), globalIndices.numel(),
        hostPool.size(0), static_cast<std::int32_t>(hostRowBytes), static_cast<std::int32_t>(layerBytes),
        static_cast<std::int32_t>(layerInGroup), static_cast<std::int32_t>(strideFactor),
        static_cast<std::int32_t>(tokensPerBlock), static_cast<std::int32_t>(layerOffset), stream);
}

void dsaKvCacheOffloadGather(th::Tensor const& hostPool, th::Tensor const& globalIndices, th::Tensor& output,
    std::int64_t strideFactor, std::int64_t tokensPerBlock, std::int64_t layerOffset,
    std::int64_t layerInGroup)
{
    validateCommon(hostPool, globalIndices);
    TORCH_CHECK(output.is_cuda() && output.scalar_type() == th::kUInt8 && output.dim() == 2 && output.is_contiguous(),
        "output must be a contiguous rank-2 CUDA uint8 tensor");
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

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "dsa_kv_cache_offload_mirror(Tensor source_pool, Tensor global_indices, Tensor(a!) host_pool, "
        "int stride_factor, int tokens_per_block, int layer_offset, int layer_in_group) -> ()");
    m.def(
        "dsa_kv_cache_offload_gather(Tensor host_pool, Tensor global_indices, Tensor(a!) output, "
        "int stride_factor, int tokens_per_block, int layer_offset, int layer_in_group) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("dsa_kv_cache_offload_mirror", &tensorrt_llm::torch_ext::dsaKvCacheOffloadMirror);
    m.impl("dsa_kv_cache_offload_gather", &tensorrt_llm::torch_ext::dsaKvCacheOffloadGather);
}
