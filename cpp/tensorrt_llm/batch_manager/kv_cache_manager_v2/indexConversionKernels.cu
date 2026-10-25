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

#include "indexConversionKernels.h"

#include "common.h"

#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_runtime_api.h>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

constexpr int kThreads = 256;

//! One CTA row per (table, buffer, row); threads walk the blocks of that row.
__global__ __launch_bounds__(kThreads) void indexConversionKernel(
    IndexTableDesc const* descs, int32_t numRows, int32_t blocksPerRow, int32_t rawRowStride)
{
    auto const& desc = descs[blockIdx.z];
    int32_t const buffer = static_cast<int32_t>(blockIdx.y);
    if (buffer >= desc.numBuffers)
    {
        return;
    }
    int32_t const row = static_cast<int32_t>(blockIdx.x);

    int32_t const* raw = desc.raw + static_cast<ptrdiff_t>(row) * rawRowStride;
    // Row-major in the entries: a row's entries are adjacent, so a request's K and V
    // sit together the way an attention kernel's page list wants them.
    int32_t* out = desc.converted + (static_cast<ptrdiff_t>(row) * desc.numBuffers + buffer) * blocksPerRow;
    int32_t const offset = desc.offsets[buffer];

    for (int32_t block = static_cast<int32_t>(threadIdx.x); block < blocksPerRow; block += kThreads)
    {
        int32_t const slot = raw[block];
        // An absent block stays absent: folding it to a safe slot would read page zero.
        out[block] = slot == kBadPageIndex.value() ? kBadPageIndex.value() : slot * desc.scale + offset;
    }
}

//! One CTA row per (table, buffer, row), as above, but scratch blocks take their slot from
//! the request rather than the page table.
__global__ __launch_bounds__(kThreads) void indexConversionScratchKernel(
    IndexTableDesc const* descs, int32_t numRows, int32_t blocksPerRow, int32_t rawRowStride)
{
    auto const& desc = descs[blockIdx.z];
    int32_t const buffer = static_cast<int32_t>(blockIdx.y);
    if (buffer >= desc.numBuffers)
    {
        return;
    }
    int32_t const row = static_cast<int32_t>(blockIdx.x);

    int32_t const* raw = desc.raw + static_cast<ptrdiff_t>(row) * rawRowStride;
    int32_t* out = desc.converted + (static_cast<ptrdiff_t>(row) * desc.numBuffers + buffer) * blocksPerRow;
    int32_t const offset = desc.offsets[buffer];

    int32_t const scratchBeg = desc.scratchBegs[row];
    int32_t const scratchEnd = desc.scratchEnds[row];
    int32_t const* scratchSlots = desc.scratchSlots + static_cast<ptrdiff_t>(row) * desc.scratchSlotStride;

    for (int32_t block = static_cast<int32_t>(threadIdx.x); block < blocksPerRow; block += kThreads)
    {
        if (block >= scratchBeg && block < scratchEnd)
        {
            // The slot cursor advances by the sub-pages each scratch block occupies. The
            // layer's own offset wraps inside the slot: a scratch slot is shared, so the
            // sub-page positions past it are already spoken for.
            int32_t const total = (block - scratchBeg) * desc.scratchPagesPerBlock;
            int32_t const slotId = scratchSlots[total / desc.scale];
            out[block] = slotId == kBadPageIndex.value()
                ? kBadPageIndex.value()
                : slotId * desc.scale + (total % desc.scale + offset) % desc.scale;
            continue;
        }
        int32_t const slot = raw[block];
        out[block] = slot == kBadPageIndex.value() ? kBadPageIndex.value() : slot * desc.scale + offset;
    }
}

} // namespace

void launchIndexConversion(IndexTableDesc const* descs, int32_t numTables, int32_t maxBuffers, int32_t numRows,
    int32_t blocksPerRow, int32_t rawRowStride, CUstream stream)
{
    if (numTables <= 0 || maxBuffers <= 0 || numRows <= 0 || blocksPerRow <= 0)
    {
        return;
    }
    // The grid's buffer extent is the widest table's, since descs lives on the device and
    // the host cannot read it back to size the grid per table.
    dim3 const grid(
        static_cast<unsigned>(numRows), static_cast<unsigned>(maxBuffers), static_cast<unsigned>(numTables));
    indexConversionKernel<<<grid, kThreads, 0, stream>>>(descs, numRows, blocksPerRow, rawRowStride);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void launchIndexConversionWithScratch(IndexTableDesc const* descs, int32_t numTables, int32_t maxBuffers,
    int32_t numRows, int32_t blocksPerRow, int32_t rawRowStride, CUstream stream)
{
    if (numTables <= 0 || maxBuffers <= 0 || numRows <= 0 || blocksPerRow <= 0)
    {
        return;
    }
    dim3 const grid(
        static_cast<unsigned>(numRows), static_cast<unsigned>(maxBuffers), static_cast<unsigned>(numTables));
    indexConversionScratchKernel<<<grid, kThreads, 0, stream>>>(descs, numRows, blocksPerRow, rawRowStride);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
