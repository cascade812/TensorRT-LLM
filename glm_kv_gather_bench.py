#!/usr/bin/env python3
# Microbenchmark for the GLM-5.2 KV-offload device-driven host gather.
# See glm_kv_cache_offload.md §6 step 2 and §4.5.
#
# Measures: GPU kernel gathering scattered rows directly from pinned host memory
# (zero-copy over PCIe/C2C), with the task list in DEVICE memory (CUDA-graph-replayable,
# data-dependent addressing) — the exact mechanism the offload design requires.
#
# Sweeps: row size (576 B = layer-major, 1728 B = token-major per 4-layer group,
# 18432 B = 32-token block granularity), rows per call (miss-batch size),
# stream launch vs captured CUDA graph, and gather concurrent with a GEMM.
# Reference: cudaMemcpyAsync H2D peak on the same buffers.

import os

import torch
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

CUDA_SRC = r"""
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// One warp per row; each lane copies 16B grains. Row addresses come from a
// device-resident index list -> data-dependent, CUDA-graph replayable.
__global__ void gather_rows_kernel(const uint4* __restrict__ pool,
                                   const int32_t* __restrict__ idx,
                                   uint4* __restrict__ out,
                                   int64_t n_rows, int64_t row_grains)
{
    int64_t warp = (int64_t)(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int64_t n_warps = (int64_t)gridDim.x * (blockDim.x / 32);
    for (int64_t r = warp; r < n_rows; r += n_warps) {
        const uint4* src = pool + (int64_t)idx[r] * row_grains;
        uint4* dst = out + r * row_grains;
        for (int64_t g = lane; g < row_grains; g += 32)
            dst[g] = src[g];
    }
}

void gather(torch::Tensor pool, torch::Tensor idx, torch::Tensor out, int64_t row_bytes, int64_t max_blocks)
{
    TORCH_CHECK(pool.is_cpu() && pool.is_pinned(), "pool must be pinned CPU");
    TORCH_CHECK(idx.is_cuda() && idx.dtype() == torch::kInt32);
    TORCH_CHECK(out.is_cuda());
    TORCH_CHECK(row_bytes % 16 == 0);
    int64_t n_rows = idx.numel(), row_grains = row_bytes / 16;
    TORCH_CHECK(out.numel() >= n_rows * row_bytes);
    int threads = 256;  // 8 warps/block
    int blocks = (int)std::min<int64_t>((n_rows + 7) / 8, max_blocks);
    auto stream = at::cuda::getCurrentCUDAStream();
    gather_rows_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(pool.data_ptr()),
        idx.data_ptr<int32_t>(),
        reinterpret_cast<uint4*>(out.data_ptr()),
        n_rows, row_grains);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

ext = load_inline(
    name="glm_gather_bench_v2",
    cpp_sources="void gather(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);",
    cuda_sources=CUDA_SRC,
    functions=["gather"],
    verbose=False,
    extra_cuda_cflags=["-O3"],
)

POOL_BYTES = (
    2 << 30
) - 4096  # ~2 GiB pinned pool (~40x GPU L2; this node's driver rejects >=4 GiB single pins)
N_IDX_BUFS = 8
GRID = 16  # blocks; 16 (~12% of H100 SMs) saturates PCIe and minimizes GEMM interference


def time_events(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn(0)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for i in range(iters):
        fn(i)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 1e3 / iters  # seconds per call


def main():
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("allocating ~2 GiB pinned pool...", flush=True)
    pool = torch.empty(POOL_BYTES, dtype=torch.uint8, pin_memory=True)

    # cudaMemcpyAsync H2D peak reference
    big = 1 << 28  # 256 MiB
    dst = torch.empty(big, dtype=torch.uint8, device=dev)
    t = time_events(
        lambda i: dst.copy_(pool[(i % 7) * big : ((i % 7) + 1) * big], non_blocking=True)
    )
    peak = big / t / 1e9
    print(f"cudaMemcpyAsync H2D peak (256 MiB contiguous): {peak:.1f} GB/s\n")

    print(
        f"{'row layout':>28s} {'rows/call':>9s} {'MB/call':>8s} {'stream GB/s':>11s} "
        f"{'graph GB/s':>10s} {'% of memcpy':>11s}"
    )
    results = {}
    for row_bytes, label in [
        (576, "576B latent, layer-major"),
        (1728, "1728B 3-layer token-major"),
        (18432, "18KB 32-token block"),
    ]:
        n_pool_rows = POOL_BYTES // row_bytes
        for rows in (2048, 8192, 32768, 131072):
            if rows * row_bytes > (2 << 30):
                continue
            out = torch.empty(rows * row_bytes, dtype=torch.uint8, device=dev)
            idxs = [
                torch.randint(0, n_pool_rows, (rows,), dtype=torch.int32, device=dev)
                for _ in range(N_IDX_BUFS)
            ]
            t_s = time_events(
                lambda i: ext.gather(pool, idxs[i % N_IDX_BUFS], out, row_bytes, GRID)
            )
            bw_s = rows * row_bytes / t_s / 1e9

            # captured CUDA graph: static index buffer refreshed by D2D copy between replays
            static_idx = idxs[0].clone()
            g = torch.cuda.CUDAGraph()
            ext.gather(pool, static_idx, out, row_bytes, GRID)  # warm the allocator
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                ext.gather(pool, static_idx, out, row_bytes, GRID)

            def replay(i):
                static_idx.copy_(idxs[i % N_IDX_BUFS], non_blocking=True)
                g.replay()

            t_g = time_events(replay)
            bw_g = rows * row_bytes / t_g / 1e9
            results[(row_bytes, rows)] = bw_g
            print(
                f"{label:>28s} {rows:>9d} {rows * row_bytes / 1e6:>8.1f} {bw_s:>11.1f} "
                f"{bw_g:>10.1f} {100 * bw_g / peak:>10.0f}%"
            )

    # overlap check: gather (stream B) concurrent with GEMM (stream A)
    print("\noverlap: 1728B gather + bf16 GEMM(8192^3) on separate streams")
    row_bytes, rows = 1728, 32768
    out = torch.empty(rows * row_bytes, dtype=torch.uint8, device=dev)
    idx = torch.randint(0, POOL_BYTES // row_bytes, (rows,), dtype=torch.int32, device=dev)
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev)
    b = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev)
    t_mm = time_events(lambda i: a @ b, iters=20)
    t_ga = time_events(lambda i: ext.gather(pool, idx, out, row_bytes, GRID), iters=20)
    sB = torch.cuda.Stream()
    n_mm = max(int(t_ga / t_mm) + 1, 2)  # enough GEMMs to cover the gather duration

    def both(i):
        with torch.cuda.stream(sB):
            ext.gather(pool, idx, out, row_bytes, GRID)
        for _ in range(n_mm):
            a @ b
        torch.cuda.current_stream().wait_stream(sB)

    t_both = time_events(both, iters=20)
    print(
        f"  GEMM alone {t_mm * 1e3:.2f} ms | gather alone {t_ga * 1e3:.2f} ms "
        f"({rows * row_bytes / t_ga / 1e9:.1f} GB/s) | {n_mm} GEMMs + gather concurrent "
        f"{t_both * 1e3:.2f} ms -> compute slowdown {t_both / (n_mm * t_mm):.2f}x while gather active"
    )


if __name__ == "__main__":
    main()
