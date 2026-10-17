<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Single-node host-memory KV transfer probe

A decode worker with host-backed sparse KV pools (DeepSeek-V4 sparse KV offload) receives
prompt history directly into pinned host memory. This probe answers two questions on one node:

1. Can a NIXL write from a GPU buffer in one process land in another process's pinned host
   memory, under the UCX settings the disaggregated tests use?
2. What happens when that host memory is registered as `VRAM`, which the disaggregated
   transfer does for every KV pool today?

## What runs

For every GPU pair in `GPU_PAIRS` and every UCX setting below, `run_single_node.sh` runs two
writer/receiver process pairs (ctx on the first GPU, gen on the second):

| Script | Destination memory | Labels | Agent |
|---|---|---|---|
| `../run_host_transfer_benchmark.py` (PR 20061, unchanged) | `torch` pinned (`cudaHostAlloc`) or GPU | true kinds; modes `gpu`, `host`, `host_to_gpu`, `gpu_to_host` | 1 thread |
| `run_hostreg_variants.py` | anonymous `mmap` + `cudaHostRegister(PORTABLE \| MAPPED)`, as KVCacheManagerV2 allocates host pools | `gpu_to_hostreg`: DRAM, device 0 (host-aware transfer); `gpu_to_hostreg_as_vram`: VRAM, GPU id (today's transfer) | production settings: progress thread, 8 threads |

Every received byte is checked against a pattern.

| UCX setting | Environment | Used by |
|---|---|---|
| `harness_nvlink` | `UCX_TLS=^ib UCX_MM_ERROR_HANDLING=y` | `tests/integration/defs/accuracy/test_disaggregated_serving.py` on NVLink nodes |
| `unit_harness` | `UCX_TLS=^ib,gdr_copy` | `tests/unittest/disaggregated/kv_transfer_harness.py` |
| `default` | none | UCX defaults |
| `harness_no_nvlink` | `UCX_TLS=^cuda_ipc UCX_MM_ERROR_HANDLING=y` | the accuracy harness without NVLink |

`UCX_PROTO_INFO=y` records the protocol UCX picks for each kind of write.

## Run it

The scripts install the post-merge wheel closest to this checkout that matches the container's
NGC PyTorch (`download_closest_wheel.py`), into a venv under `WORK_ROOT`. The checkout needs its
Git history for that lookup: clone without `--depth` (`--filter=blob:none` is fine). If the
checkout is a `git worktree`, also mount the main repository's `.git`.

Slurm with Pyxis (GB200/GB300, aarch64; four GPUs per node, so test a same-socket and a
cross-socket pair):

```bash
WORK_ROOT=/path/to/shared/scratch/ucx_host_transfer ACCOUNT=<account> PARTITION=<partition> \
GPUS=4 GPU_PAIRS="0:1 0:2" bash submit_slurm_pyxis.sh
```

`IMAGE` defaults to `LLM_SBSA_DOCKER_IMAGE` from `jenkins/current_image_tags.properties`; pulling
it needs enroot credentials for artifactory. Set `IMAGE=<file>.sqsh` to use an imported image.

ComputeLab (`crun`), from an enroot squashfs:

```bash
SQSH=/path/to/image.sqsh WORK_ROOT=/path/to/scratch/ucx_host_transfer \
QUERY='gpu.product_name="NVIDIA B200" and cpu.arch=x86_64' GPUS=2 \
bash submit_computelab_crun.sh
```

Already inside the container on a GPU node: `WORK_ROOT=<dir> GPU_PAIRS="0:1" bash run_single_node.sh`.

## Read the result

Each run writes `$WORK_ROOT/run_<time>_<job>/`:

- `summary.txt`: both tables, the UCX protocol per sweep and a verdict.
- `results.json`: the same data; `verdict` holds the answers.
- `job.log`: environment (GPUs, topology, UCX version and transports, wheel), then progress.
- `p<pair>_s<sweep>_<setting>.*.log`: per-process logs with the UCX protocol tables.

| Verdict key | Meaning |
|---|---|
| `dram_labelled_gpu_to_host_write_works` | the host-aware transfer works on this node |
| `vram_labelled_host_registration_fails` | today's transfer cannot register host-backed pools |
| `gpu_to_host_GBps`, `gpu_to_gpu_GBps`, `host_to_host_GBps` | median transport bandwidth per pair and setting |
| `cuda_to_host_protocol` | the UCX protocol for a GPU-to-host write, e.g. `software emulation` over `sysv/memory` |

Reference: B200 (x86_64), image `pytorch-26.08-py3-...-202609210610-19345`, wheel build 3006,
GPUs 0 and 1 of one node, 2026-10-16:

| Setting | GPU→GPU GB/s | GPU→host GB/s | host→host GB/s | KVCM-style DRAM label | VRAM label |
|---|---|---|---|---|---|
| `harness_nvlink` | 278 | 0.62 | 8.7 | works, 0.63 GB/s | registration fails |
| `unit_harness` | 270 | 0.39 | 8.7 | works, 0.41 GB/s | registration fails |
| `default` | 278 | 0.40 | 8.5 | works, 0.41 GB/s | registration fails |
| `harness_no_nvlink` | 0.63 | 0.63 | 8.8 | works, 0.63 GB/s | registration fails |

Every byte arrived correctly. UCX moved each GPU-to-host write by software emulation over
`sysv/memory`, which is why it is three orders of magnitude slower than GPU-to-GPU.
