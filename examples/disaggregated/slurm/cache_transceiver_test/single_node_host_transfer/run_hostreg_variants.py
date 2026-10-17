#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Same-node NIXL write from a GPU buffer into KVCM-style pinned host memory.

The destination is allocated the way KVCacheManagerV2 allocates a host-backed
sparse pool: an anonymous private mmap pinned with cudaHostRegister(PORTABLE |
MAPPED), the runtime equivalent of cuMemHostRegister(PORTABLE | DEVICEMAP) in
kv_cache_manager_v2/utils/hostMem.cpp. The agents use the production settings of
transfer.py's _create_nixl_agent: a progress thread and TRTLLM_NIXL_NUM_THREADS
(default 8) threads.

Modes:
  gpu_to_hostreg          destination registered and addressed as DRAM with device 0,
                          which is what a host-aware transfer (PR 20061, R3) does.
  gpu_to_hostreg_as_vram  the same host memory registered and addressed as VRAM with
                          the GPU id, which is what the branch's transfer does today.

Environment: ZMQ_PORT, CTX_NODE. Run one process per role on different GPUs.
"""

import argparse
import json
import mmap
import os
import sys
import time
import traceback

import numpy as np
import torch
import zmq

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from tensorrt_llm._torch.disaggregation.base.agent import (
    MemoryDescs,
    MemoryType,
    RegMemoryDescs,
    TransferOp,
    TransferRequest,
)
from tensorrt_llm._torch.disaggregation.nixl.agent import NixlTransferAgent

MODES = {"gpu_to_hostreg": "DRAM", "gpu_to_hostreg_as_vram": "VRAM"}
HOST_REGISTER_PORTABLE = 0x01
HOST_REGISTER_MAPPED = 0x02


def _cuda_ok(result, what):
    err = result[0] if isinstance(result, tuple) else result
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{what} failed: {err}")
    return result[1:] if isinstance(result, tuple) else ()


def _pattern(size, seed):
    return ((np.arange(size, dtype=np.uint32) * 17 + seed * 31) % 251).astype(np.uint8)


def _pointer_attributes(ptr):
    attrs = _cuda_ok(cudart.cudaPointerGetAttributes(ptr), "cudaPointerGetAttributes")[0]
    return {
        "type": str(attrs.type),
        "device": int(attrs.device),
        "device_pointer_equals_host": int(attrs.devicePointer) == ptr,
    }


def _make_agent(name):
    return NixlTransferAgent(
        name,
        True,
        num_threads=int(os.environ.get("TRTLLM_NIXL_NUM_THREADS", "8")),
        rank=0,
        world_size=1,
    )


def run(role, mode, size, samples, timeout_ms, out_path):
    label = MODES[mode]
    device = torch.cuda.current_device()
    port = int(os.environ["ZMQ_PORT"])
    context = zmq.Context()
    socket = context.socket(zmq.REP if role == "ctx" else zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms + 60000)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms + 60000)
    if role == "ctx":
        socket.bind(f"tcp://*:{port}")
    else:
        socket.connect(f"tcp://{os.environ['CTX_NODE']}:{port}")

    name = f"hostreg_{role}_{mode}_{port}"
    peer = f"hostreg_{'gen' if role == 'ctx' else 'ctx'}_{mode}_{port}"
    agent = _make_agent(name)
    record = {"mode": mode, "label": label, "ucx_tls": os.environ.get("UCX_TLS", "<unset>")}
    unsafe_to_release = False
    expected = _pattern(size, seed=7)

    if role == "ctx":
        src = torch.from_numpy(expected.copy()).to(f"cuda:{device}")
        torch.cuda.synchronize()
        registration = RegMemoryDescs("VRAM", [(src.data_ptr(), size, device, name)])
        agent.register_memory(registration)
        remote = socket.recv_pyobj()
        socket.send_pyobj({"agent": bytes(agent.get_local_agent_desc())})
        load_error = None
        if remote.get("reg_error"):
            print(f"[ctx] gen registration failed: {remote['reg_error']}", flush=True)
        else:
            try:
                agent.load_remote_agent(peer, remote["agent"])
            except Exception as error:  # reported to the receiver with every sample
                load_error = f"load_remote_agent: {type(error).__name__}: {error}"
                traceback.print_exc()
        while True:
            message = socket.recv_pyobj()
            if message == "done":
                socket.send_pyobj("done")
                break
            _, index = message
            reply = {"index": index}
            if load_error is not None:
                reply["error"] = load_error
                socket.send_pyobj(reply)
                continue
            try:
                request = TransferRequest(
                    TransferOp.WRITE,
                    MemoryDescs.from_arrays_uniform_device(
                        MemoryType.VRAM,
                        np.asarray([src.data_ptr()], dtype=np.int64),
                        np.asarray([size], dtype=np.int64),
                        device,
                    ),
                    MemoryDescs.from_arrays_uniform_device(
                        getattr(MemoryType, label),
                        np.asarray([remote["ptr"]], dtype=np.int64),
                        np.asarray([size], dtype=np.int64),
                        remote["device"],
                    ),
                    peer,
                    None,
                )
                start = time.perf_counter()
                unsafe_to_release = True
                status = agent.submit_transfer_requests(request)
                completed = status.wait(timeout_ms=timeout_ms)
                reply["seconds"] = time.perf_counter() - start
                reply["completed"] = bool(completed and status.is_completed())
                if not reply["completed"]:
                    reply["error"] = f"physical completion unproven: {status.last_status_str()}"
                else:
                    unsafe_to_release = False
            except Exception as error:  # report every failure kind to the receiver
                reply["error"] = f"{type(error).__name__}: {error}"
                traceback.print_exc()
            socket.send_pyobj(reply)
        if unsafe_to_release:
            sys.stdout.flush()
            os._exit(3)
        agent.deregister_memory(registration)
        agent.shutdown()
        return

    # gen: allocate the destination like a KVCM host-backed pool.
    mapping = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    dst = np.frombuffer(mapping, dtype=np.uint8)
    ptr = int(dst.ctypes.data)
    _cuda_ok(
        cudart.cudaHostRegister(ptr, size, HOST_REGISTER_PORTABLE | HOST_REGISTER_MAPPED),
        "cudaHostRegister",
    )
    record["pointer_attributes"] = _pointer_attributes(ptr)
    reg_device = 0 if label == "DRAM" else device
    registration = RegMemoryDescs(label, [(ptr, size, reg_device, name)])
    local = {"ptr": ptr, "device": reg_device}
    try:
        agent.register_memory(registration)
        record["registered"] = True
        local["agent"] = bytes(agent.get_local_agent_desc())
    except Exception as error:  # record the registration outcome, whatever it is
        record["registered"] = False
        record["reg_error"] = f"{type(error).__name__}: {error}"
        local["reg_error"] = record["reg_error"]
        traceback.print_exc()
    socket.send_pyobj(local)
    ctx_info = socket.recv_pyobj()
    results = []
    if record["registered"]:
        try:
            agent.load_remote_agent(peer, ctx_info["agent"])
        except Exception as error:  # the write direction only needs ctx to load gen
            record["gen_load_error"] = f"{type(error).__name__}: {error}"
            traceback.print_exc()
        for index in range(samples):
            dst[:] = 0
            socket.send_pyobj(("go", index))
            reply = socket.recv_pyobj()
            sample = dict(reply)
            if reply.get("error"):
                unsafe_to_release = True
                results.append(sample)
                break
            wrong = int(np.count_nonzero(dst != expected))
            sample["wrong_bytes"] = wrong
            sample["GBps"] = size / reply["seconds"] / 1e9
            results.append(sample)
    socket.send_pyobj("done")
    socket.recv_pyobj()
    record["samples"] = results
    record["ok"] = (
        record["registered"]
        and len(results) == samples
        and all(not s.get("error") and s.get("wrong_bytes") == 0 for s in results)
    )
    timed = [s["GBps"] for s in results[1:] if "GBps" in s]
    record["median_GBps"] = float(np.median(timed)) if timed else None
    with open(out_path, "w") as stream:
        json.dump(record, stream, indent=2)
    print(f"[gen] {mode}: {json.dumps({k: v for k, v in record.items() if k != 'samples'})}")
    if unsafe_to_release:
        sys.stdout.flush()
        os._exit(3)
    if record["registered"]:
        agent.deregister_memory(registration)
    agent.shutdown()
    _cuda_ok(cudart.cudaHostUnregister(ptr), "cudaHostUnregister")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("ctx", "gen"), required=True)
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--bytes", type=int, default=16 << 20)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--out", default="hostreg.json")
    args = parser.parse_args()
    torch.cuda.set_device(0)
    run(args.role, args.mode, args.bytes, args.samples, args.timeout_ms, args.out)


if __name__ == "__main__":
    main()
