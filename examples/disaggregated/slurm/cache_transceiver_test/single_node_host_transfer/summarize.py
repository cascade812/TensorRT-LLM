#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize a run of run_single_node.sh into a table and results.json.

Usage: summarize.py <run directory>
"""

import glob
import json
import os
import re
import statistics
import sys

PR_MODES = ("gpu", "host", "host_to_gpu", "gpu_to_host")
VARIANT_MODES = ("gpu_to_hostreg", "gpu_to_hostreg_as_vram")
PUT_TITLE = re.compile(r"remote memory write by ucp_put\S* from (?P<src>\S+) to (?P<dst>\S+) \|")


def load_sweeps(run):
    sweeps = {}
    with open(os.path.join(run, "sweeps.tsv")) as stream:
        for line in stream:
            sid, pair, name, env = (line.rstrip("\n").split("\t") + [""] * 4)[:4]
            sweeps[int(sid)] = {"pair": pair, "ucx": name, "env": env}
    return sweeps


def median_rate(samples, seconds_key):
    rates = [s["bytes"] / s[seconds_key] / 1e9 for s in samples if s.get(seconds_key)]
    return round(statistics.median(rates), 3) if rates else None


def pr_rows(run, sweeps):
    found = {}
    for path in glob.glob(os.path.join(run, "host_transfer", "*.json")):
        with open(path) as stream:
            data = json.load(stream)
        timed = [s for s in data["samples"] if s["scenario"] == "throughput" and not s["warmup"]]
        found[(data["sweep"], data["mode"])] = {
            "bytes_correct": all(s["byte_correct"] for s in data["samples"]),
            "GBps": median_rate(timed, "seconds"),
            "gpu_ready_GBps": median_rate(timed, "gpu_ready_seconds"),
        }
    rows = []
    for sid, info in sorted(sweeps.items()):
        for mode in PR_MODES:
            result = found.get((sid, mode), {"bytes_correct": False, "missing": True})
            rows.append(
                {"sweep": sid, "pair": info["pair"], "ucx": info["ucx"], "mode": mode, **result}
            )
    return rows


def variant_rows(run, sweeps):
    rows = []
    for sid, info in sorted(sweeps.items()):
        tag = f"p{info['pair']}_s{sid}_{info['ucx']}"
        for mode in VARIANT_MODES:
            path = os.path.join(run, f"{tag}.{mode}.json")
            row = {"sweep": sid, "pair": info["pair"], "ucx": info["ucx"], "mode": mode}
            if not os.path.exists(path):
                rows.append({**row, "ok": False, "missing": True})
                continue
            with open(path) as stream:
                data = json.load(stream)
            error = data.get("reg_error") or next(
                (s["error"] for s in data.get("samples", []) if s.get("error")), None
            )
            rows.append(
                {
                    **row,
                    "label": data.get("label"),
                    "ok": bool(data.get("ok")),
                    "registered": data.get("registered"),
                    "GBps": round(data["median_GBps"], 3) if data.get("median_GBps") else None,
                    "error": error.splitlines()[0] if error else None,
                }
            )
    return rows


def put_protocols(run, sweeps):
    """Return the UCX protocol rows chosen for GPU-source writes, per sweep."""
    protocols = {}
    for sid, info in sorted(sweeps.items()):
        tag = f"p{info['pair']}_s{sid}_{info['ucx']}"
        tables = {}
        for path in sorted(glob.glob(os.path.join(run, f"{tag}.*.log"))):
            with open(path, errors="replace") as stream:
                lines = [line.split("|", 1)[-1] if "|" in line else "" for line in stream]
            for index, line in enumerate(lines):
                match = PUT_TITLE.search("|" + line)
                if not match or not match["src"].startswith("cuda"):
                    continue
                key = f"{match['src'].split('/')[0]}->{match['dst'].split('/')[0]}"
                if key in tables:
                    continue
                cells = []
                # Title, rule, header, rule, then rows until the closing rule.
                for row in lines[index + 4 :]:
                    if not row.strip() or not row.strip().endswith("|"):
                        break
                    cells.append(
                        " | ".join(cell.strip() for cell in row.strip().rstrip("|").split("|"))
                    )
                tables[key] = cells
        protocols[sid] = tables
    return protocols


def main():
    run = sys.argv[1]
    sweeps = load_sweeps(run)
    pr = pr_rows(run, sweeps)
    variants = variant_rows(run, sweeps)
    protocols = put_protocols(run, sweeps)

    print("PR 20061 run_host_transfer_benchmark.py (torch pinned host buffers, 1 NIXL thread)")
    print(
        f"  {'sweep':>5} {'pair':>5} {'ucx':18} {'mode':12} {'bytes ok':>8} {'GB/s':>8} {'GPU-ready GB/s':>14}"
    )
    for r in pr:
        rate = "missing" if r.get("missing") else r["GBps"]
        print(
            f"  {r['sweep']:>5} {r['pair']:>5} {r['ucx']:18} {r['mode']:12} "
            f"{str(r['bytes_correct']):>8} {str(rate):>8} {str(r.get('gpu_ready_GBps')):>14}"
        )
    print("\nKVCM-style host memory (mmap + cudaHostRegister), production agent settings")
    print(f"  {'sweep':>5} {'pair':>5} {'ucx':18} {'mode':24} {'ok':>5} {'GB/s':>8}  error")
    for r in variants:
        print(
            f"  {r['sweep']:>5} {r['pair']:>5} {r['ucx']:18} {r['mode']:24} {str(r['ok']):>5} "
            f"{str(r.get('GBps')):>8}  {'missing output' if r.get('missing') else r.get('error') or ''}"
        )
    print("\nUCX protocol for writes from GPU memory (UCX_PROTO_INFO)")
    for sid, tables in protocols.items():
        info = sweeps[sid]
        for key, cells in sorted(tables.items()):
            print(f"  sweep {sid} pair {info['pair']} {info['ucx']:18} {key:12} {'; '.join(cells)}")

    by_mode = lambda rows, mode: [r for r in rows if r["mode"] == mode]  # noqa: E731
    verdict = {
        "dram_labelled_gpu_to_host_write_works": bool(by_mode(variants, "gpu_to_hostreg"))
        and all(r["ok"] for r in by_mode(variants, "gpu_to_hostreg")),
        "pr_gpu_to_host_bytes_correct": bool(by_mode(pr, "gpu_to_host"))
        and all(r["bytes_correct"] for r in by_mode(pr, "gpu_to_host")),
        "vram_labelled_host_registration_fails": bool(by_mode(variants, "gpu_to_hostreg_as_vram"))
        and all(r.get("registered") is False for r in by_mode(variants, "gpu_to_hostreg_as_vram")),
        "gpu_to_host_GBps": {
            f"{r['pair']}/{r['ucx']}": r.get("GBps") for r in by_mode(pr, "gpu_to_host")
        },
        "gpu_to_gpu_GBps": {f"{r['pair']}/{r['ucx']}": r.get("GBps") for r in by_mode(pr, "gpu")},
        "host_to_host_GBps": {
            f"{r['pair']}/{r['ucx']}": r.get("GBps") for r in by_mode(pr, "host")
        },
        "cuda_to_host_protocol": {
            f"{sweeps[sid]['pair']}/{sweeps[sid]['ucx']}": tables.get("cuda->host")
            for sid, tables in protocols.items()
        },
    }
    print("\nVerdict")
    for key, value in verdict.items():
        print(f"  {key}: {value}")
    with open(os.path.join(run, "results.json"), "w") as stream:
        json.dump(
            {
                "sweeps": sweeps,
                "pr_benchmark": pr,
                "kvcm_style": variants,
                "protocols": protocols,
                "verdict": verdict,
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
