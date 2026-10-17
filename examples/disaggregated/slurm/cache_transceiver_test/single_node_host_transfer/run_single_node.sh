#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single-node NIXL/UCX probe: can one process write a GPU buffer into another process's
# pinned host memory, as a disaggregated decode worker with host-backed KV pools needs?
#
# Run inside the TensorRT-LLM container of this revision on one node with at least two GPUs.
# For every GPU pair in GPU_PAIRS and every UCX setting the disaggregated tests use, it runs
# ../run_host_transfer_benchmark.py unchanged and run_hostreg_variants.py, then writes
# summary.txt and results.json into the run directory.
#
# Environment:
#   WORK_ROOT         Writable directory for the venv, the wheel and the run directories
#                     (default: $PWD/ucx_host_transfer).
#   GPU_PAIRS         Space-separated ctx:gen indexes into the visible GPUs (default "0:1").
#   TRTLLM_INSTALLED  Set to 1 to use the tensorrt_llm the container already has.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)
BENCH_DIR=$(cd "$HERE/.." && pwd)
WORK_ROOT=${WORK_ROOT:-$PWD/ucx_host_transfer}
GPU_PAIRS=${GPU_PAIRS:-0:1}
ARCH=$(uname -m)
RUN=$WORK_ROOT/run_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}
mkdir -p "$RUN"
exec > >(tee -a "$RUN/job.log") 2>&1
# Never run Python from the repository root: it would import the in-tree tensorrt_llm.
cd "$RUN"
# Keep every cache off the home directory, which is small or full on many clusters.
export HOME=$WORK_ROOT/home PIP_CACHE_DIR=$WORK_ROOT/pipcache XDG_CACHE_HOME=$WORK_ROOT/home/.cache
mkdir -p "$HOME"
export PYTHONDONTWRITEBYTECODE=1 TLLM_LOG_LEVEL=info

finish() {
    echo "exit_code=$1" > "$RUN/DONE"
    echo "run directory: $RUN"
    exit "$1"
}

echo "== $(hostname) arch=$ARCH job=${SLURM_JOB_ID:-local} $(date -Is)"
echo "repo $REPO at $(git -C "$REPO" log -1 --format='%h %s' 2>/dev/null)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} GPU_PAIRS=$GPU_PAIRS"
nvidia-smi --query-gpu=index,uuid,name,compute_cap,driver_version,pci.bus_id --format=csv
nvidia-smi topo -m
ucx_info -v 2>&1 | head -5
ucx_info -d 2>/dev/null | grep -E "^#\s+(Transport|Device):" | paste - - | sort | uniq -c
ls -d /opt/nvidia/nvda_nixl* 2>/dev/null
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"

# The post-merge wheel closest to this revision that matches the image's NGC PyTorch. The
# probe needs only its NIXL transfer agent, so any recent wheel answers the question.
if [ "${TRTLLM_INSTALLED:-0}" != 1 ]; then
    torch_tag=$(python3 -c "import importlib.metadata as m; print(m.version('torch'))" | tr -c 'A-Za-z0-9.\n' '_')
    VENV=$WORK_ROOT/venv-$ARCH-$torch_tag
    WHEEL_DIR=$WORK_ROOT/wheel-$ARCH-$torch_tag
    if [ ! -x "$VENV/bin/python3" ]; then
        python3 -m venv --system-site-packages "$VENV" || finish 10
    fi
    . "$VENV/bin/activate"
    if ! python3 -c "import tensorrt_llm" >/dev/null 2>&1; then
        whl=$(ls -t "$WHEEL_DIR"/*.whl 2>/dev/null | head -1)
        if [ -z "$whl" ]; then
            python3 "$HERE/download_closest_wheel.py" --repo "$REPO" --output-dir "$WHEEL_DIR" || finish 11
            whl=$(ls -t "$WHEEL_DIR"/*.whl | head -1)
        fi
        pip install --no-deps "$whl" || finish 12
        # Pin the image's torch by its installed distribution version. torch.__version__ can
        # differ from it in the local label (nv26.08 vs nv26.8.<build>), which pip cannot match.
        python3 -c "import importlib.metadata as m; print('torch==' + m.version('torch'))" > "$VENV/constraints.txt"
        python3 -c "import importlib.metadata as m; print('\n'.join(r for r in m.requires('tensorrt_llm') or [] if 'extra ==' not in r))" > "$VENV/trtllm-requirements.txt"
        pip install -c "$VENV/constraints.txt" -r "$VENV/trtllm-requirements.txt" || finish 13
    fi
fi
python3 -c "import tensorrt_llm, os; print('tensorrt_llm', tensorrt_llm.__version__, os.path.dirname(tensorrt_llm.__file__))" || finish 14
python3 -c "from tensorrt_llm._torch.disaggregation.nixl.agent import NixlTransferAgent as A; print('agent class', A)" || finish 15

# Visible GPUs as UUIDs, so a pair survives any device renumbering inside the container.
mapfile -t GPUS < <(python3 - <<'EOF'
import os, subprocess
rows = [r.split(", ") for r in subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True).strip().splitlines()]
by_index = dict(rows)
visible = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
picked = [v if v.startswith(("GPU-", "MIG-")) else by_index.get(v) for v in visible]
picked = [p for p in picked if p]
if len(picked) < 2:
    picked = [uuid for _, uuid in rows]
print("\n".join(picked))
EOF
)
echo "visible GPUs: ${GPUS[*]}"
[ "${#GPUS[@]}" -ge 2 ] || finish 20

cat > "$RUN/config.yaml" <<EOF
environment:
  work_dir: "$RUN"
host_transfer_benchmark:
  enabled: true
  modes: ["gpu", "host", "host_to_gpu", "gpu_to_host"]
  bytes_per_transfer: 16777216
  warmup: 2
  samples: 8
  delayed_completion_ms: 100
  timeout_ms: 30000
EOF
export CTT_CONFIG="$RUN/config.yaml" CTX_NODE=127.0.0.1 SLURM_LOCALID=0

# UCX settings of the disaggregated tests: the accuracy harness on NVLink nodes, the unit-test
# transfer harness, the UCX default, and the accuracy harness without NVLink.
SWEEP_NAMES=(harness_nvlink unit_harness default harness_no_nvlink)
SWEEP_ENVS=("UCX_TLS=^ib UCX_MM_ERROR_HANDLING=y" "UCX_TLS=^ib,gdr_copy" "" "UCX_TLS=^cuda_ipc UCX_MM_ERROR_HANDLING=y")

run_sweep() {
    local sid=$1 pair=$2 ctx_gpu=$3 gen_gpu=$4 name=$5
    shift 5
    local tag="p${pair}_s${sid}_${name}"
    printf '%s\t%s\t%s\t%s\n' "$sid" "$pair" "$name" "$*" >> "$RUN/sweeps.tsv"
    echo "== sweep $sid, GPU pair $pair, $name: ${*:-UCX defaults}"
    (
        for kv in "$@"; do export "$kv"; done
        export CTT_SWEEP=$sid ZMQ_PORT=$((22000 + 40 * sid)) UCX_PROTO_INFO=y
        CUDA_VISIBLE_DEVICES=$ctx_gpu timeout -k 10 300 python3 "$BENCH_DIR/run_host_transfer_benchmark.py" \
            --role ctx > "$RUN/$tag.pr.ctx.log" 2>&1 &
        local cpid=$!
        CUDA_VISIBLE_DEVICES=$gen_gpu timeout -k 10 300 python3 "$BENCH_DIR/run_host_transfer_benchmark.py" \
            --role gen > "$RUN/$tag.pr.gen.log" 2>&1
        local grc=$?
        wait $cpid
        echo "  PR benchmark: ctx_rc=$? gen_rc=$grc"
        local port=$((ZMQ_PORT + 20))
        for mode in gpu_to_hostreg gpu_to_hostreg_as_vram; do
            port=$((port + 1))
            ZMQ_PORT=$port CUDA_VISIBLE_DEVICES=$ctx_gpu timeout -k 10 180 python3 "$HERE/run_hostreg_variants.py" \
                --role ctx --mode $mode > "$RUN/$tag.$mode.ctx.log" 2>&1 &
            cpid=$!
            ZMQ_PORT=$port CUDA_VISIBLE_DEVICES=$gen_gpu timeout -k 10 180 python3 "$HERE/run_hostreg_variants.py" \
                --role gen --mode $mode --out "$RUN/$tag.$mode.json" > "$RUN/$tag.$mode.gen.log" 2>&1
            grc=$?
            wait $cpid
            echo "  $mode: ctx_rc=$? gen_rc=$grc"
        done
    )
}

pair_index=0
for pair in $GPU_PAIRS; do
    ctx_index=${pair%%:*}
    gen_index=${pair##*:}
    ctx_gpu=${GPUS[$ctx_index]:-}
    gen_gpu=${GPUS[$gen_index]:-}
    if [ -z "$ctx_gpu" ] || [ -z "$gen_gpu" ] || [ "$ctx_index" = "$gen_index" ]; then
        echo "skipping GPU pair $pair: needs two distinct visible GPUs"
        continue
    fi
    for gpu in "$ctx_gpu" "$gen_gpu"; do
        CUDA_VISIBLE_DEVICES=$gpu python3 -c "import torch; assert torch.cuda.device_count() == 1" || finish 21
    done
    for s in "${!SWEEP_NAMES[@]}"; do
        # shellcheck disable=SC2086  # each entry splits into KEY=VALUE words
        run_sweep $((pair_index * 10 + s)) "$ctx_index-$gen_index" "$ctx_gpu" "$gen_gpu" "${SWEEP_NAMES[$s]}" ${SWEEP_ENVS[$s]}
    done
    pair_index=$((pair_index + 1))
done

python3 "$HERE/summarize.py" "$RUN" | tee "$RUN/summary.txt"
finish 0
