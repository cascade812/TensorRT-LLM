#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run run_single_node.sh on ComputeLab through crun.
#
#   SQSH=<image .sqsh> WORK_ROOT=<writable dir> \
#   QUERY='gpu.product_name="NVIDIA B200" and cpu.arch=x86_64' [GPUS=2] [GPU_PAIRS="0:1"] \
#   bash submit_computelab_crun.sh
#
# crun's Docker path pulls without credentials, and artifactory refuses that, so the batch job
# starts a Pyxis container from an enroot squashfs instead. Make one once with
#   enroot import -o <file>.sqsh 'docker://artifactory.nvidia.com#<path>:<tag>'
# using the image jenkins/current_image_tags.properties records for the node's CPU.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)

if [ "${1:-}" = "--in-job" ]; then
    # Inside the batch job: the settings come from the file the submit step wrote.
    # shellcheck disable=SC1090
    . "$2"
    exec srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
        --container-image="$SQSH" --container-mounts="$MOUNTS" --container-workdir="$WORK_ROOT" \
        --export=ALL,WORK_ROOT="$WORK_ROOT",GPU_PAIRS="$GPU_PAIRS" \
        bash "$HERE/run_single_node.sh"
fi

: "${SQSH:?set SQSH to an enroot squashfs of the TensorRT-LLM image}"
: "${WORK_ROOT:?set WORK_ROOT to a writable directory}"
: "${QUERY:?set QUERY to a crun node query}"
mkdir -p "$WORK_ROOT"
settings=$WORK_ROOT/crun_settings_$(date +%Y%m%d_%H%M%S).sh
cat > "$settings" <<EOF
SQSH='$SQSH'
WORK_ROOT='$WORK_ROOT'
GPU_PAIRS='${GPU_PAIRS:-0:1}'
MOUNTS='${MOUNTS:-$REPO:$REPO,$WORK_ROOT:$WORK_ROOT}'
EOF
crun -q "$QUERY" --gpus "${GPUS:-2}" -sa --cpus-per-task=32 -b -t "${TIME:-01:00:00}" -r "$WORK_ROOT" \
    bash "$HERE/submit_computelab_crun.sh" --in-job "$settings"
