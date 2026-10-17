#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run run_single_node.sh on one node of a Slurm cluster with Pyxis/enroot.
#
#   WORK_ROOT=<writable shared dir> [ACCOUNT=<account>] [PARTITION=<partition>] \
#   [ARCH=aarch64|x86_64] [GPUS=4] [GPU_PAIRS="0:1 0:2"] [IMAGE=<image or .sqsh>] \
#   [EXTRA_SRUN_ARGS="..."] bash submit_slurm_pyxis.sh
#
# IMAGE defaults to the image this revision records in jenkins/current_image_tags.properties
# for ARCH (default aarch64: GB200/GB300), in Pyxis registry syntax. Pulling it needs enroot
# credentials for artifactory in ~/.config/enroot/.credentials; point IMAGE at a local .sqsh
# made by `enroot import` to skip the pull.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)
: "${WORK_ROOT:?set WORK_ROOT to a writable directory visible from the compute node}"
ARCH=${ARCH:-aarch64}
GPUS=${GPUS:-4}
GPU_PAIRS=${GPU_PAIRS:-0:1 0:2}
if [ -z "${IMAGE:-}" ]; then
    case "$ARCH" in
        x86_64) key=LLM_DOCKER_IMAGE ;;
        aarch64) key=LLM_SBSA_DOCKER_IMAGE ;;
        *) echo "unsupported ARCH: $ARCH" >&2; exit 1 ;;
    esac
    ref=$(sed -n "s/^${key}=//p" "$REPO/jenkins/current_image_tags.properties")
    IMAGE=${ref/\//#}  # Pyxis wants registry#path:tag
fi
mkdir -p "$WORK_ROOT"
MOUNTS=${MOUNTS:-$REPO:$REPO,$WORK_ROOT:$WORK_ROOT}
args=(-N1 --ntasks=1 --gpus-per-node="$GPUS" --time="${TIME:-01:00:00}")
[ -n "${ACCOUNT:-}" ] && args+=(-A "$ACCOUNT")
[ -n "${PARTITION:-}" ] && args+=(-p "$PARTITION")
echo "image: $IMAGE"
echo "mounts: $MOUNTS"
# shellcheck disable=SC2086  # EXTRA_SRUN_ARGS holds separate srun options
exec srun "${args[@]}" ${EXTRA_SRUN_ARGS:-} \
    --container-image="$IMAGE" --container-mounts="$MOUNTS" --container-workdir="$WORK_ROOT" \
    --export=ALL,WORK_ROOT="$WORK_ROOT",GPU_PAIRS="$GPU_PAIRS" \
    bash "$HERE/run_single_node.sh"
