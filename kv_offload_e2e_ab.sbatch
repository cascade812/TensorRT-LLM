#!/bin/bash
# End-to-end correctness A/B for the DSA KV-offload consume-mode prototype on
# GLM-5.2 (production-shaped config: TEP + attention DP, MTP3, CUTEDSL MoE,
# DSA heuristic top-k, FP8 KV, CUDA graphs, chunked prefill).
#
#   baseline (offload off)  x2   -> strict oracle + run-to-run stability probe
#   offload Setup 1 (2,3,4)      -> greedy tokens must match baseline exactly
#   offload Setup 2 (3,4)        -> greedy tokens must match baseline exactly
#
# TP/EP derive from the node count (4 GPUs per GB300 node): -N 2 => tp8/ep8
# (the pasted config), -N 1 => tp4/ep4 on exactly 4xB300.
#
# Main log: slurm-<jobid>.out; per-run logs + JSONs + verdict in
# kv_offload_job_logs/<jobid>/.
#SBATCH -A coreai_comparch_trtllm
#SBATCH -p gb300
#SBATCH -N 1
#SBATCH -J coreai_comparch_trtllm-kv_offload.e2e_ab
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fsw/coreai_comparch_trtllm/guijuz/TensorRT-LLM/slurm-%j.out

set -uo pipefail

export REPO=/lustre/fsw/coreai_comparch_trtllm/guijuz/TensorRT-LLM
export MODEL_DIR=/lustre/share/coreai_dlalgo_ci/artifacts/model/nvidia_glm-5.2-nvfp4/hf/hf-b0b2b68_orig
export WHEEL="${REPO}/build/tensorrt_llm-1.3.0rc25-2-cp312-cp312-linux_aarch64.whl"
export LOG_DIR="${REPO}/kv_offload_job_logs/${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"

GPUS_PER_NODE=4
TP=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))
EP=${TP}

banner() { printf '\n==== [%s] %s ====\n' "$(date '+%F %T')" "$*"; }
fail() { banner "FAILED: $*"; exit 1; }

banner "Job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}: tp${TP}/ep${EP}, logs in ${LOG_DIR}"
[ -d "${MODEL_DIR}" ] || fail "model dir not found: ${MODEL_DIR}"
[ -f "${WHEEL}" ] || fail "wheel not found: ${WHEEL}"

IMAGE=$(awk -F= '$1=="LLM_SBSA_DOCKER_IMAGE"{print substr($0,index($0,"=")+1); exit}' \
    "${REPO}/jenkins/current_image_tags.properties")
[ -n "${IMAGE}" ] || fail "LLM_SBSA_DOCKER_IMAGE not found"
FALLBACK_IMAGE="urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-26.05-py3-sbsa-ubuntu24.04-skip-tritondevel-202607271403-16694"
case "${IMAGE}" in
    artifactory.nvidia.com/*)
        if ! grep -qs "machine artifactory.nvidia.com" "${HOME}/.config/enroot/.credentials"; then
            banner "No artifactory credentials; using anonymous fallback image"
            IMAGE="${FALLBACK_IMAGE}"
        fi
        ;;
esac
MOUNTS="${REPO}:${REPO},/lustre/share/coreai_dlalgo_ci:/lustre/share/coreai_dlalgo_ci"
banner "Container image: ${IMAGE}"

# ---------------------------------------------------------------------------
# Provision one named container per node (install once, reuse in every run).
# ---------------------------------------------------------------------------
banner "Provisioning containers (log: ${LOG_DIR}/provision.log)"
# The import sanity check is a single process with no PMIx server, so the
# Slurm/PMI environment must be scrubbed or the container's OpenMPI aborts in
# MPI_Init (the real 8-rank run below keeps the env and uses --mpi=pmix).
srun -N "${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
    --container-image="${IMAGE}" --container-name=trtllm \
    --container-mounts="${MOUNTS}" --no-container-remap-root \
    bash -c 'cd "${REPO}" && \
        python3 -m pip install -r requirements-dev.txt && \
        python3 -m pip install --force-reinstall --no-deps "${WHEEL}" && \
        for var in $(env | grep -E "^(SLURM_|PMIX_|PMI_)" | cut -d= -f1); do unset "$var"; done && \
        export OMPI_MCA_ras=^slurm OMPI_MCA_plm=^slurm OMPI_MCA_schizo=^slurm OMPI_MCA_ess=^slurm && \
        python3 -c "import tensorrt_llm; print(\"$(hostname):\", tensorrt_llm.__version__)"' \
    > "${LOG_DIR}/provision.log" 2>&1 \
    || { tail -n 30 "${LOG_DIR}/provision.log"; fail "container provisioning failed"; }
tail -n 2 "${LOG_DIR}/provision.log"

# ---------------------------------------------------------------------------
# A/B runs. Offload is controlled via env exported around each srun; the
# SLURM/PMIX environment is kept intact (multi-node MPI needs it).
# ---------------------------------------------------------------------------
clear_offload_env() {
    unset TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_FULL_LAYER \
          TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_GROUP_LAYERS \
          TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_INCREMENTAL \
          TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_CONSUME \
          TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_MAX_HOST_GIB
}

run_case() {
    local name="$1"
    banner "E2E run ${name} (log: ${LOG_DIR}/${name}.log)"
    srun -N "${SLURM_JOB_NUM_NODES}" --ntasks-per-node="${GPUS_PER_NODE}" --mpi=pmix \
        --container-name=trtllm --container-image="${IMAGE}" \
        --container-mounts="${MOUNTS}" --no-container-remap-root \
        bash -c 'export TLLM_LOG_LEVEL=INFO && cd "${REPO}" && \
            trtllm-llmapi-launch python3 kv_offload_e2e_ab.py \
                --model "${MODEL_DIR}" --tp '"${TP}"' --ep '"${EP}"' \
                --output "${LOG_DIR}/'"${name}"'.json"' \
        > "${LOG_DIR}/${name}.log" 2>&1
    local rc=$?
    grep -m1 "\[DSA KV offload prototype\]" "${LOG_DIR}/${name}.log" || true
    tail -n 3 "${LOG_DIR}/${name}.log"
    if [ ${rc} -ne 0 ]; then
        banner "run ${name} FAILED (rc=${rc}); last 40 lines:"
        tail -n 40 "${LOG_DIR}/${name}.log"
    fi
    return ${rc}
}

clear_offload_env
run_case baseline || fail "baseline run failed"
run_case baseline_repeat || fail "baseline repeat run failed"

export TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_FULL_LAYER=all
export TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_CONSUME=1
export TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_MAX_HOST_GIB=24
export TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_GROUP_LAYERS=2,3,4
run_case offload_s1; RC_S1=$?
export TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_GROUP_LAYERS=3,4
run_case offload_s2; RC_S2=$?
clear_offload_env

# Both offload runs must have actually consumed staged host rows.
ENGAGE_FAIL=0
for name in offload_s1 offload_s2; do
    if ! grep -q "attention consumes staged host rows" "${LOG_DIR}/${name}.log"; then
        banner "ENGAGEMENT CHECK FAILED: ${name} never entered consume mode"
        ENGAGE_FAIL=1
    fi
done

# ---------------------------------------------------------------------------
# Compare token ids across runs.
# ---------------------------------------------------------------------------
cat > "${LOG_DIR}/compare.py" <<'EOS'
import json
import sys

log_dir = sys.argv[1]
runs = {}
for name in ("baseline", "baseline_repeat", "offload_s1", "offload_s2"):
    try:
        with open(f"{log_dir}/{name}.json") as f:
            runs[name] = json.load(f)
    except FileNotFoundError:
        print(f"MISSING RESULT: {name}.json")
        runs[name] = None

fail = False
base = runs["baseline"]

def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            for j, (u, v) in enumerate(zip(x, y)):
                if u != v:
                    return f"prompt {i}, token {j}: {u} vs {v}"
            return f"prompt {i}, length {len(x)} vs {len(y)}"
    return "length mismatch"

if base is None:
    print("VERDICT: FAIL (no baseline)")
    sys.exit(1)

# Sanity: the sequential oracle itself must be reproducible.
rep = runs["baseline_repeat"]
if rep is not None and rep["sequential"] != base["sequential"]:
    print("BASELINE UNSTABLE (sequential): " + first_diff(base["sequential"], rep["sequential"]))
    print("The oracle is not reproducible; offload comparison is inconclusive.")
    fail = True

batched_stable = rep is not None and rep["batched"] == base["batched"]
print(f"baseline batched run-to-run stable: {batched_stable}")

for name in ("offload_s1", "offload_s2"):
    run = runs[name]
    if run is None:
        fail = True
        continue
    if run["sequential"] == base["sequential"]:
        print(f"{name}: sequential tokens IDENTICAL to baseline  [PASS]")
    else:
        print(f"{name}: sequential MISMATCH -> " + first_diff(base["sequential"], run["sequential"]))
        fail = True
    if batched_stable:
        if run["batched"] == base["batched"]:
            print(f"{name}: batched tokens IDENTICAL to baseline  [PASS]")
        else:
            print(f"{name}: batched MISMATCH -> " + first_diff(base["batched"], run["batched"]))
            fail = True
    else:
        same = run["batched"] == base["batched"]
        print(f"{name}: batched compared informationally (baseline unstable): identical={same}")

print("VERDICT:", "FAIL" if fail else "PASS")
sys.exit(1 if fail else 0)
EOS

banner "Comparing results"
srun -N 1 -n 1 --container-name=trtllm --container-image="${IMAGE}" \
    --container-mounts="${MOUNTS}" --no-container-remap-root \
    python3 "${LOG_DIR}/compare.py" "${LOG_DIR}"
CMP_RC=$?

FINAL_RC=0
[ ${RC_S1} -eq 0 ] && [ ${RC_S2} -eq 0 ] || FINAL_RC=1
[ ${ENGAGE_FAIL} -eq 0 ] || FINAL_RC=1
[ ${CMP_RC} -eq 0 ] || FINAL_RC=1
if [ ${FINAL_RC} -eq 0 ]; then
    banner "E2E RESULT: PASS (offload consume matches GPU-resident baseline)"
else
    banner "E2E RESULT: FAIL (s1_rc=${RC_S1} s2_rc=${RC_S2} engage_fail=${ENGAGE_FAIL} cmp_rc=${CMP_RC})"
fi
banner "Done. Logs: ${LOG_DIR}"
exit ${FINAL_RC}
