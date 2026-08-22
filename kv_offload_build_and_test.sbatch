#!/bin/bash
# Build the TensorRT-LLM wheel with the DSA KV-offload consume-mode changes and
# run the offload correctness tests on a single Blackwell (SM100+) node.
#
# Submit from the repo root on the login node:
#   sbatch kv_offload_build_and_test.sbatch
#
# Main log: slurm-<jobid>.out in the repo root.
# Phase logs: kv_offload_job_logs/<jobid>/{build.log,pip_install.log,
#             pytest_consume.log,pytest_kernels.log}
#
# A single GB200 (B200, SM100) node is sufficient for these tests; the wheel is
# built for 100-real;103-real so it also runs on GB300 later. Switch the
# partition below to gb300 if gb200 is unavailable.
#SBATCH -A coreai_comparch_trtllm
#SBATCH -p gb200
#SBATCH -N 1
#SBATCH -J coreai_comparch_trtllm-kv_offload.build_test
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fsw/coreai_comparch_trtllm/guijuz/TensorRT-LLM/slurm-%j.out

set -uo pipefail

export REPO=/lustre/fsw/coreai_comparch_trtllm/guijuz/TensorRT-LLM
# PEP-427 build-tag naming ("-2-" = second build of this version): keeps the
# wheel directly pip-installable while distinguishing iterations.
export WHEEL_OUT_NAME=tensorrt_llm-1.3.0rc25-2-cp312-cp312-linux_aarch64.whl
export LOG_DIR="${REPO}/kv_offload_job_logs/${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"

banner() { printf '\n==== [%s] %s ====\n' "$(date '+%F %T')" "$*"; }
fail() { banner "FAILED: $*"; exit 1; }

banner "Job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}; logs in ${LOG_DIR}"

# Image adopted by this worktree revision (SBSA: compute nodes are aarch64).
IMAGE=$(awk -F= '$1=="LLM_SBSA_DOCKER_IMAGE"{print substr($0,index($0,"=")+1); exit}' \
    "${REPO}/jenkins/current_image_tags.properties")
[ -n "${IMAGE}" ] || fail "LLM_SBSA_DOCKER_IMAGE not found in jenkins/current_image_tags.properties"
# artifactory.nvidia.com requires an access token in ~/.config/enroot/.credentials
# (submit_job.md, "Registry credentials"). Without one, fall back to the
# anonymous-pull mirror image every previous job of this branch used (verified
# HTTP 200 anonymously; 4 days older than the adopted tag, same 26.05 base).
FALLBACK_IMAGE="urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-26.05-py3-sbsa-ubuntu24.04-skip-tritondevel-202607271403-16694"
case "${IMAGE}" in
    artifactory.nvidia.com/*)
        if ! grep -qs "machine artifactory.nvidia.com" "${HOME}/.config/enroot/.credentials"; then
            banner "No enroot credentials for artifactory.nvidia.com; using anonymous fallback image"
            IMAGE="${FALLBACK_IMAGE}"
        fi
        ;;
esac
MOUNTS="${REPO}:${REPO}"
banner "Container image: ${IMAGE}"

ARCH=$(srun -N1 -n1 uname -m)
[ "${ARCH}" = "aarch64" ] || fail "expected an aarch64 compute node, got ${ARCH}"

# --------------------------------------------------------------------------
# Phase 1: DISABLED (wheel already built by job 2741464 and saved as
# build/tensorrt_llm-1.3.0rc25-v2-aarch64.whl). Flip `if false` to `if true`
# to re-enable the rebuild after further C++ changes.
# --------------------------------------------------------------------------
if false; then
cat > "${LOG_DIR}/container_build.sh" <<'EOS'
#!/bin/bash
set -euo pipefail
cd "${REPO}"
echo "== container: $(uname -m), $(python3 --version 2>&1) =="
git status --short | head -30
# A build venv created in a different container or checkout path has broken
# interpreter symlinks/shebangs (observed: conan exit 127). Recreate it if it
# cannot execute here.
if [ -d .venv-3.12 ] && ! { .venv-3.12/bin/python3 -c pass && .venv-3.12/bin/conan --version; } \
    > /dev/null 2>&1; then
    echo "== stale .venv-3.12 detected; removing so build_wheel recreates it =="
    rm -rf .venv-3.12
fi
export CCACHE_DIR="${REPO}/.ccache"
python3 scripts/build_wheel.py \
    --use_ccache \
    --cuda_architectures "100-real;103-real" \
    --job_count "$(nproc)"
EOS
chmod +x "${LOG_DIR}/container_build.sh"

STAMP="${LOG_DIR}/.build_started"
touch "${STAMP}"
banner "Phase 1: building wheel (full log: ${LOG_DIR}/build.log)"
srun -N1 -n1 --cpus-per-task="${SLURM_CPUS_ON_NODE}" \
    --container-image="${IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --no-container-remap-root \
    bash "${LOG_DIR}/container_build.sh" > "${LOG_DIR}/build.log" 2>&1
BUILD_RC=$?
tail -n 8 "${LOG_DIR}/build.log"
if [ ${BUILD_RC} -ne 0 ]; then
    banner "PHASE 1 RESULT: BUILD FAILED (rc=${BUILD_RC}); last 60 lines:"
    tail -n 60 "${LOG_DIR}/build.log"
    exit "${BUILD_RC}"
fi

NEW_WHEEL=$(find "${REPO}/build" -maxdepth 1 -type f \
    -name 'tensorrt_llm-*cp3*linux_aarch64.whl' -newer "${STAMP}" | head -n1)
[ -n "${NEW_WHEEL}" ] || fail "build succeeded but no freshly built wheel found under ${REPO}/build"
# The renamed copy is a published artifact only: pip rejects non-PEP-427 wheel
# filenames, so the test phase installs the original wheel.
cp -f "${NEW_WHEEL}" "${REPO}/build/${WHEEL_OUT_NAME}"
export TRTLLM_WHEEL_INSTALL="${NEW_WHEEL}"
banner "PHASE 1 RESULT: BUILD OK; ${NEW_WHEEL} -> build/${WHEEL_OUT_NAME}"
fi

# Use the saved wheel; the build-tag filename is PEP-427 valid, so pip can
# install it directly.
SAVED_WHEEL="${REPO}/build/${WHEEL_OUT_NAME}"
[ -f "${SAVED_WHEEL}" ] || fail "saved wheel not found: ${SAVED_WHEEL}"
export TRTLLM_WHEEL_INSTALL="${SAVED_WHEEL}"
banner "PHASE 1 SKIPPED: using saved wheel ${SAVED_WHEEL}"

# --------------------------------------------------------------------------
# Phase 2: install the wheel in a fresh container and run the offload tests.
# --------------------------------------------------------------------------
cat > "${LOG_DIR}/container_test.sh" <<'EOS'
#!/bin/bash
set -uo pipefail
cd "${REPO}"
echo "== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1) =="
echo "== pip install (log: ${LOG_DIR}/pip_install.log) =="
python3 -m pip install -r requirements-dev.txt > "${LOG_DIR}/pip_install.log" 2>&1 \
    || { echo "requirements-dev install FAILED"; tail -n 30 "${LOG_DIR}/pip_install.log"; exit 91; }
python3 -m pip install --force-reinstall --no-deps "${TRTLLM_WHEEL_INSTALL}" \
    >> "${LOG_DIR}/pip_install.log" 2>&1 \
    || { echo "wheel install FAILED"; tail -n 30 "${LOG_DIR}/pip_install.log"; exit 92; }

for var in $(env | grep -E "^(SLURM_|PMIX_|PMI_)" | cut -d= -f1); do unset "$var"; done
export OMPI_MCA_ras=^slurm OMPI_MCA_plm=^slurm OMPI_MCA_schizo=^slurm OMPI_MCA_ess=^slurm
export PYTHONPATH="${REPO}:${REPO}/tests:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

# Prove pytest exercises this worktree and the rebuilt binary carries the new op.
python3 - <<'PY'
import os
from pathlib import Path
import torch
import tensorrt_llm
repo = Path(os.environ["REPO"]).resolve()
loaded = Path(tensorrt_llm.__file__).resolve()
print(f"TensorRT-LLM {tensorrt_llm.__version__} loaded from: {loaded}")
assert str(loaded).startswith(str(repo)), (loaded, repo)
torch.ops.trtllm.dsa_kv_cache_offload_patch  # raises if the rebuilt op is missing
print("dsa_kv_cache_offload_patch op registered: OK")
PY
[ $? -eq 0 ] || { echo "IMPORT/OP CHECK FAILED (stale wheel or wrong PYTHONPATH?)"; exit 93; }

echo
echo "== pytest 1/2: IndexShare consume-mode bit-exactness =="
python3 -m pytest tests/unittest/_torch/attention/sparse/dsa/test_dsa_sparse_mla.py \
    -k kv_offload_consume -v 2>&1 | tee "${LOG_DIR}/pytest_consume.log"
RC1=${PIPESTATUS[0]}

echo
echo "== pytest 2/2: offload transfer-kernel unit tests =="
python3 -m pytest tests/unittest/_torch/attention/sparse/test_dsa_kv_cache_offload.py \
    -v 2>&1 | tee "${LOG_DIR}/pytest_kernels.log"
RC2=${PIPESTATUS[0]}

echo
echo "== TEST SUMMARY =="
[ "${RC1}" -eq 0 ] && echo "kv_offload_consume  : PASS" || echo "kv_offload_consume  : FAIL (rc=${RC1})"
[ "${RC2}" -eq 0 ] && echo "offload kernel tests: PASS" || echo "offload kernel tests: FAIL (rc=${RC2})"
[ "${RC1}" -eq 0 ] && [ "${RC2}" -eq 0 ]
EOS
chmod +x "${LOG_DIR}/container_test.sh"

banner "Phase 2: installing wheel and running offload tests"
srun -N1 -n1 --cpus-per-task="${SLURM_CPUS_ON_NODE}" \
    --container-image="${IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --no-container-remap-root \
    bash "${LOG_DIR}/container_test.sh"
TEST_RC=$?

if [ ${TEST_RC} -eq 0 ]; then
    banner "PHASE 2 RESULT: ALL TESTS PASSED"
else
    banner "PHASE 2 RESULT: TESTS FAILED (rc=${TEST_RC}); see ${LOG_DIR}/pytest_*.log"
fi
banner "Done. Wheel: build/${WHEEL_OUT_NAME}; logs: ${LOG_DIR}"
exit "${TEST_RC}"
