#!/bin/bash
# DeepSeek V3.1-NVFP4 Benchmark Script
# Runs benchmarks with MTP/EAGLE3/PARD and +SA configurations
#
# Test Plan:
# 1. EAGLE3 speculative decoding (optional, ENABLE_EAGLE3=true)
# 2. EAGLE3 + SA (optional, ENABLE_EAGLE3=true)
# 3. EAGLE3 + SA + Global Pool (optional, ENABLE_EAGLE3=true)
# 4. PARD speculative decoding (optional, ENABLE_PARD=true)
# 5. PARD + SA (optional, ENABLE_PARD=true)
# 6. PARD + SA + Global Pool (optional, ENABLE_PARD=true)
# 7. Baseline reference
# 8. MTP, MTP + SA, MTP + SA + Global Pool (optional, ENABLE_MTP=true)
#
# Dataset types:
#   code_edits    - glaiveai/code_edits_sample (good for SA acceptance rate)
#   repobench     - RepoBench-P v1.1 cross_file_first (good for global pool,
#                   requests grouped by repo share large context snippets)
#   crosscodeeval - CrossCodeEval (cross-file code completion, ideal for SA + global pool,
#                   prompt includes cross-file context, output reuses identifiers from context)
#   swebench      - SWE-bench Verified (real GitHub issues + code context,
#                   grouped by repo, issue + hints + test patch as context)

set -e

# ============================================================
# Configuration
# ============================================================
# MTP (optional) target model
MODEL_PATH="/home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V3-0324-FP4"
MODEL_NAME="deepseek-ai/DeepSeek-V3-0324-FP4"

# EAGLE3 target model (default: single-GPU friendly)
EAGLE3_TARGET_PATH="${EAGLE3_TARGET_PATH:-/home/scratch.trt_llm_data_ci/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct}"
EAGLE3_TARGET_NAME="${EAGLE3_TARGET_NAME:-meta-llama/Llama-3.1-8B-Instruct}"

# PARD target model (default: match TRT-LLM integration tests)
PARD_TARGET_PATH="${PARD_TARGET_PATH:-/home/scratch.trt_llm_data_ci/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct}"
PARD_TARGET_NAME="${PARD_TARGET_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
EAGLE3_MODEL="${EAGLE3_MODEL:-yuhuili/EAGLE3-LLaMA3.1-Instruct-8B}"
PARD_MODEL="${PARD_MODEL:-/home/scratch.trt_llm_data_ci/llm-models/PARD-Llama-3.2-1B}"

# Dataset selection: "code_edits" or "repobench" or "crosscodeeval" or "swebench"R
DATASET_TYPE=${DATASET_TYPE:-code_edits}

# code_edits settings
SOURCE_JSON="/home/scratch.guijuz_coreai/code_edits_sample/edits_data.json"

# repobench settings (pre-downloaded RepoBench-P v1.1 cross_file_first)
REPOBENCH_JSON="/home/scratch.guijuz_coreai/repobench_python_v1.1/repobench_data.json"
# Max repos to sample from (requests grouped by repo for cross-request overlap)
REPOBENCH_MAX_REPOS=${REPOBENCH_MAX_REPOS:-10}
# Max requests per repo (0 = all)
REPOBENCH_MAX_PER_REPO=${REPOBENCH_MAX_PER_REPO:-0}
# Output tokens for RepoBench (128=next-line, 512=longer generation for more decode time)
REPOBENCH_OUTPUT_TOKENS=${REPOBENCH_OUTPUT_TOKENS:-128}

# crosscodeeval settings (downloaded on first run from HuggingFace)
# Language: python, java, typescript, csharp
CCEVAL_LANGUAGE=${CCEVAL_LANGUAGE:-python}
# Max repos to sample from (requests grouped by repo for cross-request overlap)
CCEVAL_MAX_REPOS=${CCEVAL_MAX_REPOS:-10}
# Max requests per repo (0 = all)
CCEVAL_MAX_PER_REPO=${CCEVAL_MAX_PER_REPO:-0}
# Output tokens (default 128 for line completion)
CCEVAL_OUTPUT_TOKENS=${CCEVAL_OUTPUT_TOKENS:-128}

# swebench settings (downloaded on first run from HuggingFace)
# Max repos to sample from (requests grouped by repo for cross-request overlap)
SWEBENCH_MAX_REPOS=${SWEBENCH_MAX_REPOS:-10}
# Max requests per repo (0 = all)
SWEBENCH_MAX_PER_REPO=${SWEBENCH_MAX_PER_REPO:-0}
# Output tokens (default 512 for patch generation)
SWEBENCH_OUTPUT_TOKENS=${SWEBENCH_OUTPUT_TOKENS:-512}

SAMPLE_SIZE=100
RANDOM_SEED=${RANDOM_SEED:-42}  # Use env var or default to 42 for reproducibility
TP_SIZE=${TP_SIZE:-1}
ENABLE_MTP=${ENABLE_MTP:-false}
ENABLE_EAGLE3=${ENABLE_EAGLE3:-false}
ENABLE_PARD=${ENABLE_PARD:-false}
NUM_REQUESTS=100
WARMUP=2
CONCURRENCY=${CONCURRENCY:-1}
MTP_BATCH_SIZE=${MTP_BATCH_SIZE:-4}
MTP_CONCURRENCY=${MTP_CONCURRENCY:-$MTP_BATCH_SIZE}
OUTPUT_DIR="./sa_comparison_results_$(date +%Y%m%d_%H%M%S)"
DATASET="$OUTPUT_DIR/sampled_data.jsonl"  # Will be created by sampling


EAGLE3_BATCH_SIZE=${EAGLE3_BATCH_SIZE:-4}
EAGLE3_CONCURRENCY=${EAGLE3_CONCURRENCY:-$EAGLE3_BATCH_SIZE}
# Which EAGLE3 variants to run: comma-separated list of "base", "sa", "sa_global" (default: all)
EAGLE3_VARIANTS=${EAGLE3_VARIANTS:-base,sa,sa_global}

PARD_BATCH_SIZE=${PARD_BATCH_SIZE:-4}
PARD_CONCURRENCY=${PARD_CONCURRENCY:-$PARD_BATCH_SIZE}
PARD_DRAFT_LEN=${PARD_DRAFT_LEN:-4}

# ============================================================
# Setup
# ============================================================
echo "============================================================"
echo "DeepSeek V3-0324-FP4 Benchmark Script"
echo "EAGLE3/PARD vs +SA Comparison (single-GPU default)"
echo "============================================================"
echo ""

# Check dataset source exists
if [ "$DATASET_TYPE" = "repobench" ]; then
    if [ ! -f "$REPOBENCH_JSON" ]; then
        echo "ERROR: RepoBench JSON not found: $REPOBENCH_JSON"
        echo "Please download RepoBench-P v1.1 first (see script header)."
        exit 1
    fi
elif [ "$DATASET_TYPE" = "crosscodeeval" ]; then
    echo "CrossCodeEval dataset will be downloaded from HuggingFace (language=$CCEVAL_LANGUAGE)"
elif [ "$DATASET_TYPE" = "swebench" ]; then
    echo "SWE-bench Verified will be downloaded from HuggingFace"
elif [ "$DATASET_TYPE" = "code_edits" ]; then
    if [ ! -f "$SOURCE_JSON" ]; then
        echo "ERROR: Source JSON not found: $SOURCE_JSON"
        echo "Please ensure the source data file exists."
        exit 1
    fi
else
    echo "ERROR: Unknown DATASET_TYPE: $DATASET_TYPE"
    echo "Supported values: code_edits, repobench, crosscodeeval, swebench"
    exit 1
fi

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path not found: $MODEL_PATH"
    exit 1
fi

# Check EAGLE3/PARD target model paths
if [ "$ENABLE_EAGLE3" = "true" ]; then
    if [ ! -d "$EAGLE3_TARGET_PATH" ]; then
        echo "ERROR: EAGLE3 target model path not found: $EAGLE3_TARGET_PATH"
        exit 1
    fi
fi

if [ "$ENABLE_PARD" = "true" ]; then
    if [ ! -d "$PARD_TARGET_PATH" ]; then
        echo "ERROR: PARD target model path not found: $PARD_TARGET_PATH"
        exit 1
    fi

    # Basic sanity check: PARD target and draft should not be identical
    if [ "$PARD_TARGET_PATH" = "$PARD_MODEL" ] || [ "$PARD_TARGET_NAME" = "$PARD_MODEL" ]; then
        echo "ERROR: PARD target and draft model are identical. Please set different values for:"
        echo "       - PARD_TARGET_PATH/PARD_TARGET_NAME (target)"
        echo "       - PARD_MODEL (draft)"
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Dataset type: $DATASET_TYPE"
echo "MTP target model path: $MODEL_PATH"
echo "EAGLE3 target model path: $EAGLE3_TARGET_PATH"
echo "PARD target model path: $PARD_TARGET_PATH"
if [ "$DATASET_TYPE" = "repobench" ]; then
    echo "RepoBench JSON: $REPOBENCH_JSON"
    echo "Max repos: $REPOBENCH_MAX_REPOS"
    echo "Max per repo: $REPOBENCH_MAX_PER_REPO (0=all)"
elif [ "$DATASET_TYPE" = "crosscodeeval" ]; then
    echo "CrossCodeEval language: $CCEVAL_LANGUAGE"
    echo "Max repos: $CCEVAL_MAX_REPOS"
    echo "Max per repo: $CCEVAL_MAX_PER_REPO (0=all)"
elif [ "$DATASET_TYPE" = "swebench" ]; then
    echo "SWE-bench Verified (HuggingFace)"
    echo "Max repos: $SWEBENCH_MAX_REPOS"
    echo "Max per repo: $SWEBENCH_MAX_PER_REPO (0=all)"
else
    echo "Source JSON: $SOURCE_JSON"
fi
echo "EAGLE3 model: $EAGLE3_MODEL"
echo "PARD model: $PARD_MODEL"
echo "Sample size: $SAMPLE_SIZE"
echo "Random seed: $RANDOM_SEED"
echo "TP Size: $TP_SIZE"
echo "MTP enabled: $ENABLE_MTP"
echo "EAGLE3 enabled: $ENABLE_EAGLE3"
echo "PARD enabled: $ENABLE_PARD"
echo ""

# ============================================================
# Sample Data from Source JSON
# ============================================================

if [ "$DATASET_TYPE" = "repobench" ]; then
    echo "Preparing RepoBench-P dataset (max_repos=$REPOBENCH_MAX_REPOS, max_per_repo=$REPOBENCH_MAX_PER_REPO)..."

    python3 << SAMPLE_SCRIPT
import json
import random
from collections import defaultdict

source_json = "$REPOBENCH_JSON"
output_jsonl = "$DATASET"
sample_size = $SAMPLE_SIZE
random_seed = $RANDOM_SEED
max_repos = $REPOBENCH_MAX_REPOS
max_per_repo = $REPOBENCH_MAX_PER_REPO

random.seed(random_seed)

with open(source_json, 'r') as f:
    data = json.load(f)

print(f"  Total entries in RepoBench source: {len(data)}")

# Group by repo (entries are already ordered by repo)
repo_rows = defaultdict(list)
for entry in data:
    repo_rows[entry['repo_name']].append(entry)

# Sort repos by number of requests (descending) for maximum overlap
repos_sorted = sorted(repo_rows.keys(), key=lambda r: -len(repo_rows[r]))

# Select top N repos
selected_repos = repos_sorted[:max_repos]
print(f"  Selected {len(selected_repos)} repos (from {len(repos_sorted)} with >=5 requests)")

# Collect requests, optionally capping per-repo
selected = []
for repo in selected_repos:
    rows = repo_rows[repo]
    if max_per_repo > 0 and len(rows) > max_per_repo:
        rows = random.sample(rows, max_per_repo)
    selected.extend(rows)

print(f"  Total selected requests: {len(selected)}")

# No sample_size cap for repobench: use all selected requests
# so that all repos are represented in the dataset

# Repo-grouped ordering: requests from the same repo stay adjacent
# so they are likely in-flight together, maximizing global pool hits
print(f"  Final dataset size: {len(selected)}")
for repo in selected_repos:
    count = sum(1 for e in selected if e['repo_name'] == repo)
    if count > 0:
        print(f"    {repo}: {count} requests")

output_tokens = $REPOBENCH_OUTPUT_TOKENS
with open(output_jsonl, 'w') as f:
    for task_id, entry in enumerate(selected):
        record = {
            "task_id": task_id,
            "prompt": entry['prompt'],
            "output_tokens": output_tokens
        }
        f.write(json.dumps(record) + '\n')

print(f"  Saved RepoBench dataset to: {output_jsonl}")
SAMPLE_SCRIPT

elif [ "$DATASET_TYPE" = "crosscodeeval" ]; then
    echo "Preparing CrossCodeEval dataset (language=$CCEVAL_LANGUAGE, max_repos=$CCEVAL_MAX_REPOS, max_per_repo=$CCEVAL_MAX_PER_REPO)..."

    DATASET_OUT="$DATASET" \
    RANDOM_SEED_VAL="$RANDOM_SEED" \
    CCEVAL_LANGUAGE_VAL="$CCEVAL_LANGUAGE" \
    CCEVAL_MAX_REPOS_VAL="$CCEVAL_MAX_REPOS" \
    CCEVAL_MAX_PER_REPO_VAL="$CCEVAL_MAX_PER_REPO" \
    CCEVAL_OUTPUT_TOKENS_VAL="$CCEVAL_OUTPUT_TOKENS" \
    python3 << 'SAMPLE_SCRIPT'
import json
import os
import random
from collections import defaultdict

output_jsonl = os.environ["DATASET_OUT"]
random_seed = int(os.environ["RANDOM_SEED_VAL"])
language = os.environ["CCEVAL_LANGUAGE_VAL"]
max_repos = int(os.environ["CCEVAL_MAX_REPOS_VAL"])
max_per_repo = int(os.environ["CCEVAL_MAX_PER_REPO_VAL"])
output_tokens = int(os.environ["CCEVAL_OUTPUT_TOKENS_VAL"])

random.seed(random_seed)

from huggingface_hub import hf_hub_download

# Use the oracle_bm25 variant which includes retrieved cross-file context
filename = f"crosscodeeval_data/{language}/line_completion_oracle_bm25.jsonl"
print(f"  Downloading CrossCodeEval: {filename}")

local_path = hf_hub_download(
    repo_id="Vincentvmt/CrossCodeEval",
    filename=filename,
    repo_type="dataset",
)
print(f"  Downloaded to: {local_path}")

data = []
with open(local_path, 'r') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

print(f"  Total entries: {len(data)}")

# Group by repository for repo-grouped ordering (maximizes global pool hits)
repo_rows = defaultdict(list)
for entry in data:
    repo = entry['metadata']['repository']
    repo_rows[repo].append(entry)

repos_sorted = sorted(repo_rows.keys(), key=lambda r: -len(repo_rows[r]))
selected_repos = repos_sorted[:max_repos]
print(f"  Selected {len(selected_repos)} repos (from {len(repos_sorted)} total)")

selected = []
for repo in selected_repos:
    rows = repo_rows[repo]
    if max_per_repo > 0 and len(rows) > max_per_repo:
        rows = random.sample(rows, max_per_repo)
    selected.extend(rows)

print(f"  Total selected requests: {len(selected)}")
print(f"  Final dataset size: {len(selected)}")
for repo in selected_repos:
    count = sum(1 for e in selected if e['metadata']['repository'] == repo)
    if count > 0:
        print(f"    {repo}: {count} requests")

with open(output_jsonl, 'w') as f:
    for task_id, entry in enumerate(selected):
        # Prepend cross-file context to the prompt so SA can match
        # cross-file identifiers (imports, class names, API calls)
        crossfile_ctx = ""
        if 'crossfile_context' in entry and entry['crossfile_context']:
            ctx = entry['crossfile_context']
            if isinstance(ctx, dict) and 'text' in ctx:
                crossfile_ctx = ctx['text']
            elif isinstance(ctx, str):
                crossfile_ctx = ctx

        in_file_prompt = entry['prompt']

        if crossfile_ctx:
            full_prompt = (
                f"# Cross-file context:\n{crossfile_ctx}\n\n"
                f"# Current file:\n{in_file_prompt}"
            )
        else:
            full_prompt = in_file_prompt

        record = {
            "task_id": task_id,
            "prompt": full_prompt,
            "output_tokens": output_tokens
        }
        f.write(json.dumps(record) + '\n')

print(f"  Saved CrossCodeEval dataset to: {output_jsonl}")
SAMPLE_SCRIPT

elif [ "$DATASET_TYPE" = "swebench" ]; then
    echo "Preparing SWE-bench Verified dataset (max_repos=$SWEBENCH_MAX_REPOS, max_per_repo=$SWEBENCH_MAX_PER_REPO)..."

    DATASET_OUT="$DATASET" \
    RANDOM_SEED_VAL="$RANDOM_SEED" \
    SWEBENCH_MAX_REPOS_VAL="$SWEBENCH_MAX_REPOS" \
    SWEBENCH_MAX_PER_REPO_VAL="$SWEBENCH_MAX_PER_REPO" \
    SWEBENCH_OUTPUT_TOKENS_VAL="$SWEBENCH_OUTPUT_TOKENS" \
    python3 << 'SAMPLE_SCRIPT'
import json
import os
import random
from collections import defaultdict

output_jsonl = os.environ["DATASET_OUT"]
random_seed = int(os.environ["RANDOM_SEED_VAL"])
max_repos = int(os.environ["SWEBENCH_MAX_REPOS_VAL"])
max_per_repo = int(os.environ["SWEBENCH_MAX_PER_REPO_VAL"])
output_tokens = int(os.environ["SWEBENCH_OUTPUT_TOKENS_VAL"])

random.seed(random_seed)

from datasets import load_dataset

print("  Downloading SWE-bench Verified from HuggingFace...")
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
data = list(ds)
print(f"  Total entries: {len(data)}")

# Group by repo for repo-grouped ordering (maximizes global pool hits)
repo_rows = defaultdict(list)
for entry in data:
    repo_rows[entry['repo']].append(entry)

repos_sorted = sorted(repo_rows.keys(), key=lambda r: -len(repo_rows[r]))
selected_repos = repos_sorted[:max_repos]
print(f"  Selected {len(selected_repos)} repos (from {len(repos_sorted)} total)")

selected = []
for repo in selected_repos:
    rows = repo_rows[repo]
    if max_per_repo > 0 and len(rows) > max_per_repo:
        rows = random.sample(rows, max_per_repo)
    selected.extend(rows)

print(f"  Total selected requests: {len(selected)}")
print(f"  Final dataset size: {len(selected)}")
for repo in selected_repos:
    count = sum(1 for e in selected if e['repo'] == repo)
    if count > 0:
        print(f"    {repo}: {count} requests")

with open(output_jsonl, 'w') as f:
    for task_id, entry in enumerate(selected):
        # Build a rich prompt from all available context fields:
        #   - problem_statement: the GitHub issue (title + body, often has code/tracebacks)
        #   - hints_text: discussion comments (often has code snippets, proposed fixes)
        #   - test_patch: the test code added by the fix PR (contains identifiers from the fix)
        # This maximizes SA material: the patch output will reuse identifiers from these contexts.
        parts = []

        if entry.get('test_patch'):
            parts.append(f"# Related test changes:\n```\n{entry['test_patch']}\n```")

        if entry.get('hints_text'):
            parts.append(f"# Discussion:\n{entry['hints_text']}")

        parts.append(f"# Issue:\n{entry['problem_statement']}")
        parts.append("# Please generate the patch to fix this issue.\n")

        full_prompt = "\n\n".join(parts)

        record = {
            "task_id": task_id,
            "prompt": full_prompt,
            "output_tokens": output_tokens
        }
        f.write(json.dumps(record) + '\n')

print(f"  Saved SWE-bench dataset to: {output_jsonl}")
SAMPLE_SCRIPT

else
    echo "Sampling $SAMPLE_SIZE random entries from code_edits source data..."

    python3 << SAMPLE_SCRIPT
import json
import random

source_json = "$SOURCE_JSON"
output_jsonl = "$DATASET"
sample_size = $SAMPLE_SIZE
random_seed = $RANDOM_SEED

random.seed(random_seed)

with open(source_json, 'r') as f:
    data = json.load(f)

total_entries = len(data)
print(f"  Total entries in source: {total_entries}")

if sample_size > total_entries:
    print(f"  WARNING: Sample size ({sample_size}) > total entries ({total_entries})")
    print(f"  Using all {total_entries} entries instead")
    sample_size = total_entries

sampled_indices = random.sample(range(total_entries), sample_size)
sampled_data = [data[i] for i in sampled_indices]

print(f"  Sampled {len(sampled_data)} entries (seed={random_seed})")

with open(output_jsonl, 'w') as f:
    for task_id, entry in enumerate(sampled_data):
        prompt = f'''You are a code editing assistant. Given the following code and edit instruction, provide the modified code.

### Original Code:
\`\`\`
{entry['code']}
\`\`\`

### Edit Instruction:

{entry['edit']}

### Modified Code:
'''
        record = {
            "task_id": task_id,
            "prompt": prompt,
            "output_tokens": 256
        }
        f.write(json.dumps(record) + '\n')

print(f"  Saved sampled dataset to: {output_jsonl}")
SAMPLE_SCRIPT
fi

if [ ! -f "$DATASET" ]; then
    echo "ERROR: Failed to create sampled dataset"
    exit 1
fi

# Derive NUM_REQUESTS from the actual dataset size
NUM_REQUESTS=$(wc -l < "$DATASET")
echo "Dataset: $DATASET ($NUM_REQUESTS requests)"
echo ""

# ============================================================
# Create Config Files
# ============================================================
echo "Creating configuration files..."

if [ "$ENABLE_MTP" = "true" ]; then
    # MTP config with cudagraph and overlap scheduler enabled
    # NOTE: enable_attention_dp must be false for speculative decoding to work properly
    cat > "$OUTPUT_DIR/config_mtp.yml" << EOF
max_batch_size: $MTP_BATCH_SIZE
speculative_config:
  decoding_type: "MTP"
  num_nextn_predict_layers: 8
EOF
    echo "  - Created: config_mtp.yml (MTP, max_batch_size=$MTP_BATCH_SIZE)"

    # MTP + SA config with sa_spec_threshold = 4
    # NOTE: enable_attention_dp must be false for speculative decoding to work properly
    cat > "$OUTPUT_DIR/config_mtp_sa.yml" << EOF
max_batch_size: $MTP_BATCH_SIZE
speculative_config:
  decoding_type: "MTP"
  num_nextn_predict_layers: 8
  use_sa_spec: true
  sa_spec_threshold: 4
EOF
    echo "  - Created: config_mtp_sa.yml (MTP + SA, max_batch_size=$MTP_BATCH_SIZE)"

    # MTP + SA + Global Pool config
    cat > "$OUTPUT_DIR/config_mtp_sa_global.yml" << EOF
max_batch_size: $MTP_BATCH_SIZE
speculative_config:
  decoding_type: "MTP"
  num_nextn_predict_layers: 8
  use_sa_spec: true
  sa_spec_threshold: 4
  enable_global_pool: true
EOF
    echo "  - Created: config_mtp_sa_global.yml (MTP + SA + Global Pool, max_batch_size=$MTP_BATCH_SIZE)"
else
    echo "  - Skipped MTP configs (ENABLE_MTP=false)"
fi

if [ "$ENABLE_EAGLE3" = "true" ]; then
    # EAGLE3 config
    cat > "$OUTPUT_DIR/config_eagle3.yml" << EOF
max_batch_size: $EAGLE3_BATCH_SIZE
enable_chunked_prefill: true
speculative_config:
  decoding_type: Eagle3
  max_draft_len: 8
  speculative_model: $EAGLE3_MODEL
EOF
    echo "  - Created: config_eagle3.yml (EAGLE3, max_batch_size=$EAGLE3_BATCH_SIZE, chunked_prefill=true)"

    # EAGLE3 + SA config
    cat > "$OUTPUT_DIR/config_eagle3_sa.yml" << EOF
max_batch_size: $EAGLE3_BATCH_SIZE
enable_chunked_prefill: true
speculative_config:
  decoding_type: Eagle3
  max_draft_len: 8
  speculative_model: $EAGLE3_MODEL
  use_sa_spec: true
  sa_spec_threshold: 4
EOF
    echo "  - Created: config_eagle3_sa.yml (EAGLE3 + SA, max_batch_size=$EAGLE3_BATCH_SIZE, chunked_prefill=true)"

    # EAGLE3 + SA + Global Pool config
    cat > "$OUTPUT_DIR/config_eagle3_sa_global.yml" << EOF
max_batch_size: $EAGLE3_BATCH_SIZE
enable_chunked_prefill: true
speculative_config:
  decoding_type: Eagle3
  max_draft_len: 8
  speculative_model: $EAGLE3_MODEL
  use_sa_spec: true
  sa_spec_threshold: 4
  enable_global_pool: true
EOF
    echo "  - Created: config_eagle3_sa_global.yml (EAGLE3 + SA + Global Pool, max_batch_size=$EAGLE3_BATCH_SIZE, chunked_prefill=true)"
else
    echo "  - Skipped EAGLE3 configs (ENABLE_EAGLE3=false)"
fi



if [ "$ENABLE_PARD" = "true" ]; then
    # PARD config
    cat > "$OUTPUT_DIR/config_pard.yml" << EOF
max_batch_size: $PARD_BATCH_SIZE
speculative_config:
  decoding_type: PARD
  max_draft_len: $PARD_DRAFT_LEN
  speculative_model: $PARD_MODEL
EOF
    echo "  - Created: config_pard.yml (PARD, max_batch_size=$PARD_BATCH_SIZE, max_draft_len=$PARD_DRAFT_LEN)"

    # PARD + SA config
    cat > "$OUTPUT_DIR/config_pard_sa.yml" << EOF
max_batch_size: $PARD_BATCH_SIZE
speculative_config:
  decoding_type: PARD
  max_draft_len: $PARD_DRAFT_LEN
  speculative_model: $PARD_MODEL
  use_sa_spec: true
  sa_spec_threshold: 4
EOF
    echo "  - Created: config_pard_sa.yml (PARD + SA, max_batch_size=$PARD_BATCH_SIZE, max_draft_len=$PARD_DRAFT_LEN)"

    # PARD + SA + Global Pool config
    cat > "$OUTPUT_DIR/config_pard_sa_global.yml" << EOF
max_batch_size: $PARD_BATCH_SIZE
speculative_config:
  decoding_type: PARD
  max_draft_len: $PARD_DRAFT_LEN
  speculative_model: $PARD_MODEL
  use_sa_spec: true
  sa_spec_threshold: 4
  enable_global_pool: true
EOF
    echo "  - Created: config_pard_sa_global.yml (PARD + SA + Global Pool, max_batch_size=$PARD_BATCH_SIZE, max_draft_len=$PARD_DRAFT_LEN)"
else
    echo "  - Skipped PARD configs (ENABLE_PARD=false)"
fi

# Baseline configs (no speculative decoding, but match batch/prefill settings)
if [ "$ENABLE_EAGLE3" = "true" ]; then
    cat > "$OUTPUT_DIR/config_baseline.yml" << EOF
max_batch_size: $EAGLE3_BATCH_SIZE
enable_chunked_prefill: true
EOF
    echo "  - Created: config_baseline.yml (baseline, max_batch_size=$EAGLE3_BATCH_SIZE, chunked_prefill=true)"
elif [ "$ENABLE_PARD" = "true" ]; then
    cat > "$OUTPUT_DIR/config_baseline.yml" << EOF
max_batch_size: $PARD_BATCH_SIZE
enable_chunked_prefill: true
EOF
    echo "  - Created: config_baseline.yml (baseline, max_batch_size=$PARD_BATCH_SIZE, chunked_prefill=true)"
elif [ "$ENABLE_MTP" = "true" ]; then
    cat > "$OUTPUT_DIR/config_baseline.yml" << EOF
max_batch_size: $MTP_BATCH_SIZE
EOF
    echo "  - Created: config_baseline.yml (baseline, max_batch_size=$MTP_BATCH_SIZE)"
else
    cat > "$OUTPUT_DIR/config_baseline.yml" << 'EOF'
{}
EOF
    echo "  - Created: config_baseline.yml (baseline, defaults)"
fi

echo ""

# ============================================================
# Initialize Summary File
# ============================================================
SUMMARY_FILE="$OUTPUT_DIR/benchmark_summary.log"
DATASET_DETAILS_CAPTURED=false

DATASET_SOURCE_LABEL="$SOURCE_JSON"
if [ "$DATASET_TYPE" = "repobench" ]; then
    DATASET_SOURCE_LABEL="$REPOBENCH_JSON (max_repos=$REPOBENCH_MAX_REPOS, max_per_repo=$REPOBENCH_MAX_PER_REPO)"
elif [ "$DATASET_TYPE" = "crosscodeeval" ]; then
    DATASET_SOURCE_LABEL="Vincentvmt/CrossCodeEval (language=$CCEVAL_LANGUAGE, max_repos=$CCEVAL_MAX_REPOS, max_per_repo=$CCEVAL_MAX_PER_REPO)"
elif [ "$DATASET_TYPE" = "swebench" ]; then
    DATASET_SOURCE_LABEL="princeton-nlp/SWE-bench_Verified (max_repos=$SWEBENCH_MAX_REPOS, max_per_repo=$SWEBENCH_MAX_PER_REPO)"
fi

cat > "$SUMMARY_FILE" << EOF
============================================================
DeepSeek V3.1-NVFP4 Benchmark Summary
============================================================
Date: $(date)
Dataset Type: $DATASET_TYPE
MTP Target Model: $MODEL_PATH
EAGLE3 Target Model: $EAGLE3_TARGET_PATH
PARD Target Model: $PARD_TARGET_PATH
Source Data: $DATASET_SOURCE_LABEL
Sampled Dataset: $DATASET
Dataset Size: $NUM_REQUESTS (random seed: $RANDOM_SEED)
Num Requests: $NUM_REQUESTS
Warmup: $WARMUP
Concurrency: $CONCURRENCY
TP Size: $TP_SIZE
============================================================

Test Plan:
1. EAGLE3 speculative decoding (optional, ENABLE_EAGLE3=$ENABLE_EAGLE3)
2. EAGLE3 + SA (optional, ENABLE_EAGLE3=$ENABLE_EAGLE3)
3. EAGLE3 + SA + Global Pool (optional, ENABLE_EAGLE3=$ENABLE_EAGLE3)
4. PARD speculative decoding (optional, ENABLE_PARD=$ENABLE_PARD)
5. PARD + SA (optional, ENABLE_PARD=$ENABLE_PARD)
6. PARD + SA + Global Pool (optional, ENABLE_PARD=$ENABLE_PARD)
7. Baseline reference (no speculative decoding)
8. MTP, MTP + SA, MTP + SA + Global Pool (optional, ENABLE_MTP=$ENABLE_MTP)

============================================================

EOF

# ============================================================
# Benchmark Function
# ============================================================
run_benchmark() {
    local config_name="$1"
    local config_file="$2"
    local benchmark_model_name="${3:-$MODEL_NAME}"
    local benchmark_model_path="${4:-$MODEL_PATH}"
    local log_file="$OUTPUT_DIR/bench_${config_name}.log"
    local json_file="$OUTPUT_DIR/bench_${config_name}.json"

    echo "============================================================"
    echo "Running benchmark: $config_name"
    echo "Config file: $config_file"
    echo "Model name: $benchmark_model_name"
    echo "Model path: $benchmark_model_path"
    echo "Log file: $log_file"
    echo "JSON report: $json_file"
    echo "============================================================"

    # Record start time
    local start_time=$(date +%s)

    # Run benchmark
    trtllm-bench --model "$benchmark_model_name" \
        --model_path "$benchmark_model_path" \
        latency \
        --dataset "$DATASET" \
        --tp "$TP_SIZE" \
        --config "$config_file" \
        --num_requests "$NUM_REQUESTS" \
        --warmup "$WARMUP" \
        --concurrency "$CONCURRENCY" \
        --report_json "$json_file" \
        --backend pytorch 2>&1 | while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Capture DATASET DETAILS once (from the first successful benchmark)
    if [ $exit_code -eq 0 ] && [ "$DATASET_DETAILS_CAPTURED" = false ]; then
        echo "------------------------------------------------------------" >> "$SUMMARY_FILE"
        echo "DATASET DETAILS" >> "$SUMMARY_FILE"
        echo "------------------------------------------------------------" >> "$SUMMARY_FILE"
        # Extract dataset details section from log
        grep -E "(Number of Sequences|Average Input Length|Average Output Length)" "$log_file" | head -5 >> "$SUMMARY_FILE" 2>/dev/null
        echo "" >> "$SUMMARY_FILE"
        DATASET_DETAILS_CAPTURED=true
    fi

    # Extract batch size from config file
    local batch_size
    batch_size=$(grep -E '^\s*max_batch_size:' "$config_file" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    if [ -z "$batch_size" ]; then
        batch_size="default"
    fi

    # Extract and append results to summary
    echo "" >> "$SUMMARY_FILE"
    echo "------------------------------------------------------------" >> "$SUMMARY_FILE"
    echo "Configuration: $config_name" >> "$SUMMARY_FILE"
    echo "Batch Size: $batch_size" >> "$SUMMARY_FILE"
    echo "Concurrency: $CONCURRENCY" >> "$SUMMARY_FILE"
    echo "Duration: ${duration}s" >> "$SUMMARY_FILE"
    echo "Exit code: $exit_code" >> "$SUMMARY_FILE"
    echo "------------------------------------------------------------" >> "$SUMMARY_FILE"

    if [ $exit_code -eq 0 ]; then
        # Extract specific metrics from log file
        echo "" >> "$SUMMARY_FILE"

        # Total Output Throughput (tokens/sec)
        throughput=$(grep -i "Total Output Throughput" "$log_file" | grep -oE '[0-9]+\.?[0-9]*' | tail -1)
        if [ -n "$throughput" ]; then
            echo "Total Output Throughput (tokens/sec):    $throughput" >> "$SUMMARY_FILE"
        fi

        # Average TTFT (ms)
        ttft=$(grep -i "Average time-to-first-token \[TTFT\]" "$log_file" | grep -oE '[0-9]+\.?[0-9]*' | tail -1)
        if [ -n "$ttft" ]; then
            echo "Average TTFT (ms):                        $ttft" >> "$SUMMARY_FILE"
        fi

        # Average TPOT (ms)
        tpot=$(grep -i "Average time-per-output-token \[TPOT\]" "$log_file" | grep -oE '[0-9]+\.?[0-9]*' | tail -1)
        if [ -n "$tpot" ]; then
            echo "Average TPOT (ms):                        $tpot" >> "$SUMMARY_FILE"
        fi

        # Extract Draft Acceptance Rate and Acceptance Length from JSON
        if [ -f "$json_file" ]; then
            python3 -c "
import json

try:
    with open('$json_file', 'r') as f:
        data = json.load(f)

    if 'decoding_stats' in data:
        stats = data['decoding_stats']

        if 'draft_acceptance_rate_percentiles' in stats:
            dar = stats['draft_acceptance_rate_percentiles']
            avg = dar.get('average', None)
            if avg is not None:
                print(f'Average Draft Acceptance Rate:            {avg:.4f}')

        if 'acceptance_length_percentiles' in stats:
            al = stats['acceptance_length_percentiles']
            avg = al.get('average', None)
            if avg is not None:
                print(f'Average Acceptance Length:                {avg:.4f}')
except Exception as e:
    pass
" >> "$SUMMARY_FILE" 2>/dev/null
        fi
    else
        echo "  FAILED - Check $log_file for details" >> "$SUMMARY_FILE"
    fi

    echo ""
    return $exit_code
}

# ============================================================
# Run All Benchmarks
# ============================================================
echo ""
echo "Starting benchmarks..."
echo ""

# Track overall success
all_passed=true

# Optional MTP runs (usually disabled on single-H100 setups)
if [ "$ENABLE_MTP" = "true" ]; then
    SAVED_CONCURRENCY=$CONCURRENCY
    CONCURRENCY=$MTP_CONCURRENCY

    # 1. MTP only (with cudagraph and overlap scheduler)
    echo ""
    echo "============================================================"
    echo "Part 1: MTP Speculative Decoding (batch=$MTP_BATCH_SIZE, concurrency=$MTP_CONCURRENCY)"
    echo "============================================================"
    echo ""

    if ! run_benchmark "mtp" "$OUTPUT_DIR/config_mtp.yml" "$MODEL_NAME" "$MODEL_PATH"; then
        echo "WARNING: MTP benchmark failed"
        all_passed=false
    fi

    # 2. MTP + SA
    echo ""
    echo "============================================================"
    echo "Part 2: MTP + SA (batch=$MTP_BATCH_SIZE, concurrency=$MTP_CONCURRENCY)"
    echo "============================================================"
    echo ""

    if ! run_benchmark "mtp_sa" "$OUTPUT_DIR/config_mtp_sa.yml" "$MODEL_NAME" "$MODEL_PATH"; then
        echo "WARNING: MTP+SA benchmark failed"
        all_passed=false
    fi

    # 3. MTP + SA + Global Pool
    echo ""
    echo "============================================================"
    echo "Part 3: MTP + SA + Global Pool (batch=$MTP_BATCH_SIZE, concurrency=$MTP_CONCURRENCY)"
    echo "============================================================"
    echo ""

    if ! run_benchmark "mtp_sa_global" "$OUTPUT_DIR/config_mtp_sa_global.yml" "$MODEL_NAME" "$MODEL_PATH"; then
        echo "WARNING: MTP+SA+Global Pool benchmark failed"
        all_passed=false
    fi

    CONCURRENCY=$SAVED_CONCURRENCY
else
    echo "Skipping MTP and MTP+SA benchmarks (ENABLE_MTP=false)"
fi

# Optional EAGLE3 runs
if [ "$ENABLE_EAGLE3" = "true" ]; then
    SAVED_CONCURRENCY=$CONCURRENCY
    CONCURRENCY=$EAGLE3_CONCURRENCY

    if [[ ",$EAGLE3_VARIANTS," == *",base,"* ]]; then
        # 3. EAGLE3 Speculative Decoding
        echo ""
        echo "============================================================"
        echo "Part 3: EAGLE3 Speculative Decoding (batch=$EAGLE3_BATCH_SIZE, concurrency=$EAGLE3_CONCURRENCY)"
        echo "============================================================"
        echo ""

        if ! run_benchmark "eagle3" "$OUTPUT_DIR/config_eagle3.yml" "$EAGLE3_TARGET_NAME" "$EAGLE3_TARGET_PATH"; then
            echo "WARNING: EAGLE3 benchmark failed"
            all_passed=false
        fi
    else
        echo "Skipping EAGLE3 base (EAGLE3_VARIANTS=$EAGLE3_VARIANTS)"
    fi

    if [[ ",$EAGLE3_VARIANTS," == *",sa,"* ]]; then
        # 4. EAGLE3 + SA
        echo ""
        echo "============================================================"
        echo "Part 4: EAGLE3 + SA (batch=$EAGLE3_BATCH_SIZE, concurrency=$EAGLE3_CONCURRENCY)"
        echo "============================================================"
        echo ""

        if ! run_benchmark "eagle3_sa" "$OUTPUT_DIR/config_eagle3_sa.yml" "$EAGLE3_TARGET_NAME" "$EAGLE3_TARGET_PATH"; then
            echo "WARNING: EAGLE3+SA benchmark failed"
            all_passed=false
        fi
    else
        echo "Skipping EAGLE3+SA (EAGLE3_VARIANTS=$EAGLE3_VARIANTS)"
    fi

    if [[ ",$EAGLE3_VARIANTS," == *",sa_global,"* ]]; then
        # 5. EAGLE3 + SA + Global Pool
        echo ""
        echo "============================================================"
        echo "Part 5: EAGLE3 + SA + Global Pool (batch=$EAGLE3_BATCH_SIZE, concurrency=$EAGLE3_CONCURRENCY)"
        echo "============================================================"
        echo ""

        if ! run_benchmark "eagle3_sa_global" "$OUTPUT_DIR/config_eagle3_sa_global.yml" "$EAGLE3_TARGET_NAME" "$EAGLE3_TARGET_PATH"; then
            echo "WARNING: EAGLE3+SA+Global Pool benchmark failed"
            all_passed=false
        fi
    else
        echo "Skipping EAGLE3+SA+Global Pool (EAGLE3_VARIANTS=$EAGLE3_VARIANTS)"
    fi

    CONCURRENCY=$SAVED_CONCURRENCY
else
    echo "Skipping EAGLE3 and EAGLE3+SA benchmarks (ENABLE_EAGLE3=false)"
fi

# Optional PARD runs
if [ "$ENABLE_PARD" = "true" ]; then
    SAVED_CONCURRENCY=$CONCURRENCY
    CONCURRENCY=$PARD_CONCURRENCY

    # 5. PARD Speculative Decoding
    echo ""
    echo "============================================================"
    echo "Part 5: PARD Speculative Decoding (batch=$PARD_BATCH_SIZE, concurrency=$PARD_CONCURRENCY, K=$PARD_DRAFT_LEN)"
    echo "============================================================"
    echo ""

    if ! run_benchmark "pard" "$OUTPUT_DIR/config_pard.yml" "$PARD_TARGET_NAME" "$PARD_TARGET_PATH"; then
        echo "WARNING: PARD benchmark failed"
        all_passed=false
    fi

    # 6. PARD + SA
    echo ""
    echo "============================================================"
    echo "Part 6: PARD + SA (batch=$PARD_BATCH_SIZE, concurrency=$PARD_CONCURRENCY, K=$PARD_DRAFT_LEN)"
    echo "============================================================"
    echo ""

    if ! run_benchmark "pard_sa" "$OUTPUT_DIR/config_pard_sa.yml" "$PARD_TARGET_NAME" "$PARD_TARGET_PATH"; then
        echo "WARNING: PARD+SA benchmark failed"
        all_passed=false
    fi

    # 7. PARD + SA + Global Pool
    echo ""
    echo "============================================================"
    echo "Part 7: PARD + SA + Global Pool (batch=$PARD_BATCH_SIZE, concurrency=$PARD_CONCURRENCY, K=$PARD_DRAFT_LEN)"
    echo "============================================================"
    echo ""

    if ! run_benchmark "pard_sa_global" "$OUTPUT_DIR/config_pard_sa_global.yml" "$PARD_TARGET_NAME" "$PARD_TARGET_PATH"; then
        echo "WARNING: PARD+SA+Global Pool benchmark failed"
        all_passed=false
    fi

    CONCURRENCY=$SAVED_CONCURRENCY
else
    echo "Skipping PARD and PARD+SA benchmarks (ENABLE_PARD=false)"
fi

# 8. Baseline (no speculative decoding)
# MTP and EAGLE3/PARD use different target models, so run separate baselines.
# Match concurrency to the corresponding speculative decoding config for fair comparison.
if [ "$ENABLE_MTP" = "true" ]; then
    SAVED_CONCURRENCY=$CONCURRENCY
    CONCURRENCY=$MTP_CONCURRENCY

    echo ""
    echo "============================================================"
    echo "Part 8a: Baseline for MTP (${MODEL_NAME}, concurrency=$CONCURRENCY)"
    echo "============================================================"
    echo ""

    if ! run_benchmark "baseline" "$OUTPUT_DIR/config_baseline.yml" "$MODEL_NAME" "$MODEL_PATH"; then
        echo "WARNING: MTP Baseline benchmark failed"
        all_passed=false
    fi

    CONCURRENCY=$SAVED_CONCURRENCY
fi

if [ "$ENABLE_EAGLE3" = "true" ] || [ "$ENABLE_PARD" = "true" ]; then
    SAVED_CONCURRENCY=$CONCURRENCY
    if [ "$ENABLE_EAGLE3" = "true" ]; then
        CONCURRENCY=$EAGLE3_CONCURRENCY
    else
        CONCURRENCY=$PARD_CONCURRENCY
    fi

    echo ""
    echo "============================================================"
    echo "Part 8b: Baseline for EAGLE3/PARD (${EAGLE3_TARGET_NAME}, concurrency=$CONCURRENCY)"
    echo "============================================================"
    echo ""

    baseline_ep_name="baseline"
    # Avoid overwriting MTP baseline if both are enabled
    if [ "$ENABLE_MTP" = "true" ]; then
        baseline_ep_name="baseline_ep"
    fi

    if ! run_benchmark "$baseline_ep_name" "$OUTPUT_DIR/config_baseline.yml" "$EAGLE3_TARGET_NAME" "$EAGLE3_TARGET_PATH"; then
        echo "WARNING: EAGLE3/PARD Baseline benchmark failed"
        all_passed=false
    fi

    CONCURRENCY=$SAVED_CONCURRENCY
fi

# Fallback: if neither MTP nor EAGLE3/PARD enabled, run MTP model baseline
if [ "$ENABLE_MTP" != "true" ] && [ "$ENABLE_EAGLE3" != "true" ] && [ "$ENABLE_PARD" != "true" ]; then
    echo ""
    echo "============================================================"
    echo "Part 8: Baseline (No Speculative Decoding)"
    echo "============================================================"
    echo ""

    if ! run_benchmark "baseline" "$OUTPUT_DIR/config_baseline.yml" "$MODEL_NAME" "$MODEL_PATH"; then
        echo "WARNING: Baseline benchmark failed"
        all_passed=false
    fi
fi

# ============================================================
# Generate Comparison Table
# ============================================================
echo "" >> "$SUMMARY_FILE"
echo "============================================================" >> "$SUMMARY_FILE"
echo "COMPARISON TABLE" >> "$SUMMARY_FILE"
echo "============================================================" >> "$SUMMARY_FILE"

python3 << PYTHON_SCRIPT >> "$SUMMARY_FILE"
import json
import re
from pathlib import Path

output_dir = "$OUTPUT_DIR"
enable_mtp = "${ENABLE_MTP}".lower() == "true"
enable_eagle3 = "${ENABLE_EAGLE3}".lower() == "true"
enable_pard = "${ENABLE_PARD}".lower() == "true"

def extract_from_log(log_file):
    """Extract metrics from log file."""
    throughput = None
    ttft = None
    tpot = None

    try:
        with open(log_file, 'r') as f:
            content = f.read()

        match = re.search(r'Total Output Throughput.*?([0-9]+\.?[0-9]*)', content, re.IGNORECASE)
        if match:
            throughput = float(match.group(1))

        match = re.search(r'Average time-to-first-token \[TTFT\].*?([0-9]+\.?[0-9]*)', content, re.IGNORECASE)
        if match:
            ttft = float(match.group(1))

        match = re.search(r'Average time-per-output-token \[TPOT\].*?([0-9]+\.?[0-9]*)', content, re.IGNORECASE)
        if match:
            tpot = float(match.group(1))
    except Exception:
        pass

    return throughput, ttft, tpot

def load_metrics(config_name):
    """Load metrics for a given config name."""
    json_file = Path(output_dir) / f"bench_{config_name}.json"
    log_file = Path(output_dir) / f"bench_{config_name}.log"

    if not json_file.exists():
        return None

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        perf = data.get('performance', {})
        streaming = data.get('streaming_metrics', {})
        throughput = perf.get('system_output_throughput_tok_s')
        ttft = streaming.get('avg_ttft_ms')
        tpot = streaming.get('avg_tpot_ms')

        if throughput is None or ttft is None or tpot is None:
            log_tp, log_ttft, log_tpot = extract_from_log(log_file)
            throughput = throughput or log_tp
            ttft = ttft or log_ttft
            tpot = tpot or log_tpot

        decoding = data.get('decoding_stats', {}) or {}

        acceptance_rate = None
        acceptance_len = None
        dar = decoding.get('draft_acceptance_rate_percentiles')
        if dar is not None:
            acceptance_rate = dar.get('average')
        al = decoding.get('acceptance_length_percentiles')
        if al is not None:
            acceptance_len = al.get('average')

        return {
            'throughput': throughput,
            'ttft': ttft,
            'tpot': tpot,
            'accept_rate': acceptance_rate,
            'accept_len': acceptance_len
        }
    except Exception:
        return None

def format_val(val, fmt=':.2f'):
    if val is None:
        return "N/A"
    if fmt == ':.2f':
        return f"{val:.2f}"
    elif fmt == ':.4f':
        return f"{val:.4f}"
    return str(val)

# Load metrics
mtp_metrics = load_metrics("mtp")
mtp_sa_metrics = load_metrics("mtp_sa")
mtp_sa_global_metrics = load_metrics("mtp_sa_global")
eagle3_metrics = load_metrics("eagle3")
eagle3_sa_metrics = load_metrics("eagle3_sa")
eagle3_sa_global_metrics = load_metrics("eagle3_sa_global")
pard_metrics = load_metrics("pard")
pard_sa_metrics = load_metrics("pard_sa")
pard_sa_global_metrics = load_metrics("pard_sa_global")
baseline_metrics = load_metrics("baseline")
# EAGLE3/PARD use a separate baseline when MTP is also enabled
baseline_ep_metrics = load_metrics("baseline_ep") if (enable_mtp and (enable_eagle3 or enable_pard)) else None

def get_baseline_for(group):
    """Return the correct baseline for a method group."""
    if group == "mtp":
        return baseline_metrics
    else:
        return baseline_ep_metrics or baseline_metrics

def print_row(label, m):
    if m:
        tp = format_val(m['throughput'])
        ttft = format_val(m['ttft'])
        tpot = format_val(m['tpot'])
        ar = format_val(m['accept_rate'], ':.4f')
        al = format_val(m['accept_len'])
        print(f"{label:<25} {tp:>22} {ttft:>12} {tpot:>12} {ar:>12} {al:>12}")
    else:
        print(f"{label:<25} {'N/A':>22} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

def print_baseline_row(label, m):
    if m:
        tp = format_val(m['throughput'])
        ttft = format_val(m['ttft'])
        tpot = format_val(m['tpot'])
        print(f"{label:<25} {tp:>22} {ttft:>12} {tpot:>12} {'N/A':>12} {'N/A':>12}")
    else:
        print(f"{label:<25} {'N/A':>22} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

# Print comparison table
print("")
print(f"{'Configuration':<25} {'Total Output TP(tok/s)':>22} {'TTFT(ms)':>12} {'TPOT(ms)':>12} {'Accept%':>12} {'AcceptLen':>12}")
print("-" * 95)

if enable_mtp:
    print_row("MTP", mtp_metrics)
    print_row("MTP + SA", mtp_sa_metrics)
    print_row("MTP + SA + Global", mtp_sa_global_metrics)
    print_baseline_row("Baseline (MTP model)", baseline_metrics)

if enable_eagle3:
    ep_bl = get_baseline_for("eagle3")
    print_row("EAGLE3", eagle3_metrics)
    print_row("EAGLE3 + SA", eagle3_sa_metrics)
    print_row("EAGLE3 + SA + Global", eagle3_sa_global_metrics)
    print_baseline_row("Baseline (EAGLE3 model)", ep_bl)

if enable_pard:
    ep_bl = get_baseline_for("pard")
    print_row("PARD", pard_metrics)
    print_row("PARD + SA", pard_sa_metrics)
    print_row("PARD + SA + Global", pard_sa_global_metrics)
    if not enable_eagle3:
        print_baseline_row("Baseline (PARD model)", ep_bl)

if not enable_mtp and not enable_eagle3 and not enable_pard:
    print_baseline_row("Baseline", baseline_metrics)

print("")

# Print speedup comparison
def print_speedups(group_name, methods, bl):
    if not bl or not bl['throughput']:
        return
    baseline_tp = bl['throughput']
    print(f"Speedup vs Baseline ({group_name}):")
    for label, m in methods:
        if m and m['throughput']:
            speedup = m['throughput'] / baseline_tp
            print(f"  {label + ':':<20}{speedup:.2f}x")
    print("")

if enable_mtp:
    print_speedups("MTP model", [("MTP", mtp_metrics), ("MTP + SA", mtp_sa_metrics), ("MTP + SA + Global", mtp_sa_global_metrics)], baseline_metrics)

if enable_eagle3:
    ep_bl = get_baseline_for("eagle3")
    print_speedups("EAGLE3 model", [("EAGLE3", eagle3_metrics), ("EAGLE3 + SA", eagle3_sa_metrics), ("EAGLE3 + SA + Global", eagle3_sa_global_metrics)], ep_bl)

if enable_pard:
    ep_bl = get_baseline_for("pard")
    print_speedups("PARD model", [("PARD", pard_metrics), ("PARD + SA", pard_sa_metrics), ("PARD + SA + Global", pard_sa_global_metrics)], ep_bl)

# Print SA improvement over base method
def print_sa_improvement(base_label, base_m, sa_label, sa_m):
    if not (base_m and sa_m and base_m['throughput'] and sa_m['throughput']):
        return
    improvement = (sa_m['throughput'] - base_m['throughput']) / base_m['throughput'] * 100
    pad = " " * len(f"{sa_label} Improvement over {base_label}: ")
    print(f"{sa_label} Improvement over {base_label}: {improvement:+.1f}% throughput")

    if base_m['accept_rate'] and sa_m['accept_rate']:
        ar_improvement = (sa_m['accept_rate'] - base_m['accept_rate']) / base_m['accept_rate'] * 100
        print(f"{pad}{ar_improvement:+.1f}% acceptance rate")

    if base_m['accept_len'] and sa_m['accept_len']:
        al_improvement = (sa_m['accept_len'] - base_m['accept_len']) / base_m['accept_len'] * 100
        print(f"{pad}{al_improvement:+.1f}% acceptance length")

if enable_mtp:
    print_sa_improvement("MTP", mtp_metrics, "SA", mtp_sa_metrics)
    print_sa_improvement("MTP", mtp_metrics, "SA+Global", mtp_sa_global_metrics)

if enable_eagle3:
    print_sa_improvement("EAGLE3", eagle3_metrics, "SA", eagle3_sa_metrics)
    print_sa_improvement("EAGLE3", eagle3_metrics, "SA+Global", eagle3_sa_global_metrics)

if enable_pard:
    print_sa_improvement("PARD", pard_metrics, "SA", pard_sa_metrics)
    print_sa_improvement("PARD", pard_metrics, "SA+Global", pard_sa_global_metrics)

print("")
PYTHON_SCRIPT

# ============================================================
# Generate Markdown Report (comparison_table.md)
# ============================================================
MD_FILE="$OUTPUT_DIR/comparison_table.md"
echo "Generating markdown report: $MD_FILE"

python3 << PYTHON_MD_SCRIPT > "$MD_FILE"
import json
import re
from pathlib import Path

output_dir = "$OUTPUT_DIR"
model_path = "$MODEL_PATH"
eagle3_target_path = "$EAGLE3_TARGET_PATH"
pard_target_path = "$PARD_TARGET_PATH"
dataset_path = "$DATASET"
source_json = "$SOURCE_JSON"
dataset_type = "$DATASET_TYPE"
repobench_json = "$REPOBENCH_JSON"
repobench_max_repos = "$REPOBENCH_MAX_REPOS"
repobench_max_per_repo = "$REPOBENCH_MAX_PER_REPO"
cceval_language = "$CCEVAL_LANGUAGE"
cceval_max_repos = "$CCEVAL_MAX_REPOS"
swebench_max_repos = "$SWEBENCH_MAX_REPOS"
swebench_max_per_repo = "$SWEBENCH_MAX_PER_REPO"
cceval_max_per_repo = "$CCEVAL_MAX_PER_REPO"
sample_size = "$SAMPLE_SIZE"
random_seed = "$RANDOM_SEED"
tp_size = "$TP_SIZE"
enable_mtp = "${ENABLE_MTP}".lower() == "true"
enable_eagle3 = "${ENABLE_EAGLE3}".lower() == "true"
enable_pard = "${ENABLE_PARD}".lower() == "true"

def extract_from_log(log_file):
    """Extract metrics from log file."""
    throughput = None
    ttft = None
    tpot = None

    try:
        with open(log_file, 'r') as f:
            content = f.read()

        match = re.search(r'Total Output Throughput.*?([0-9]+\.?[0-9]*)', content, re.IGNORECASE)
        if match:
            throughput = float(match.group(1))

        match = re.search(r'Average time-to-first-token \[TTFT\].*?([0-9]+\.?[0-9]*)', content, re.IGNORECASE)
        if match:
            ttft = float(match.group(1))

        match = re.search(r'Average time-per-output-token \[TPOT\].*?([0-9]+\.?[0-9]*)', content, re.IGNORECASE)
        if match:
            tpot = float(match.group(1))
    except Exception:
        pass

    return throughput, ttft, tpot

def extract_dataset_info(log_file):
    """Extract dataset info from log file."""
    info = {}
    try:
        with open(log_file, 'r') as f:
            content = f.read()

        match = re.search(r'Number of Sequences:\s*(\d+)', content)
        if match:
            info['num_sequences'] = int(match.group(1))

        match = re.search(r'Average Input Length \(tokens\):\s*([0-9.]+)', content)
        if match:
            info['avg_input_len'] = float(match.group(1))

        match = re.search(r'Average Output Length \(tokens\):\s*([0-9.]+)', content)
        if match:
            info['avg_output_len'] = float(match.group(1))
    except Exception:
        pass

    return info

def load_metrics(config_name):
    """Load metrics for a given config name."""
    json_file = Path(output_dir) / f"bench_{config_name}.json"
    log_file = Path(output_dir) / f"bench_{config_name}.log"

    if not json_file.exists():
        return None

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        perf = data.get('performance', {})
        streaming = data.get('streaming_metrics', {})
        throughput = perf.get('system_output_throughput_tok_s')
        ttft = streaming.get('avg_ttft_ms')
        tpot = streaming.get('avg_tpot_ms')

        if throughput is None or ttft is None or tpot is None:
            log_tp, log_ttft, log_tpot = extract_from_log(log_file)
            throughput = throughput or log_tp
            ttft = ttft or log_ttft
            tpot = tpot or log_tpot

        decoding = data.get('decoding_stats', {}) or {}

        acceptance_rate = None
        acceptance_len = None
        dar = decoding.get('draft_acceptance_rate_percentiles')
        if dar is not None:
            acceptance_rate = dar.get('average')
        al = decoding.get('acceptance_length_percentiles')
        if al is not None:
            acceptance_len = al.get('average')

        return {
            'throughput': throughput,
            'ttft': ttft,
            'tpot': tpot,
            'accept_rate': acceptance_rate,
            'accept_len': acceptance_len
        }
    except Exception:
        return None

def format_val(val, fmt=':.2f'):
    if val is None:
        return "N/A"
    if fmt == ':.2f':
        return f"{val:.2f}"
    elif fmt == ':.4f':
        return f"{val:.4f}"
    return str(val)

def format_comparison_md(val1, val2, metric_type, fmt=':.2f'):
    """Format comparison for markdown: val1 (val2), bold val1 if >5% better."""
    if val1 is None:
        return "N/A"

    if fmt == ':.2f':
        val1_str = f"{val1:.2f}"
        val2_str = f"{val2:.2f}" if val2 is not None else None
    else:
        val1_str = f"{val1:.4f}"
        val2_str = f"{val2:.4f}" if val2 is not None else None

    if val2 is None:
        return val1_str

    # Determine if val1 is better (>5% difference)
    threshold = 0.05
    val1_better = False

    if metric_type in ['throughput', 'accept_rate', 'accept_len']:
        # Higher is better
        if val2 > 0 and (val1 - val2) / val2 > threshold:
            val1_better = True
    else:
        # Lower is better (ttft, tpot)
        if val1 > 0 and (val2 - val1) / val1 > threshold:
            val1_better = True

    if val1_better:
        return f"**{val1_str}** ({val2_str})"
    else:
        return f"{val1_str} ({val2_str})"

# Get dataset info from first available log
dataset_info = {}
for config in ["mtp", "mtp_sa", "mtp_sa_global", "eagle3", "eagle3_sa", "eagle3_sa_global", "pard", "pard_sa", "pard_sa_global", "baseline"]:
    log_file = Path(output_dir) / f"bench_{config}.log"
    if log_file.exists():
        dataset_info = extract_dataset_info(log_file)
        if dataset_info:
            break

# Load all metrics
mtp_metrics = load_metrics("mtp")
mtp_sa_metrics = load_metrics("mtp_sa")
mtp_sa_global_metrics = load_metrics("mtp_sa_global")
eagle3_metrics = load_metrics("eagle3")
eagle3_sa_metrics = load_metrics("eagle3_sa")
eagle3_sa_global_metrics = load_metrics("eagle3_sa_global")
pard_metrics = load_metrics("pard")
pard_sa_metrics = load_metrics("pard_sa")
pard_sa_global_metrics = load_metrics("pard_sa_global")
baseline_metrics = load_metrics("baseline")
baseline_ep_metrics = load_metrics("baseline_ep") if (enable_mtp and (enable_eagle3 or enable_pard)) else None

def get_baseline_for(group):
    if group == "mtp":
        return baseline_metrics
    else:
        return baseline_ep_metrics or baseline_metrics

# Generate markdown
print("# DeepSeek V3.1-NVFP4 Speculative Decoding + SA Benchmark Results")
print("")
print("## Configuration")
print("")
print("| Setting | Value |")
print("|---------|-------|")
print(f"| **MTP Target** | {model_path} |")
print(f"| **EAGLE3 Target** | {eagle3_target_path} |")
print(f"| **PARD Target** | {pard_target_path} |")
print(f"| **TP Size** | {tp_size} |")
print(f"| **Dataset Type** | {dataset_type} |")
print(f"| **Sample Size** | {sample_size} (seed: {random_seed}) |")
if dataset_type == "repobench":
    print(f"| **Source Data** | {repobench_json} |")
    print(f"| **Max Repos** | {repobench_max_repos} |")
    print(f"| **Max Per Repo** | {repobench_max_per_repo} (0=all) |")
elif dataset_type == "crosscodeeval":
    print(f"| **Source Data** | Vincentvmt/CrossCodeEval (HuggingFace) |")
    print(f"| **Language** | {cceval_language} |")
    print(f"| **Max Repos** | {cceval_max_repos} |")
    print(f"| **Max Per Repo** | {cceval_max_per_repo} (0=all) |")
elif dataset_type == "swebench":
    print(f"| **Source Data** | princeton-nlp/SWE-bench_Verified (HuggingFace) |")
    print(f"| **Max Repos** | {swebench_max_repos} |")
    print(f"| **Max Per Repo** | {swebench_max_per_repo} (0=all) |")
else:
    print(f"| **Source Data** | {source_json} |")
print(f"| **CUDA Graph** | Enabled (max_batch_size=8) |")
print(f"| **Overlap Scheduler** | Enabled |")
if enable_mtp:
    print(f"| **MTP Layers** | 8 |")
if enable_eagle3:
    print(f"| **EAGLE3 Draft Length** | 4 |")
if enable_pard:
    print(f"| **PARD Draft Length** | 4 |")
print(f"| **SA Threshold** | 4 |")
print("")

print("## Dataset Details")
print("")
print("| Metric | Value |")
print("|--------|-------|")
if dataset_type == "repobench":
    dataset_label = "RepoBench-P v1.1 (cross_file_first, repo-grouped)"
elif dataset_type == "crosscodeeval":
    dataset_label = f"CrossCodeEval ({cceval_language}, oracle_bm25, repo-grouped)"
elif dataset_type == "swebench":
    dataset_label = "SWE-bench Verified (issue + hints + test_patch, repo-grouped)"
else:
    dataset_label = "Code Edit Examples (randomly sampled)"
print(f"| **Dataset** | {dataset_label} |")
print(f"| **Number of Requests** | {dataset_info.get('num_sequences', 'N/A')} |")
print(f"| **Average Input Length** | {dataset_info.get('avg_input_len', 'N/A')} tokens |")
print(f"| **Average Output Length** | {dataset_info.get('avg_output_len', 'N/A')} tokens |")
print("")

print("## Comparison Table")
print("")
print("| Configuration | Total Output Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Accept Rate | Accept Length |")
print("|---------------|--------------------------------|-----------|-----------|-------------|---------------|")

def print_md_row(label, m):
    if m:
        tp = format_val(m['throughput'])
        ttft = format_val(m['ttft'])
        tpot = format_val(m['tpot'])
        ar = format_val(m['accept_rate'], ':.4f')
        al = format_val(m['accept_len'])
        print(f"| {label} | {tp} | {ttft} | {tpot} | {ar} | {al} |")
    else:
        print(f"| {label} | N/A | N/A | N/A | N/A | N/A |")

def print_md_comparison_row(label, m, ref):
    if m:
        tp = format_comparison_md(m['throughput'], ref['throughput'] if ref else None, 'throughput')
        ttft = format_comparison_md(m['ttft'], ref['ttft'] if ref else None, 'ttft')
        tpot = format_comparison_md(m['tpot'], ref['tpot'] if ref else None, 'tpot')
        ar = format_comparison_md(m['accept_rate'], ref['accept_rate'] if ref else None, 'accept_rate', ':.4f')
        al = format_comparison_md(m['accept_len'], ref['accept_len'] if ref else None, 'accept_len')
        print(f"| {label} | {tp} | {ttft} | {tpot} | {ar} | {al} |")
    else:
        print(f"| {label} | N/A | N/A | N/A | N/A | N/A |")

def print_md_baseline_row(label, m):
    if m:
        tp = format_val(m['throughput'])
        ttft = format_val(m['ttft'])
        tpot = format_val(m['tpot'])
        print(f"| {label} | {tp} | {ttft} | {tpot} | N/A | N/A |")
    else:
        print(f"| {label} | N/A | N/A | N/A | N/A | N/A |")

if enable_mtp:
    print_md_row("MTP", mtp_metrics)
    print_md_comparison_row("MTP + SA", mtp_sa_metrics, mtp_metrics)
    print_md_comparison_row("MTP + SA + Global", mtp_sa_global_metrics, mtp_metrics)
    print_md_baseline_row("Baseline (MTP model)", baseline_metrics)

if enable_eagle3:
    ep_bl = get_baseline_for("eagle3")
    print_md_row("EAGLE3", eagle3_metrics)
    print_md_comparison_row("EAGLE3 + SA", eagle3_sa_metrics, eagle3_metrics)
    print_md_comparison_row("EAGLE3 + SA + Global", eagle3_sa_global_metrics, eagle3_metrics)
    print_md_baseline_row("Baseline (EAGLE3 model)", ep_bl)

if enable_pard:
    ep_bl = get_baseline_for("pard")
    print_md_row("PARD", pard_metrics)
    print_md_comparison_row("PARD + SA", pard_sa_metrics, pard_metrics)
    print_md_comparison_row("PARD + SA + Global", pard_sa_global_metrics, pard_metrics)
    if not enable_eagle3:
        print_md_baseline_row("Baseline (PARD model)", ep_bl)

if not enable_mtp and not enable_eagle3 and not enable_pard:
    print_md_baseline_row("Baseline", baseline_metrics)

print("")
print("## Key Observations")
print("")

# Calculate improvements helper
def print_md_improvement(heading, base_m, sa_m):
    if not (base_m and sa_m and base_m['throughput'] and sa_m['throughput']):
        return
    tp_improvement = (sa_m['throughput'] - base_m['throughput']) / base_m['throughput'] * 100
    print(f"### {heading}")
    print(f"- **Total Output Throughput:** {tp_improvement:+.1f}%")

    if base_m['tpot'] and sa_m['tpot']:
        tpot_improvement = (base_m['tpot'] - sa_m['tpot']) / base_m['tpot'] * 100
        print(f"- **TPOT:** {tpot_improvement:+.1f}% (lower is better)")

    if base_m['accept_rate'] and sa_m['accept_rate']:
        ar_improvement = (sa_m['accept_rate'] - base_m['accept_rate']) / base_m['accept_rate'] * 100
        print(f"- **Acceptance Rate:** {sa_m['accept_rate']:.2%} vs {base_m['accept_rate']:.2%} ({ar_improvement:+.1f}%)")

    if base_m['accept_len'] and sa_m['accept_len']:
        al_improvement = (sa_m['accept_len'] - base_m['accept_len']) / base_m['accept_len'] * 100
        print(f"- **Acceptance Length:** {sa_m['accept_len']:.2f} vs {base_m['accept_len']:.2f} tokens ({al_improvement:+.1f}%)")
    print("")

if enable_mtp:
    print_md_improvement("SA Improvement over MTP", mtp_metrics, mtp_sa_metrics)
    print_md_improvement("SA + Global Pool Improvement over MTP", mtp_metrics, mtp_sa_global_metrics)

if enable_eagle3:
    print_md_improvement("SA Improvement over EAGLE3", eagle3_metrics, eagle3_sa_metrics)
    print_md_improvement("SA + Global Pool Improvement over EAGLE3", eagle3_metrics, eagle3_sa_global_metrics)

if enable_pard:
    print_md_improvement("SA Improvement over PARD", pard_metrics, pard_sa_metrics)
    print_md_improvement("SA + Global Pool Improvement over PARD", pard_metrics, pard_sa_global_metrics)

# Speedup vs baseline (per-group)
def print_md_speedup_table(group_name, methods, bl):
    if not bl or not bl['throughput']:
        return
    bl_tp = bl['throughput']
    print(f"### Speedup vs Baseline ({group_name}, {bl_tp:.2f} tok/s)")
    print(f"| Configuration | Speedup |")
    print("|---------------|---------|")
    for label, m in methods:
        if m and m['throughput']:
            speedup = m['throughput'] / bl_tp
            print(f"| {label} | **{speedup:.2f}x** |")
    print("")

if enable_mtp:
    print_md_speedup_table("MTP model", [("MTP", mtp_metrics), ("MTP + SA", mtp_sa_metrics), ("MTP + SA + Global", mtp_sa_global_metrics)], baseline_metrics)

if enable_eagle3:
    ep_bl = get_baseline_for("eagle3")
    print_md_speedup_table("EAGLE3 model", [("EAGLE3", eagle3_metrics), ("EAGLE3 + SA", eagle3_sa_metrics), ("EAGLE3 + SA + Global", eagle3_sa_global_metrics)], ep_bl)

if enable_pard:
    ep_bl = get_baseline_for("pard")
    print_md_speedup_table("PARD model", [("PARD", pard_metrics), ("PARD + SA", pard_sa_metrics), ("PARD + SA + Global", pard_sa_global_metrics)], ep_bl)

PYTHON_MD_SCRIPT

echo "Markdown report generated: $MD_FILE"

# ============================================================
# Final Summary
# ============================================================
echo "" >> "$SUMMARY_FILE"
echo "============================================================" >> "$SUMMARY_FILE"
echo "Benchmark Complete: $(date)" >> "$SUMMARY_FILE"
if $all_passed; then
    echo "Status: ALL PASSED" >> "$SUMMARY_FILE"
else
    echo "Status: SOME FAILED - Check individual logs" >> "$SUMMARY_FILE"
fi
echo "============================================================" >> "$SUMMARY_FILE"

echo ""
echo "============================================================"
echo "All benchmarks complete!"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"
echo "  - Summary log: $SUMMARY_FILE"
echo "  - Markdown report: $MD_FILE (ready to copy & share!)"
echo ""
echo "Benchmark configs:"
if [ "$ENABLE_MTP" = "true" ]; then
    echo "  - bench_mtp.{log,json} - MTP only"
    echo "  - bench_mtp_sa.{log,json} - MTP + SA"
    echo "  - bench_mtp_sa_global.{log,json} - MTP + SA + Global Pool"
fi
if [ "$ENABLE_EAGLE3" = "true" ]; then
    echo "  - bench_eagle3.{log,json} - EAGLE3 only"
    echo "  - bench_eagle3_sa.{log,json} - EAGLE3 + SA"
    echo "  - bench_eagle3_sa_global.{log,json} - EAGLE3 + SA + Global Pool"
fi
if [ "$ENABLE_PARD" = "true" ]; then
    echo "  - bench_pard.{log,json} - PARD only"
    echo "  - bench_pard_sa.{log,json} - PARD + SA"
    echo "  - bench_pard_sa_global.{log,json} - PARD + SA + Global Pool"
fi
echo "  - bench_baseline.{log,json} - Baseline for MTP model (no speculative decoding)"
if [ "$ENABLE_MTP" = "true" ] && ([ "$ENABLE_EAGLE3" = "true" ] || [ "$ENABLE_PARD" = "true" ]); then
    echo "  - bench_baseline_ep.{log,json} - Baseline for EAGLE3/PARD model (no speculative decoding)"
fi
echo ""
echo "To view results:"
echo "  cat $SUMMARY_FILE"
echo "  cat $MD_FILE"
echo ""
