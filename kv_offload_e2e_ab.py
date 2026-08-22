# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end A/B runner for the DSA KV-offload consume-mode prototype.

Loads GLM-5.2 with the production-shaped config (TEP + attention DP + MTP3 +
CUTEDSL MoE + DSA heuristic top-k), generates greedily from deterministic
prompts, and dumps the produced token ids to JSON. The KV-offload prototype is
controlled purely through TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_* environment
variables set by the caller, so running this script twice (offload off/on) and
diffing the JSON is the correctness check: bit-exact attention must yield
identical greedy tokens.

Launched under trtllm-llmapi-launch (one task per GPU, --mpi=pmix).
"""

import argparse
import json
import os

from tensorrt_llm import LLM, SamplingParams

# Token lengths chosen to cover: indexer-skip short seqs (< index_topk 2048),
# sparse selection (> 2048), and long-context chunked prefill (max_num_tokens
# 128 => hundreds of chunks with per-chunk host mirroring).
PROMPT_TOKEN_LENS = [200, 1024, 4096, 16384, 32768]
MAX_NEW_TOKENS = 96

BASE_TEXT = (
    "The library of Alexandria was one of the largest and most significant "
    "libraries of the ancient world. It flourished under the patronage of the "
    "Ptolemaic dynasty and functioned as a major center of scholarship. The "
    "library was part of a larger research institution called the Mouseion, "
    "where many of the most famous thinkers of the ancient world studied. "
)


def build_prompts(llm):
    # Reuse the tokenizer the LLM already loaded: a bare
    # AutoTokenizer.from_pretrained trips transformers' strict config
    # validation on GLM-5.2's custom layer_types. Deterministic synthetic ids
    # are an equally valid A/B input if no tokenizer is exposed.
    tokenizer = getattr(llm, "tokenizer", None)
    if tokenizer is not None:
        base_ids = list(tokenizer.encode(BASE_TEXT * 700))
    else:
        print("[e2e_ab] no tokenizer on LLM; using synthetic token ids", flush=True)
        base_ids = [1000 + (i * 37) % 30000 for i in range(40000)]
    assert len(base_ids) >= max(PROMPT_TOKEN_LENS), len(base_ids)
    return [{"prompt_token_ids": base_ids[:n]} for n in PROMPT_TOKEN_LENS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--output", required=True)
    # Bounds the KV pool (and therefore the prototype's pinned host mirror,
    # ~19 groups x max_kv_tokens x 1728 B per rank) instead of letting
    # free_gpu_memory_fraction size a near-HBM-large pool.
    parser.add_argument("--max-kv-tokens", type=int, default=262144)
    args = parser.parse_args()

    offload_env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_")
    }
    print(f"[e2e_ab] offload env: {offload_env or '(none: GPU-resident baseline)'}", flush=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=1,
        moe_expert_parallel_size=args.ep,
        enable_attention_dp=True,
        enable_lm_head_tp_in_adp=False,
        trust_remote_code=True,
        moe_config=dict(backend="CUTEDSL", use_low_precision_moe_combine=True),
        sparse_attention_config=dict(
            algorithm="dsa",
            enable_heuristic_topk=True,
            use_cute_dsl_paged_mqa_logits=True,
        ),
        cuda_graph_config=dict(enable_padding=True, batch_sizes=[1, 2, 4, 8, 16]),
        max_batch_size=16,
        max_num_tokens=128,
        max_seq_len=1048576,
        enable_chunked_prefill=True,
        kv_cache_config=dict(
            dtype="fp8",
            free_gpu_memory_fraction=0.9,
            max_tokens=args.max_kv_tokens,
            host_cache_size=0,
            tokens_per_block=64,
            enable_block_reuse=False,
            event_buffer_max_size=0,
            # The KV-offload prototype hooks DSACacheManager (V1); "auto" now
            # resolves to DSACacheManagerV2, which silently bypasses it.
            use_kv_cache_manager_v2=False,
        ),
        speculative_config=dict(decoding_type="MTP", max_draft_len=3),
        stream_interval=20,
        num_postprocess_workers=4,
    )

    sampling = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        top_k=1,
        ignore_eos=True,
    )
    prompts = build_prompts(llm)

    try:
        # Phase 1: one request at a time -- deterministic batch composition,
        # the strict token-equality oracle across runs.
        sequential = []
        for i, prompt in enumerate(prompts):
            out = llm.generate([prompt], sampling)[0]
            token_ids = list(out.outputs[0].token_ids)
            sequential.append(token_ids)
            print(
                f"[e2e_ab] sequential {i}: prompt={PROMPT_TOKEN_LENS[i]} tok "
                f"-> {len(token_ids)} tok, head={token_ids[:8]}",
                flush=True,
            )

        # Phase 2: all requests at once -- exercises concurrent gathers, CUDA
        # graph batch buckets, and MTP under load. Batch composition may vary
        # run-to-run, so the comparator gates this phase on baseline
        # reproducibility.
        batched_out = llm.generate(prompts, sampling)
        batched = [list(out.outputs[0].token_ids) for out in batched_out]
        print(f"[e2e_ab] batched: {[len(ids) for ids in batched]}", flush=True)
    finally:
        llm.shutdown()

    with open(args.output, "w") as f:
        json.dump(
            {
                "offload_env": offload_env,
                "prompt_token_lens": PROMPT_TOKEN_LENS,
                "max_new_tokens": MAX_NEW_TOKENS,
                "sequential": sequential,
                "batched": batched,
            },
            f,
        )
    print(f"[e2e_ab] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
