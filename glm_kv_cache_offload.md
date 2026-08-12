# GLM-5.2 IndexShare KV Offload — Decode-Phase Evaluation

*2026-08-10. Evaluation of the idea from the Slack thread
[#C0BHB1UVB1Q/p1784079767330349](https://nvidia.slack.com/archives/C0BHB1UVB1Q/p1784079767330349)
(Jonas Li / Xiaoming Chen / Patrice Castonguay / Allen Yuan / Sharan Chetlur):
move the KV cache of GLM-5.2's "shared" indexer layers (layers 2–4 of each 4-layer index-share group)
to CPU memory and copy back only the top-k-selected tokens per decode iteration.
Grounded in Allen's raw study data
(`/home/scratch.alyuan_other/workspace/glm52-indexer-topk-study-20260714/`) and the current
TensorRT-LLM `main` codebase.*

---

## Part 0 — Slack thread summary (16 replies, Jul 14–27)

### Original post — Jonas Li (Jul 14)

The **GLM-5.2 IndexShare idea**, originally proposed by Xiaoming Chen. In GLM-5.2, every 4 layers share one set of top-K indexer indices: layer 1 runs the indexer to generate indices, and layers 2–4 reuse those indices to select from their own KV cache blocks. Xiaoming's proposal: move the entire KV cache of layers 2–4 to **CPU memory**, then do runtime CPU→GPU copies of just the blocks selected by layer 1's top-k indices. The goal is freeing GPU memory for higher concurrency and better KV cache hit rate.

The China team already discussed it and is **skeptical of the cost-benefit ratio**, citing two challenges: a mid-iteration KV cache manager redesign, and serialized CPU↔GPU transfers. Since the US side now supports this model, Jonas is asking whether it's doable.

### Discussion highlights

**Patrice Castonguay** was open to exploring it and proposed hiding the transfers with a pipelined schedule: layer 1 computes topK indices, then layer N's attention runs on stream 1 while layer N+1's KV transfers on stream 2. He noted the key open question is whether the transfer can be *fully* overlapped with compute. **Xiaoming** added the transfer could overlap with MoE compute too, not just attention.

**The 80% problem:** Xiaoming pointed to Allen Yuan's study showing you'd need to load almost 80% of total KV cache, which kills most of the memory savings. Patrice pushed back — shouldn't that depend on ISL? With K=2048 he'd expect a small fraction at 100K+ context. Xiaoming clarified the 80% is a **prefill** issue: with `max_num_tokens=8192`, you have 8192 queries each selecting 2048 tokens, so their union covers nearly everything. Patrice conceded he was thinking of decode.

**Allen Yuan's study** ([summary here](https://sc.talos.nvidia.com/view/home/scratch.alyuan_other/workspace/glm52-indexer-topk-study-20260714/summary.html)): for a *single* query, touched KV blocks grow slowly with context — at block size 32, the median goes from 432 blocks (8–32k context) to only 686 (512k–1M). But in prefill, the union of blocks across a chunk's tokens covers basically all historical blocks. His conclusion: **the idea could work for decode** — keep KV on CPU, stream small amounts on demand, fit a bigger batch, and improve decode throughput *if* the transfer overlaps.

**Jonas** clarified "mid-iteration KVCM redesign": KV cache movement currently happens at request boundaries; this needs movement at *layer* boundaries within a single iteration — size of that change and the actual perf gain both unclear.

### Outcome

- **Iman Tabrizian** filed tracking ticket [TRTLLM-14330](https://jirasw.nvidia.com/browse/TRTLLM-14330); Patrice tagged Peiheng Hu for visibility.
- **Sharan Chetlur** (Jul 27) asked Allen how hard it'd be to model the decode-side concurrency uplift.
- **Allen's final reply**: it's tractable as a first-order estimate — freed GPU memory ÷ per-request KV footprint = extra batch size (the direct concurrency win). Two dynamics: sparsity grows with context length (longer context → bigger gain), while per-iteration transfer volume scales with batch size. The main uncertainty is the overlap assumption — whether layer-N compute can fully hide layer-(N+1)'s KV transfer at the target batch size.

So the current state: skeptical for prefill (80% coverage kills it), cautiously promising for decode, pending a perf model that Allen sketched but hasn't built yet.

---

## Part 1 — Model facts confirmed from the code

The model-side picture is fully confirmed from the code (this was worth verifying — a few numbers in the Slack thread are already stale):

- GLM-5.2 = `GlmMoeDsaForCausalLM`, riding the DeepSeek-V3.2 DSA path: **MLA, FP8 KV, 576 B/token/layer**, 78 layers, `index_topk=2048`, sharing schedule `freq=4, offset=2` → **21 full-indexer layers, 57 "shared" layers** whose latent KV is the offload candidate (57×576 = 32.8 KB/token, **69% of the KV footprint**).
- Notably, Allen's measured 55,224 B/token predates PR #16558, which already stopped allocating indexer k-cache on shared layers — today's baseline is 47,700 B/token. So the ceiling for this idea: resident KV drops to 14,868 B/token + a small GPU hot-set, ≈ **3× concurrency** at fixed KV budget.
- MTP index-share across draft steps is already merged and tested for GLM-5.2 (`max_draft_len=3`), which conveniently means draft iterations reuse the same top-k — no fetch multiplication there.
- Current GLM-5.2 test configs only exercise `max_seq_len` ≈ 8–10k, where KV isn't the binding constraint — this optimization's payoff is squarely the long-context/agentic regime (the study corpus went to 642k; the model supports ~200k).

---

## Part 2 — Full evaluation

### Verdict

**Feasible and worth pursuing for decode — but only in a specific form**: token-granular delta-fetching against a GPU-resident hot set, not the per-step block prefetch that was implicitly costed in the Slack thread. In that form the math works comfortably on GB200/GB300 and works at long context even on x86+PCIe hosts, yielding **~2.5–3× decode concurrency at 32k–200k context** (hard ceiling 3.21×). The 80%-coverage objection is a prefill fact and doesn't apply to decode. The biggest open risk is not bandwidth — it's an unmeasured cache-hit-rate assumption that can be settled in a day with Allen's existing dumps.

A key input: Allen's study directory is on the shared filesystem (`/home/scratch.alyuan_other/workspace/glm52-indexer-topk-study-20260714/`), so this evaluation uses his **raw stats**, including a metric the thread never discussed.

### 1. Why the thread's numbers looked bad — and what the data actually says

Allen's headline number (686 blocks × 22.4 KB × 3 layers ≈ 46 MB per group per step) embeds two pessimistic choices:

1. **Block granularity.** The top-2048 tokens scatter ~3 tokens per 32-token block at deep context, so fetching whole blocks moves ~7–10× more bytes than needed. Fetching at token granularity (576 B/token/layer, FP8 MLA latent) cuts the naive volume from ~721 MB to **67 MB per sequence per step** across all 57 shared layers. Note: fetch at token granularity: 2048 tokens x 576 bytes x 57 layers = 67.2 MB per sequence per step; fetch at block granularity: 686 blocks x 32 tokens/block x 576 bytes x 57 layers = 720.7 MB per sequence per step

2. **No reuse across steps.** His own `jaccard_gap1` data (per-layer step-to-step Jaccard of the top-k sets: 0.47–0.78, mean 0.61 → **~76% token overlap between consecutive steps**) shows the selection is a stable core plus a churning periphery — and the gap-1→4 curves *flatten*, meaning dropped tokens get re-selected. Keep a per-(request, group) working set on GPU and fetch only misses: at the *measured* 24% churn that's **16 MB/seq/step**; with a working set  **~2× topk**, plausibly **~8 MB/seq/step**.

TRT-LLM already bets on exactly this temporal correlation — the GVR heuristic top-k (`cute_dsl_gvr_topk_decode`, seeded from `heuristic_prev_topk`; tech blog 21) reuses the previous step's indices to accelerate top-k. The same statistical property underwrites delta-fetching.

Two thread-facts confirmed as correct: prefill offload is dead (touched-fraction median is 1.0 up to 256k ctx — worse than the quoted 80%), and one fetch cannot serve multiple groups (cross-group Jaccard only 0.2–0.5; each layer has its own KV values anyway).

### 2. Bandwidth feasibility

Per-group indices are produced by the group's full-indexer layer, so the tightest fetch window is **~1 layer of compute** (~320 µs at TPOT 25 ms / 78 layers); layers 3–4 of each group get 2–3× that. Maximum per-GPU decode batch **B** before transfers stop hiding:

| | miss 24% (measured) | miss ~12% (larger hot set) | full refetch |
|---|---|---|---|
| **GB200/GB300** (C2C, ~400 GB/s eff.) | B ≈ 455 | B ≈ 900 | B ≈ 109 |
| **x86 + PCIe Gen5** (~40 GB/s eff.) | B ≈ 45 | B ≈ 90 | B ≈ 11 |

<details>
<summary><i>Derivation — how the max-B values are computed (click to expand)</i></summary>

> **Constraint modeled (per-layer window).** A group's top-k indices exist only after its full-indexer layer computes them, and the first shared layer's attention starts ~one decoder-layer of compute later. So the fetch for that layer — for all B sequences at once — must hide inside ~one layer's compute time:
>
> ```
> window   w   = TPOT / num_layers   = 25 ms / 78 ≈ 320 µs
> bytes/layer  = B × miss × K × c    = B × miss × 2048 × 576 B
> feasible iff   B × miss × 2048 × 576  ≤  w × BW
> →  B_max     = (w × BW) / (miss × 2048 × 576)
> ```
>
> **Worked example — the "455" cell** (GB200/300, miss 24%):
> - fetch per sequence per layer: 0.239 × 2048 × 576 B ≈ **282 KB**
> - budget per window: 320 µs × 400 GB/s = **128 MB**
> - 128 MB ÷ 282 KB ≈ **455 sequences**
>
> Same formula fills the rest: miss 12% → 141 KB/seq → ≈ 900; full refetch → 1.18 MB/seq → ≈ 109. On PCIe Gen5 the window budget is 320 µs × 40 GB/s = 12.8 MB → 45 / 90 / 11.
>
> **Second (looser) constraint — sustained bandwidth** over the whole step, all 57 layers: `B ≤ BW × TPOT / (miss × K × c × 57)` → 622 (C2C) and 62 (PCIe) at miss 24%. The table shows the per-layer-window limit because it is the tighter of the two.
>
> **Built-in conservatism:** only the group's *first* shared layer has a 1-layer window; layers 3–4 of each group get 2–3 layers of slack.
>
> **Assumptions that move these numbers:** TPOT = 25 ms (B_max scales linearly with TPOT — a 40 ms step raises every limit 1.6×); effective bandwidths 400 GB/s (C2C reads from Grace LPDDR) and 40 GB/s (PCIe Gen5 x16 at ~1.7 KB gathered-read granularity). *The PCIe figure is now measured-backed (§6.2): 74% of memcpy peak at 1728 B rows on a Gen4 H100 → ≈41 GB/s extrapolated to Gen5. C2C still needs a GB200/300 measurement.*
>
> **Cross-check at the memory-derived batch (ties §2 to §3):** at 128k ctx the KV savings yield B ≈ 38 (see §3 derivation). Its traffic: sustained = 38 × 16.1 MB ÷ 25 ms ≈ **25 GB/s**; tightest window = 38 × 282 KB ÷ 320 µs ≈ **34 GB/s** — both under PCIe Gen5's ~40 GB/s. At 64k, B ≈ 72 → 46–63 GB/s: over PCIe's comfortable budget (marginal), far under C2C's.

</details>

Cross-checking against the batches the memory savings actually produce: at 128k ctx the uplifted batch is B≈38 → needs 25 GB/s sustained (fits even PCIe); at 64k, B≈72 → ~46–63 GB/s (marginal on PCIe, trivial on C2C). So: **GB-class hardware is comfortable everywhere; x86 PCIe works for the ≥100k-context regime and is marginal below**. Conveniently, feasibility and payoff both grow with context length. (This dev node's H100 is PCIe Gen4 — halve the PCIe numbers.)

MTP: verify passes carry 1+`max_draft_len` query positions (union ≈ ×1.2–1.5 fetch volume), but `mtp_index_share` is already merged for GLM-5.2 and the working set persists across the draft loop, so drafts don't refetch. Folded into the margins above.

### 3. Concurrency / throughput uplift (Sharan's question)

GLM-5.2 (code-confirmed): 78 layers, all MLA+DSA, 576 B/token/layer FP8 latent; sharing schedule `freq=4, offset=2` → 21 full-indexer layers, **57 shared layers = 68.8% of the 47.7 KB/token KV footprint** is offloadable. (Note: Allen's logged 55.2 KB/token predates PR #16558, which stopped allocating indexer k-cache on shared layers.) With a 4k-token working set per group:

| ctx | seqs/GPU baseline → offload (80 GB KV, B200 TP8) | uplift | host pinned RAM/GPU |
|---|---|---|---|
| 32k | 51 → 129 | 2.5× | ~140 GB |
| 64k | 26 → 72 | 2.8× | ~155 GB |
| 128k | 13 → 38 | 3.0× | ~165 GB |
| 198k | 8 → 25 | 3.1× | ~170 GB |

<details>
<summary><i>Derivation — per-token footprints, the 80 GB budget, and each table column (click to expand)</i></summary>

> **Where 47,700 B/token (baseline) comes from:**
>
> | Component | Formula | Bytes/token |
> |---|---|---|
> | MLA latent KV, all 78 layers | 78 × 576 B (FP8; 512 kv_lora + 64 rope) | 44,928 |
> | Indexer k-cache, 21 full-indexer layers only | 21 × 132 B (128 FP8 + 4 B fp32 scale) | 2,772 |
> | **Total** | | **47,700** |
>
> **Where 14,868 B/token (GPU-resident after offload) comes from** — the offload moves only the 57 shared layers' latent KV (57 × 576 = 32,832 B/token) to CPU; the rest cannot leave the GPU:
>
> | Stays on GPU | Bytes/token | Why it can't be offloaded |
> |---|---|---|
> | Latent KV of the 21 full-indexer layers | 21 × 576 = 12,096 | Their indices are produced *inside the same layer* (indexer → attention back-to-back) — no window to fetch misses. Offloadable in principle with previous-step-index prefetch; a follow-on, not the base design. |
> | Indexer k-cache (21 layers) | 2,772 | The indexer scores **every** historical token each step — a dense scan of the whole context. Sparse fetching is impossible by construction; this is the hard floor. |
> | **Total** | **14,868** | = 47,700 − 32,832. Hence the uplift ceiling 3.21× = 47,700 / 14,868. |
>
> **The 80 GB/GPU KV budget (B200 TP8 example):** 180 GB HBM − ~54 GB weights (NVFP4, 433 GB ÷ 8 ranks) − ~15 GB activations/CUDA-graph pools ≈ 111 GB free, × ~0.7 `free_gpu_memory_fraction` ≈ **80 GB**. (Same recipe for GB300 TP4: 288 − 108 − 15 ≈ 165 GB × 0.7 ≈ 115 GB.)
>
> **Per-sequence ledger at 128k ctx (131,072 tokens):**
>
> ```
> Baseline (all on GPU):        47,700 × 131,072           = 6.25 GB/seq → 80 / 6.25 → B ≈ 13
>
> After offload:
>   → CPU pinned host:          32,832 × 131,072           = 4.30 GB/seq  (57 shared layers — the context KV that moves)
>   → GPU, scales with ctx:     14,868 × 131,072           = 1.95 GB/seq  (full layers' latent + indexer k-cache)
>   → GPU, fixed hot set:       57 × 4,096 tok × 576 B     = 0.13 GB/seq
>   GPU total                                              = 2.08 GB/seq → 80 / 2.08 → B ≈ 38   (3.0×)
> ```
>
> Check: 1.95 + 4.30 = 6.25 — every byte accounted for.
>
> **Other rows/columns:** same formulas with ctx = 32k / 64k / 198k — the two ctx-scaling terms grow with ctx while the 0.13 GB hot set stays constant (which is why the uplift approaches the 3.21× ceiling as ctx grows). The "host pinned RAM/GPU" column = 32,832 × ctx × uplifted-B, e.g. 4.30 GB × 38 ≈ 165 GB at 128k.

</details>

Net decode throughput ≈ uplift × (TPOT_old/TPOT_new); with TPOT growing 10–25% at the larger batch, expect **~2–2.5× tokens/s/GPU** in the long-context regime. Three ceilings apply: the 3.21× KV ratio, the transfer-hiding limits above, and **host RAM** (140–240 GB pinned per GPU — fits GB300's 240 GB/GPU and typical 2 TB x86 hosts, but it's a real third constraint, and it grows toward ~1 TB/GPU if anyone dreams of 1M-context batches).

Honest caveat: every current GLM-5.2 test config runs `max_seq_len` 8–10k, where KV isn't the binding constraint and this idea buys nothing. The payoff is entirely the long-context/agentic regime — which is exactly the study's corpus (up to 642k ISL).

### 4. Implementation reality — is the "mid-iteration KVCM redesign" fear justified?

Partially. Nothing today moves KV mid-iteration: offload/onboard triggers only at block eviction and request admission (`WindowBlockManager::getFreeBlock`, `onboardBlock`), and a block spans *all* pools of its window manager at once. But the redesign is bounded, because the hard parts have precedents:

- **Per-layer-subset pools already exist**: the indexer k-cache mask (#16558, `kvCacheManager.cpp:1069-1124`) allocates a pool over only the 21 full-indexer layers. A `hostResidentLayerMask` for the 57 shared layers follows the same pattern — the host side is a plain always-pinned mirror pool with no block state machine, not a rework of the existing offload machinery.
- **The attention kernel needs no change.** Decode sparse MLA (`flash_mla_sparse_fwd`) already gathers rows from a flat pool by per-token indices, produced by `convert_req_index_to_global`. For shared layers, swap that translation for a working-set slot lookup: hits map to slots, misses go to a device-driven gather kernel that reads pinned host memory (the V2 manager's `batchedCopy` kernel is 90% of this, but must take its task list from device memory instead of `__grid_constant__` args to be CUDA-graph-replayable with data-dependent indices — that's the one genuinely new kernel).
- **Per-layer multi-stream + events inside captured graphs is established** (DSv4's indexer overlap, `maybe_execute_in_parallel`), and a capture-safe pinned-H2D precedent exists in `moeOp.cpp`.
- **Disagg is the natural first target**: gen-side GB200/GB300 configs already exist, and NIXL can land shared-layer KV directly into the host pool at ctx→gen transfer — no demotion step at all. Aggregated serving needs a one-time background demotion at the prefill→decode boundary (~4.3 GB per 128k seq, amortized over thousands of decode steps).

Rough estimate: a flag-gated, decode-only, GB300-disagg-first prototype is ~2 engineer-months; productionizing (eviction under pressure, all graph families, agg-serving demotion, fallbacks) is the larger tail.

### 4.5 CUDA-graph compatibility

**Compatible — but only with a device-driven fetch design; every host-orchestrated variant is ruled out.** This is the deeper reason §4 insists on a new gather kernel rather than reusing the existing onboard machinery.

The constraint: TRT-LLM decode captures the **entire model forward** in a CUDA graph (`CUDAGraphRunner`); at replay only `input_ids`/`position_ids` are copied into static buffers — everything inside must be address-stable and replayable with zero host involvement. All GLM-5.2 production configs run with CUDA graphs on, and the fetch addresses depend on top-k values computed **on device, inside the graph, each step**:

| Implementation | Graph-compatible? | Why |
|---|---|---|
| Host-orchestrated `cudaMemcpyAsync` per step | ❌ | Memcpy nodes bake src/dst addresses at capture; the host would need the miss list → D2H sync mid-step → breaks replay |
| V2's `batchedCopy` as-is | ❌ | Task list passed as `__grid_constant__` kernel args — addresses frozen at capture |
| CPU-side gather into staging + fixed-address memcpy | ❌ | Host can't see the device-computed indices without a mid-graph sync |
| **Gather kernel reading pinned host memory (zero-copy/UVA), task list in device memory** | ✅ | A kernel node's *memory access pattern* may be fully data-dependent; only buffer base addresses must be stable |

The workable design: GPU threads read the miss list from a fixed device buffer, compute source offsets into the pinned host pool (fixed base address for process lifetime), and load directly over PCIe/C2C. On GB200/GB300 this is especially natural — NVLink-C2C is cache-coherent, so host memory is just memory to GPU loads. The graph captures it as an ordinary kernel node. New-token writeback to the host pool is the same pattern in reverse (device-side scatter to pinned memory).

**Every supporting pattern is already proven in-tree:**

- **Data-dependent indices inside graphs**: the DSA indexer already computes top-k *inside* the captured graph, into graph-stable buffers (`topk_indices_buffer`, `shared_topk_indices` via `get_empty(..., capture_graph=...)`). The miss-resolve kernel is the same pattern.
- **Per-layer multi-stream + events inside capture**: capture runs under `with_multi_stream(True)`, and DSv4 already overlaps its indexer on aux streams with per-layer events inside captured graphs — the fetch-done → attention-start dependency is identical.
- **Pinned H2D inside a captured graph**: precedent in `moeOp.cpp` (MoE-LoRA slot tables), which documents the rules — pinned fixed-address source, persistent destination, no reallocation between captures.
- **Variable miss counts**: graph structure is fixed, so kernels launch worst-case-sized (grid-stride, early-exit on a device counter). Transfer *duration* varies per replay — graphs don't care; a tail-case overrun shows up as TPOT jitter via the event wait, not a correctness issue.
- **Graph families**: DSA already captures separate families for short vs. long sequences (short seqs skip the indexer) — the offload path only needs to exist in the long-seq family.

**Conditions / sharp edges:**

1. **The host pool must be pre-allocated at startup and never reallocated** — growth invalidates the captured base address and forces recapture. Same discipline as the existing secondary pool, but sized up-front.
2. **Mixed batches within one graph** (the fiddliest design point): a captured graph executes one code path for all requests. If some requests in a batch are offloaded and some GPU-resident (fallback mode), the index space must unify both — flat offsets resolvable into either the working-set pool or the GPU pool — or the scheduler must segregate them into different graph batches.
3. **Per-request mode switches** (fallback, admission, prefill→decode demotion) happen outside the graph, at the prepare/scheduling boundary — like block-table updates today.
4. All slot tables and scratch live in the existing `cuda_graph_buffers` fixed-address scheme, sized for each graph bucket's max batch; CUDA-graph padding slots resolve to a dummy slot with zero misses.

Summary: incompatible with any design that puts the host in the per-step loop (which is what the V1/V2 onboard machinery is); fully compatible with the device-driven gather design, with the graph-related engineering risk concentrated in mixed-batch handling (point 2), not in capture/replay itself.

### 5. Key risks

1. **The miss rate at working-set sizes > topk is unmeasured** — the single load-bearing unknown. Jaccard gap-1 only proves 24%. *(Update: §6.1 simulation bounds it — W=K ≈ 21–24% is model-independent and sufficient for GB-class; W=4×K is anywhere from 0.6% to 19% depending on re-selection behavior, which only the raw dumps can resolve.)*
2. **Tail behavior**: p90 scatter is ~2× median → some steps overrun the fetch window → TPOT jitter (stall, not corruption). Needs a per-layer budget + per-request fallback to GPU-resident mode.
3. Host memory bandwidth contention (Grace LPDDR shared with CPU work; x86 NUMA placement), and 100–200 GB pinned registration at startup.
4. Under attention-DP, per-rank batches keep fetches rank-local (good); pure TP replicates the latent KV per rank and multiplies host-read traffic — prefer attention-DP deployments (already GLM-5.2's high-throughput config).

### 6. Recommended next steps (cheap → expensive)

1. **Cache simulation on Allen's existing per-step index dumps**: hit rate vs. working-set size (1×, 2×, 4× topk), with recency pinning (last-256 tokens are selected ~75% of steps — pin them). One day of work; turns the biggest unknown into a number.

   **Status (2026-08-11): done in bounded form — the definitive number still needs the raw dumps.** The raw per-step rows are *not* on the scratch share (`/home/scratch.alyuan_other/**` holds only the aggregated stats and plots; the rows behind `rows_per_layer=100,642` presumably live on the GB300 cluster). What was built instead: `glm_kv_cache_sim.py` (next to this file) — an **exact-LRU working-set simulator** ready to replay the real dumps via `--real` the moment the rows are available, plus a calibrated synthetic mode used to bound the answer today. Results:

   | W (slots/seq/group) | pool model (optimistic) | fresh model (pessimistic) |
   |---|---|---|
   | 1×K = 2048 | 20.6% miss (worst layer 31%) | 23.8% (worst 35%) |
   | 1.5×K = 3072 | 16.5% | 22.4% |
   | 2×K = 4096 | 13.2% (8.9 MB/seq/step) | 21.5% (14.4 MB/seq/step) |
   | 4×K = 8192 | **0.6%** (0.4 MB/seq/step) | **19.2%** (12.9 MB/seq/step) |

   Three takeaways:
   - **W=K miss ≈ 21–24% is model-independent** (both endpoints reproduce the measured gap-1 Jaccard) — the §2 feasibility table's "miss 24%" column is robust, so GB-class feasibility does not depend on the open question.
   - **The value of W&gt;K is provably undetermined by the published aggregates**: a "recurring warm pool" model and a "fresh-drift" model both calibrate to J(1..4) within ±0.03–0.05 yet give 0.6% vs 19.2% at W=4×K. Pairwise Jaccards cannot distinguish re-selection from novelty — only the raw trajectories can. Plan with 24%; treat anything better as upside.
   - **Recency pinning (last-256) buys only ~1–1.5 points** at W=K and ~nothing at larger W (recent tokens are near-permanent hits anyway); keep it in the real design as a *guarantee* (newest tokens never fetched, no writeback race), not as a perf lever.

   One nuance for the §2 window math: each group inherits the churn of *its own* full-indexer layer, so a high-churn group (J≈0.47 → ~35% miss) pays worst-case on all 3 of its shared layers — per-group fetch budgeting should use per-group churn with p90 headroom, mitigated by the 2–3× window slack of group layers 3–4.

   <details>
   <summary><i>Method (click to expand)</i></summary>

   > Exact LRU over per-step top-K sets, per (request, layer): current selections always resident; misses = selections absent from the working set (decode-born tokens enter resident — their KV is produced on GPU; prefill tokens' first selection is a compulsory miss); eviction by least-recently-selected. Synthetic trajectories per layer: sticky core + geometric churn + recency zone, with per-layer core fraction solved so gap-1 Jaccard matches `stats_main.json` **exactly per layer** (21 values, 0.474–0.778), churn retention and core drift tuned so mean J(1..4) ≈ (0.61, 0.54, 0.51, 0.51) vs. targets (0.61, 0.58, 0.51, 0.49) read from `plots/4_temporal_jaccard.png`; U-shaped positional prior and last-64 ≈ 90% selection per `edge_selection`. K=2048, ctx=96k, 512 decode steps, steady state after step 32. The two endpoint models differ only in where churn replacements come from: a warm pool of 4×K distinct positions (re-selection) vs. always-new positions (drift). Run: `python3 glm_kv_cache_sim.py --pc 0.30 --drift 0.004 [--pin 256]`.
   >
   > **The ask to Allen**: the per-step rows behind `rows_per_layer=100,642` — schema `(request_id, layer, step, ctx_len, topk[2048])` in any of npz/parquet — and ideally ≥64 *consecutive* steps per request (gaps 1–4 in the study prove consecutive trajectories were captured). The `--real` adapter is a ~20-line loader once the format is known.

   </details>
2. **Microbenchmark the device-driven host gather**: ~1.7 KB granularity (3 layers × 576 B, token-major host layout) random reads over C2C and PCIe Gen5, inside a captured graph. Validates the effective-bandwidth assumptions.

   **Status (2026-08-11): done for PCIe (measured on H100 / Gen4 x16; Gen5 by link-ratio extrapolation). C2C still needs a GB200/300 node.** Implemented `glm_kv_gather_bench.py` (next to this file): a warp-per-row CUDA kernel gathering scattered rows **directly from pinned host memory with the index list in device memory** — the exact §4.5 mechanism. Results (vs. 28.4 GB/s `cudaMemcpyAsync` peak on this Gen4 x16 link):

   | random-row layout | gather GB/s (stream) | (graph replay) | % of memcpy peak |
   |---|---|---|---|
   | 576 B (latent, layer-major) | 17.1–19.8 | same | 60–70% |
   | **1728 B (3-layer token-major)** | **20.8–21.3** | **same** | **74–75%** |
   | 18 KB (32-token block) | 25.6–26.1 | same | 90–92% |

   Findings:
   - **The §2 bandwidth assumption is validated**: 74% efficiency × Gen5's ~56 GB/s memcpy peak ≈ **41 GB/s effective** for 1728 B token-major rows (assumed: 40). Efficiency is flat from 3.5 MB to 226 MB per call, so both the per-layer-window and sustained regimes see the same number.
   - **CUDA-graph replay is free**: captured-graph bandwidth ≡ stream-launch bandwidth in every cell — empirically confirms §4.5's capture/replay story for data-dependent host gathers.
   - **Token-major host layout is worth ~+25%** over per-layer 576 B rows (74% vs 60–70%). Block granularity is the most efficient per byte (92%) but moves 7–10× the bytes — confirmed worse end-to-end.
   - **Grid-size sweep (the design rule for the real kernel)**: 16 thread blocks (~12% of H100's SMs) already saturate the link — 20.8 GB/s vs 18.8 at 4096 blocks — and concurrent GEMM throughput degrades only **1.15×** while the gather is in flight (vs 1.76× with a full-GPU launch). The production gather should be a fixed-small-grid "SM DMA agent", like comm kernels.
   - Caveats: this node is PCIe **Gen4** (absolute numbers ≈ half of Gen5); its driver rejects single pinned allocations ≥4 GiB (pool must be allocated in chunks — verify on target hosts); C2C behavior (coherent loads from Grace LPDDR, expected several-hundred GB/s) is the remaining measurement, needing a GB200/GB300 node.
3. **One-group prototype** (3 shared layers host-resident) on a GB300 setup at 64–128k ISL; measure TPOT delta at fixed batch, then batch uplift at fixed TPOT. **An aggregated setup (TP4/EP4, as in Allen's study) is the recommended prototype vehicle** — the decode path is identical to disagg-gen, it's apples-to-apples with the §6.1 index statistics, and a starting config exists (`tests/scripts/perf-sanity/aggregated/glm5_fp4_grace_blackwell.yaml`). Handle the prefill→decode transition by **dual-writing** the group's shared-layer KV to both GPU and host pools during prefill (the prototype measures decode overhead, not memory reclamation, so keeping both copies is fine); measure steady-state decode after prefill drains. Deferred by this choice, knowingly: NIXL→host-pool landing (disagg-specific plumbing) and batch-uplift-at-fixed-TPOT (needs all 19 groups offloaded, in any setup). Bonus measurement available only in agg: chunked-prefill chunks competing with decode gathers for SMs and link bandwidth.

Bottom line for the thread: the China team's skepticism was justified against the naive design (block-granular, no cache — that version genuinely doesn't fit in the compute window), but Allen's own data contains the evidence that a delta-fetching variant clears the bar with room to spare on GB-class hardware, in exactly the long-context regime GLM-5.2 is built for.
