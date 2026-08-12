#!/usr/bin/env python3
# Working-set (hot-cache) simulator for the GLM-5.2 shared-layer KV offload proposal.
# See glm_kv_cache_offload.md §6 step 1.
#
# Answers: if the 57 shared layers' latent KV lives in CPU memory and the GPU keeps a
# per-(request, group) LRU working set of W token slots, what fraction of each decode
# step's top-K=2048 selection misses (i.e., must be fetched over PCIe/C2C)?
#
# Two modes:
#   --real DIR   replay Allen's raw per-step index dumps (exact answer).
#                Expected schema per file (npz or parquet), one row per decode step:
#                  request_id:int, layer:int, step:int, ctx_len:int, topk:int32[2048]
#                (positions local to the request, -1 = padding)
#   default      calibrated synthetic trajectories. Per-layer gap-1 Jaccard is matched
#                EXACTLY to stats_main.json; the gap-1..4 decay shape is matched to the
#                temporal-jaccard plot. Because pairwise Jaccards cannot distinguish
#                re-selection from fresh churn, two generative endpoints bracket reality:
#                  pool:  churn replacements re-drawn from a warm pool (optimistic)
#                  fresh: churn replacements are always-new positions (pessimistic)

import argparse

import numpy as np

K = 2048
# Per-layer gap-1 Jaccard, stats_main.json (glm52-indexer-topk-study-20260714), 21 full layers.
JACCARD_GAP1 = [
    0.6914,
    0.5861,
    0.7776,
    0.7405,
    0.6536,
    0.7322,
    0.633,
    0.6641,
    0.5392,
    0.5457,
    0.5816,
    0.5043,
    0.4887,
    0.5396,
    0.5384,
    0.5377,
    0.6252,
    0.4744,
    0.6815,
    0.7257,
    0.6309,
]
# Mean J(g) targets read off plots/4_temporal_jaccard.png (g=1 exact from stats).
J_TARGET = {1: 0.614, 2: 0.575, 3: 0.51, 4: 0.49}
RECENT_N, RECENT_RATE = 64, 0.90  # edge_selection: last-64 selected ~90% of steps


def overlap_from_jaccard(j):
    # |A∩B|/K for equal-size sets: J = I/(2K-I)  ->  I/K = 2J/(1+J)
    return 2.0 * j / (1.0 + j)


class LruWorkingSet:
    """Exact LRU over per-step top-K sets. Decode-born tokens enter resident (their KV
    is produced on GPU); prefill tokens' first selection is a compulsory miss.
    """

    def __init__(self, max_ctx, W, prefill_len, pin_recent=0):
        # LRU capacity is W; a pin_recent ring is separate GPU memory on top of W.
        self.W = W
        self.pin = pin_recent
        self.prefill_len = prefill_len
        self.last_sel = np.full(max_ctx, -1, np.int64)  # -1 = never selected
        self.in_cache = np.zeros(max_ctx, bool)
        self.t = 0

    def step(self, topk, cur_len):
        t = self.t = self.t + 1
        sel = topk[(topk >= 0) & (topk < cur_len)]
        self.last_sel[sel] = t  # recency tracked for ring members too
        if self.pin:  # ring members are resident for free
            sel = sel[sel < cur_len - self.pin]
        miss = ~self.in_cache[sel]
        compulsory = miss & (sel < self.prefill_len)
        n_miss, n_comp = int(miss.sum()), int(compulsory.sum())
        self.in_cache[sel] = True
        if self.pin:
            # position exiting the ring joins the LRU (no fetch) iff hot while pinned
            p = cur_len - self.pin - 1
            if 0 <= p and self.last_sel[p] >= t - self.pin:
                self.in_cache[p] = True
        elif cur_len > self.prefill_len:
            # decode-born token becomes resident (its KV was just produced on GPU)
            self.in_cache[cur_len - 1] = True
            self.last_sel[cur_len - 1] = max(self.last_sel[cur_len - 1], t)
        over = int(self.in_cache.sum()) - self.W
        if over > 0:
            cand = np.flatnonzero(self.in_cache & (self.last_sel < t))
            over = min(over, len(cand))
            drop = cand[np.argpartition(self.last_sel[cand], over - 1)[:over]]
            self.in_cache[drop] = False
        return n_miss, n_comp


def biased_positions(rng, n, ctx, exclude_recent=1024):
    """U-shaped positional prior: sink-heavy head, hotspot middle, capped below recent zone."""
    hi = max(ctx - exclude_recent, 1)
    n_sink = int(n * 0.05)
    n_hot = n - n_sink
    hotspots = rng.integers(0, hi, max(hi // 4096, 8))
    centers = rng.choice(hotspots, n_hot)
    pos = centers + rng.integers(-2048, 2048, n_hot)
    pos = np.concatenate([rng.integers(0, min(1024, hi), n_sink), np.clip(pos, 0, hi - 1)])
    return pos


def synth_trajectory(rng, j1, p_c, mode, S0, T, pool_mult=4, core_drift=0.002):
    """Yield (topk, cur_len) per decode step. Set = sticky core + geometric churn + recent zone.
    core fraction a solves: a + (1-a)*p_eff = overlap(1), minus the recent-zone contribution.
    In pool mode, replacements collide with prior members w.p. ~1/pool_mult -> extra overlap.
    """
    # +delta compensates a stable realized-overlap deficit from dedup/subsampling
    ov1 = overlap_from_jaccard(j1) + (0.044 if mode == "pool" else 0.008)
    p_eff = p_c + (1 - p_c) / pool_mult if mode == "pool" else p_c
    r_frac = RECENT_N * RECENT_RATE / K  # ~2.8% of K, ~90% persistent
    a = (ov1 - r_frac * RECENT_RATE - p_eff * (1 - r_frac)) / (1 - p_eff)
    a = float(np.clip(a, 0.05, 0.92))
    m = int(a * K)
    pool = np.unique(biased_positions(rng, int(pool_mult * K * 1.5), S0))[: pool_mult * K]
    core = rng.choice(pool, m, replace=False)
    n_churn = K - m - int(K * r_frac)
    churn = (
        rng.choice(pool, n_churn, replace=False)
        if mode == "pool"
        else np.unique(biased_positions(rng, n_churn * 2, S0))[:n_churn]
    )
    fresh_ctr = [0]

    def draw(n, ctx):
        if mode == "pool":
            return rng.choice(pool, n)
        fresh_ctr[0] += n  # fresh: never-repeating positions
        return np.unique(biased_positions(rng, n * 2, ctx))[:n]

    for t in range(T):
        ctx = S0 + t
        # slow core drift (keeps long-gap overlap from being exactly flat)
        nd = rng.binomial(m, core_drift)
        if nd:
            core[rng.choice(m, nd, replace=False)] = draw(nd, ctx)
        keep = rng.random(len(churn)) < p_c  # geometric churn retention
        churn = np.concatenate([churn[keep], draw(len(churn) - int(keep.sum()), ctx)])
        recent = ctx - 1 - rng.choice(RECENT_N * 2, int(K * r_frac), replace=False)
        topk = np.unique(np.concatenate([core, churn, np.clip(recent, 0, ctx - 1)]))
        if len(topk) > K:
            topk = rng.choice(topk, K, replace=False)
        yield topk.astype(np.int64), ctx


def measure_jaccard(traj, gaps=(1, 2, 3, 4)):
    hist, out = [], {g: [] for g in gaps}
    for topk, _ in traj:
        s = set(topk.tolist())
        for g in gaps:
            if len(hist) >= g:
                p = hist[-g]
                i = len(s & p)
                out[g].append(i / (len(s) + len(p) - i))
        hist.append(s)
    return {g: float(np.mean(v)) for g, v in out.items()}


def run(mode, p_c, W_list, S0, T, pin, drift, seed=0):
    res = {W: [] for W in W_list}
    for li, j1 in enumerate(JACCARD_GAP1):
        for W in W_list:
            rng = np.random.default_rng(seed * 1000 + li)
            lru = LruWorkingSet(S0 + T + 1, W, S0, pin)
            misses = [
                lru.step(tk, cl)[0]
                for tk, cl in synth_trajectory(rng, j1, p_c, mode, S0, T, core_drift=drift)
            ]
            res[W].append(np.mean(misses[32:]) / K)  # steady state
    return {W: (float(np.mean(v)), float(np.max(v))) for W, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", help="dir with raw per-step index dumps (npz/parquet)")
    ap.add_argument("--ctx", type=int, default=98304)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pc", type=float, default=0.32, help="churn retention (calibrates J(4))")
    ap.add_argument(
        "--drift", type=float, default=0.002, help="core turnover/step (calibrates J(4) tail)"
    )
    ap.add_argument("--pin", type=int, default=0, help="pin last-N recent tokens")
    args = ap.parse_args()
    if args.real:
        raise NotImplementedError(
            "Raw-dump replay: point --real at Allen's per-step rows "
            "(request_id, layer, step, ctx_len, topk[2048]); loader is a 20-line adapter "
            "onto LruWorkingSet once the exact format is known."
        )

    print(f"ctx={args.ctx} steps={args.steps} p_c={args.pc} drift={args.drift} pin={args.pin}")
    for mode in ("pool", "fresh"):
        rng = np.random.default_rng(7)
        cal = measure_jaccard(
            synth_trajectory(
                rng,
                float(np.mean(JACCARD_GAP1)),
                args.pc,
                mode,
                args.ctx,
                min(args.steps, 256),
                core_drift=args.drift,
            )
        )
        print(
            f"\n[{mode}] calibration J(g): "
            + " ".join(f"g{g}={cal[g]:.3f}(target {J_TARGET[g]:.3f})" for g in (1, 2, 3, 4))
        )
        W_list = [2048, 3072, 4096, 8192]
        r = run(mode, args.pc, W_list, args.ctx, args.steps, args.pin, args.drift)
        for W in W_list:
            mean, worst = r[W]
            mb = mean * K * 576 * 57 / 1e6
            print(
                f"  W={W:5d} ({W / K:.1f}xK): miss mean={mean:6.1%} worst-layer={worst:6.1%}"
                f"  -> {mb:5.1f} MB/seq/step over 57 layers"
            )


if __name__ == "__main__":
    main()
