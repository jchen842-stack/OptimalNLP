"""Run the OptimalCE optimal search on REAL SNLI token concept masks.

This is the synthetic sweep's measurement core (`synthetic_overlap_sweep.HeapProbe` /
`StubConfig` / quantity helpers) pointed at real concept masks built by
`real_token_masks.py`, so peak-frontier / visited / time / IoU numbers are directly
comparable to `results/overlap_sweep.csv` and `results/beam_sweep.csv`.

The concept masks are real. The **neuron is a proxy** — an OR of three real concepts plus
label noise, mirroring `synthetic_overlap_sweep.make_neuron`. That is deliberate and it
bounds the claim: this measures how the SEARCH behaves under real token concept structure
(which is what the frontier explosion is about — the combinatorics live on the concept
side), NOT what real neurons in a trained model encode. Real activations require the
encoder checkpoint and are a separate step.

Usage::

    python src/real_token_search.py --arms tag all --lengths 3 4
"""

import argparse
import csv
import os
import sys

import numpy as np

# Upstream `compositional` package (patched with MAX_FRONTIER_SIZE) and our synthetic harness.
UPSTREAM = os.environ.get("OPTIMALCE_UPSTREAM", os.path.expanduser("~/projects/optimalce"))
sys.path.insert(0, UPSTREAM)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importing the synthetic harness installs its meta-path stub finder for the vision-only
# deps (detectron2, cityscapesscripts) as a side effect, which is what lets `compositional`
# import on a machine that has neither.
from synthetic_overlap_sweep import (  # noqa: E402
    HeapProbe, StubConfig, compute_disjoint_info, compute_quantities, _Halt,
)
import real_token_masks as rtm  # noqa: E402

import heapq  # noqa: E402
import time  # noqa: E402
from compositional import optimal  # noqa: E402

ARMS = {
    # Single-valued categories are disjoint by construction: a token has exactly one POS
    # tag, one dep label, one lemma. `tag`/`dep` are closed-class and saturate (~23 tags
    # meet min_support), so they cannot supply a K-matched control at K=30+. `lemma` is
    # open-class and disjoint, so it scales to any K -- use it as the high-K control.
    "tag": ("tag-only (disjoint control)", ["tag"]),
    "dep": ("dep-only", ["dep"]),
    "lemma": ("lemma-only (disjoint control, scales with K)", ["lemma"]),
    "tagdep": ("tag+dep", ["tag", "dep"]),
    "all": ("all categories (token regime)", rtm.CATEGORIES),
}


def load_real_neurons(path, n_units, rng, dmin=0.15, dmax=0.85):
    """Pick `n_units` real unit masks from a `real_activations.py` dump.

    Units are filtered to a sensible density band — a unit that fires on ~everything or
    ~nothing has a degenerate optimum and would measure thresholding, not search behaviour.
    Selection is seeded so runs are reproducible.
    """
    z = np.load(path)
    acts, density = z["acts"], z["density"]
    eligible = np.where((density >= dmin) & (density <= dmax))[0]
    if len(eligible) == 0:
        raise SystemExit(f"no units in density band [{dmin}, {dmax}] in {path}")
    pick = rng.choice(eligible, size=min(n_units, len(eligible)), replace=False)
    return [(int(u), acts[u].astype(bool)) for u in sorted(pick)], bool(z["untrained"])


def make_proxy_neuron(dense, rng, noise, n_target=3):
    """Proxy neuron: OR of three real concepts + label noise.

    Concepts are chosen from the middle of the frequency order rather than the top, so the
    target is neither near-empty nor near-universal (a trivially coverable target would
    terminate the search early and hide the frontier behaviour we are measuring).
    """
    K, M = dense.shape
    coverage = dense.sum(axis=1)
    order = np.argsort(-coverage)
    mid = order[K // 3: K // 3 + n_target] if K >= n_target * 2 else order[:n_target]
    target = np.zeros(M, dtype=bool)
    for k in mid:
        target |= dense[k]
    flip = rng.random(M) < noise
    return np.where(flip, ~target, target), [int(k) for k in mid]


def run_one(dense, neuron_bits, length, cap, time_budget, beam_cap):
    """One search run; mirrors synthetic_overlap_sweep.run_level but takes masks directly."""
    import scipy.sparse as sparse
    import torch

    K, M = dense.shape
    masks = [sparse.csr_matrix(dense[c].reshape(1, M)) for c in range(K)]
    common, unique, uncoverable, _ = compute_quantities(dense, M)
    disjoint_info = compute_disjoint_info(dense, K)
    bitmaps = torch.from_numpy(neuron_bits.reshape(1, M))

    probe = HeapProbe(cap, time_budget=time_budget)
    optimal.heapq = probe
    optimal.MAX_FRONTIER_SIZE = beam_cap
    cfg = StubConfig(length, M)

    t0 = time.time()
    halt = ""
    # optimal.py streams a per-pop progress line; silence it so the sweep output stays readable.
    devnull = open(os.devnull, "w")
    saved_stdout = sys.stdout
    try:
        sys.stdout = devnull
        _, best_iou, visited, expanded, estimated = optimal.compute_optimal_explanations(
            bitmaps=bitmaps, masks=masks, masks_info=(common, unique, uncoverable),
            disjoint_info=disjoint_info, config=cfg,
        )
    except _Halt as h:
        halt = h.reason
        best_iou, visited, expanded, estimated = float("nan"), -1, -1, -1
    finally:
        sys.stdout = saved_stdout
        devnull.close()
        optimal.heapq = heapq
        optimal.MAX_FRONTIER_SIZE = None
        dt = time.time() - t0

    return {
        "peak_frontier": probe.peak, "halted": halt or "no", "visited": visited,
        "expanded": expanded, "estimated": estimated,
        "best_iou": round(best_iou, 4) if best_iou == best_iou else None,
        "time_s": round(dt, 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feats", default=os.path.expanduser(
        "~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats"))
    ap.add_argument("--max_sents", type=int, default=200)
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--min_support", type=int, default=5)
    ap.add_argument("--arms", nargs="+", default=["tag", "all"], choices=list(ARMS))
    ap.add_argument("--lengths", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--beam_list", nargs="+", default=["none"],
                    help="beam widths; 'none' = exact optimal")
    ap.add_argument("--cap", type=int, default=200000)
    ap.add_argument("--time_budget", type=float, default=60.0)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neuron", choices=["proxy", "real"], default="proxy",
                    help="proxy = OR of 3 concepts + noise; real = units from real_activations.py")
    ap.add_argument("--acts", default="results/real_activations.npz")
    ap.add_argument("--units", type=int, default=5, help="how many real units to run")
    ap.add_argument("--out", default="results/real_token_search.csv")
    args = ap.parse_args()

    tokens = rtm.load_tokens(args.feats, args.max_sents)
    print(f"[real-search] {len(tokens)} tokens | K={args.K} cap={args.cap} "
          f"budget={args.time_budget}s\n", flush=True)

    rows = []
    for arm in args.arms:
        name, cats = ARMS[arm]
        concepts = rtm.select_concepts(tokens, cats, args.K, args.min_support)
        dense = rtm.build_dense(tokens, concepts)
        diag = rtm.diagnostics(dense)
        if args.neuron == "proxy":
            bits, target_ks = make_proxy_neuron(
                dense, np.random.default_rng(args.seed + 999), args.noise)
            neurons = [("proxy", bits)]
            label = f"OR{[f'{concepts[k][0]}={concepts[k][1]}' for k in target_ks]}"
        else:
            picked, untrained = load_real_neurons(
                args.acts, args.units, np.random.default_rng(args.seed + 999))
            if len(picked[0][1]) != len(tokens):
                raise SystemExit(
                    f"activation/mask token mismatch: {len(picked[0][1])} vs {len(tokens)} "
                    "-- regenerate the .npz with the same --max_sents")
            neurons = [(f"unit{u}", b) for u, b in picked]
            label = f"{len(neurons)} real units ({'untrained' if untrained else 'trained'})"

        print(f"=== {name} === overlap={diag['mean_overlap']} "
              f"common_frac={diag['common_frac']} disjoint_pairs={diag['disjoint_pairs']} "
              f"| neuron={label}\n", flush=True)

        for nid, neuron_bits in neurons:
            for length in args.lengths:
                for bw in args.beam_list:
                    beam_cap = None if str(bw).lower() == "none" else int(bw)
                    print(f"--- {arm} {nid} length={length} beam={bw} "
                          f"density={neuron_bits.mean():.3f} ---", flush=True)
                    res = run_one(dense, neuron_bits, length, args.cap,
                                  args.time_budget, beam_cap)
                    row = {"arm": name, "categories": "+".join(cats), "neuron": nid,
                           "density": round(float(neuron_bits.mean()), 3),
                           "length": length, "beam": bw,
                           "mean_overlap": diag["mean_overlap"],
                           "common_frac": diag["common_frac"],
                           "disjoint_pairs": diag["disjoint_pairs"], **res}
                    rows.append(row)
                    print(row, "\n", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")

    cols = ["arm", "neuron", "density", "length", "beam", "mean_overlap", "disjoint_pairs",
            "peak_frontier", "visited", "best_iou", "time_s", "halted"]
    print("\n=== SUMMARY ===")
    print(" ".join(f"{c:>14}" for c in cols))
    for r in rows:
        print(" ".join(f"{str(r.get(c, ''))[:14]:>14}" for c in cols))


if __name__ == "__main__":
    raise SystemExit(main())
