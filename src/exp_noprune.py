"""Experiment C — a provably-no-prune configuration of optimal.py, on all 27 pairs.

The 2/27 exact-search misses are real (oracle check 10b) but the cause is unidentified: the
disjoint-branch hypothesis was falsified (see the RETRACTION section in METHOD_NOTES). This
test does not guess at a cause. It turns pruning OFF entirely and asks a binary question:

    with nothing pruned, does the search reach the brute-force optimum on all 27?

    YES -> the fault is entirely in the ceiling estimate, and this config is "exact" for real.
    NO  -> the fault is NOT the ceiling. It is in expansion, dedup, or the grammar walk,
           which is a larger finding than the current one.

## How the config is built

`optimal.py` is NOT edited. Its pinned source is read, a small set of exact textual
substitutions is applied, and the result is exec'd as a separate module `optimal_noprune`.
Every substitution asserts it matched exactly the expected number of times, so a silent
no-op is impossible. The substitutions are printed at runtime.

Disabled, all together -- disabling any one alone leaves a live prune path:

  1. `reduce_frontier` -> returns the frontier unchanged.
  2. `minimum_threshold` in `estimate_iou_frontier` -> never allowed to rise off
     `global_min_threshold`.
  3. the `if node_path_max_iou > 0` gate in `estimate_iou_frontier` -> always taken. A node
     whose estimate is 0 or negative is dropped otherwise, so leaving this in means the
     config is not no-prune.
  4. (BEYOND the three) `minimum_threshold` also rises inside `perform_search` itself
     (:742, :793, :827, :865 at the pinned SHA) and drives the node-skip
     `if -e_node < minimum_threshold: continue` at :697. That skip is a prune. Pinning only
     the `estimate_iou_frontier` copy would leave it live, so the incumbent-driven skip is
     disabled too and the threshold is held at 0.0 throughout.

What is deliberately LEFT ALONE, so that a failure localises: `expand_node`, the
`recent_nodes` dedup, `apply_distributive_property`, and the heuristic estimators themselves.

Usage::

    python src/exp_noprune.py --run
"""

import argparse
import csv
import os
import re
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

OUT = os.path.join(REPO, "results", "noprune_L3.csv")

from exp_beam_width import PAIRS, K, LENGTH, MAX_SENTS, MIN_SUPPORT, CAP  # noqa: E402

TIME_BUDGET = 1500.0

# =====================================================================================
# PRE-REGISTERED PREDICTIONS — committed before the no-prune config was run once.
# =====================================================================================
PREDICTIONS = """
N1  ALL 27 MATCH THE BRUTE-FORCE ORACLE EXACTLY.
    With pruning fully off, every one of the 27 pairs returns the true in-grammar length-3
    optimum, to within 1e-12, including the two that miss under the published config
    (trained a=0.2 unit88, trained a=0.05 unit86).
    SUPPORTED only if 27/27 match.
    If ANY pair still misses, the fault is NOT the ceiling estimate -- it is in expansion,
    dedup, or the grammar walk. That is a larger finding than the current one and the run
    stops there and reports it.

N2  RUNTIME RISES BUT STAYS UNDER THE CAP.
    All 27 complete without hitting the 1500s soft cap, and the median runtime is higher
    than the published config's. Reported as a factor.
    A factor near 1.0x would mean the substitutions did not take effect and the whole run
    is void -- which is why each substitution also asserts its own match count.
    SUPPORTED if 0 halts AND median(noprune) > median(published).

N3  CONTROL — NO PAIR GETS WORSE.
    Removing pruning cannot cost IoU. No pair returns a lower IoU than the published run.
    This could fail: if it does, the config is not a pure relaxation and N1 is not
    interpretable as "the ceiling was the whole fault".
    SUPPORTED if min over pairs of (noprune_iou - published_iou) >= -1e-12.

N4  CONTROL — THE SEARCH ACTUALLY DID MORE WORK.
    Visited-node count rises on every one of the 27 pairs. A pair where visited is unchanged
    was not affected by the relaxation at all, which would make its N1 result uninformative.
    SUPPORTED if visited_noprune > visited_published on all 27.
"""

# (pattern, replacement, expected_count) -- applied to the pinned optimal.py source.
SUBS = [
    # 1. reduce_frontier returns the frontier unchanged.
    (r"    reduced_frontier = \[\]\n"
     r"    for node in frontier:\n"
     r"        iou = node\[0\]\n"
     r"        if -iou >= threshold:\n"
     r"            reduced_frontier\.append\(node\)\n"
     r"    heapq\.heapify\(reduced_frontier\)\n"
     r"    return reduced_frontier",
     "    # NO-PRUNE: threshold prune disabled.\n"
     "    reduced_frontier = list(frontier)\n"
     "    heapq.heapify(reduced_frontier)\n"
     "    return reduced_frontier", 1),
    # 2. minimum_threshold never rises inside estimate_iou_frontier.
    (r"        if min_score > minimum_threshold:\n"
     r"            minimum_threshold = min_score",
     "        if False:  # NO-PRUNE: minimum_threshold pinned\n"
     "            minimum_threshold = min_score", 1),
    # 3. the >0 gate always taken.
    (r"            if node_path_max_iou > 0:",
     "            if True:  # NO-PRUNE: >0 gate disabled", 1),
    # 3b. that gate's companion sanity raise can now fire legitimately; disable it.
    (r"                if node_path_max_iou < minimum_threshold:\n"
     r"                    raise ValueError\(",
     "                if False:  # NO-PRUNE: invariant no longer applies\n"
     "                    raise ValueError(", 1),
    # 4. the incumbent-driven node skip in perform_search.
    (r"        if -e_node < minimum_threshold:\n"
     r"            # Unuseful node, skip it",
     "        if False:  # NO-PRUNE: incumbent skip disabled\n"
     "            # Unuseful node, skip it", 1),
    # 4b. hold the incumbent threshold at 0 everywhere in perform_search.
    (r"                    minimum_threshold = new_min\n",
     "                    minimum_threshold = 0.0  # NO-PRUNE\n", 1),
    (r"                        minimum_threshold = label_iou\n",
     "                        minimum_threshold = 0.0  # NO-PRUNE\n", 1),
    (r"                                minimum_threshold = ancestor_iou\n",
     "                                minimum_threshold = 0.0  # NO-PRUNE\n", 1),
]


def build_noprune_module():
    """Read pinned optimal.py, apply the substitutions, exec as `optimal_noprune`."""
    # Build from the PINNED CLEAN checkout, not the patched working copy: our patch rewrites
    # reduce_frontier's return, and starting from clean keeps `_apply_beam_cap` out of the
    # picture entirely, so this config is upstream-plus-no-prune and nothing else.
    clean = os.path.join(REPO, ".upstream-clean", "compositional", "optimal.py")
    upstream = os.environ.get("OPTIMALCE_UPSTREAM", os.path.expanduser("~/projects/optimalce"))
    src_path = clean if os.path.exists(clean) else os.path.join(
        upstream, "compositional", "optimal.py")
    src = open(src_path).read()
    print(f"[no-prune] source: {src_path}")
    for pat, rep, want in SUBS:
        src, n = re.subn(pat, rep, src)
        head = rep.strip().splitlines()[0][:64]
        assert n == want, f"substitution matched {n}x, expected {want}x: {head}"
        print(f"  applied {n}x: {head}")
    mod = types.ModuleType("optimal_noprune")
    mod.__dict__["__name__"] = "optimal_noprune"
    mod.__dict__["__file__"] = src_path + " (NO-PRUNE)"
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    sys.modules["optimal_noprune"] = mod
    return mod


def grammar_max_all(dense, neurons):
    Kc = dense.shape[0]
    N = np.array(neurons)
    nsz = N.sum(1).astype(np.int64)
    best = np.zeros(len(neurons), dtype=np.float64)
    leaves = [dense[i] for i in range(Kc)]

    def moves(m):
        for j in range(Kc):
            yield m | leaves[j]
            yield m & leaves[j]
            yield m & ~leaves[j]

    for i in range(Kc):
        for m2 in moves(leaves[i]):
            for m3 in moves(m2):
                inter = (N & m3).sum(1).astype(np.int64)
                size = np.int64(m3.sum())
                best = np.maximum(best, inter / np.maximum(size + nsz - inter, 1))
    return best


def run_with(module, dense, neuron_bits, concepts):
    """One search run against `module` (either upstream optimal or the no-prune build)."""
    import heapq
    import scipy.sparse as sparse
    import torch
    from synthetic_overlap_sweep import (HeapProbe, compute_disjoint_info,
                                         compute_quantities, StubConfig, _Halt)
    import real_token_search as rts

    Kc, M = dense.shape
    masks = [sparse.csr_matrix(dense[c].reshape(1, M)) for c in range(Kc)]
    common, unique, uncoverable, _ = compute_quantities(dense, M)
    disjoint_info = compute_disjoint_info(dense, Kc)
    bitmaps = torch.from_numpy(neuron_bits.reshape(1, M))

    probe = HeapProbe(CAP, time_budget=TIME_BUDGET)
    module.heapq = probe
    module.MAX_FRONTIER_SIZE = None
    cfg = StubConfig(LENGTH, M)
    t0, halt, best_label = time.time(), "", None
    devnull = open(os.devnull, "w")
    saved = sys.stdout
    try:
        sys.stdout = devnull
        best_label, best_iou, visited, expanded, estimated = \
            module.compute_optimal_explanations(
                bitmaps=bitmaps, masks=masks, masks_info=(common, unique, uncoverable),
                disjoint_info=disjoint_info, config=cfg)
    except _Halt as h:
        halt, best_iou, visited, expanded = h.reason, float("nan"), -1, -1
    finally:
        sys.stdout = saved
        devnull.close()
        module.heapq = heapq
        dt = time.time() - t0

    iou, formula = float("nan"), None
    if best_label is not None:
        m = rts.eval_formula(best_label, dense)
        iou = int((m & neuron_bits).sum()) / int((m | neuron_bits).sum())
        formula = rts.render(best_label, concepts)
    return {"iou": iou, "formula": formula, "visited": visited, "expanded": expanded,
            "time_s": round(dt, 2), "peak": probe.peak, "halted": halt or "no"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.parse_args()
    print(PREDICTIONS, flush=True)

    sys.path.insert(0, os.environ.get("OPTIMALCE_UPSTREAM",
                                      os.path.expanduser("~/projects/optimalce")))
    import real_token_masks as rtm
    import real_token_search as rts
    from compositional import optimal as upstream_optimal

    noprune = build_noprune_module()

    feats = os.path.expanduser(
        "~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats")
    tokens = rtm.load_tokens(feats, MAX_SENTS)
    _, cats = rts.ARMS["all"]
    concepts = rtm.select_concepts(tokens, cats, K, MIN_SUPPORT)
    dense = rtm.build_dense(tokens, concepts)

    keys, neurons = [], []
    for arm, alpha, units in PAIRS:
        z = np.load(os.path.join(REPO, "results", f"acts2k_{arm}_a{alpha}.npz"))
        for u in units:
            keys.append((arm, alpha, u))
            neurons.append(z["acts"][u].astype(bool))
    print(f"\n[no-prune] {len(tokens)} tokens | K={K} length={LENGTH} | {len(keys)} pairs\n",
          flush=True)

    print("in-grammar optima (brute force) ...", flush=True)
    gram = grammar_max_all(dense, neurons)

    rows = []
    for (arm, alpha, u), neuron, g in zip(keys, neurons, gram):
        pub = run_with(upstream_optimal, dense, neuron, concepts)
        npr = run_with(noprune, dense, neuron, concepts)
        miss = g - npr["iou"] > 1e-12
        rows.append({
            "arm": arm, "alpha": alpha, "unit": f"unit{u}",
            "in_grammar_max": repr(g),
            "published_iou": repr(pub["iou"]), "noprune_iou": repr(npr["iou"]),
            "published_missed": int(g - pub["iou"] > 1e-12), "noprune_missed": int(miss),
            "delta_iou": repr(npr["iou"] - pub["iou"]),
            "published_visited": pub["visited"], "noprune_visited": npr["visited"],
            "published_expanded": pub["expanded"], "noprune_expanded": npr["expanded"],
            "published_peak": pub["peak"], "noprune_peak": npr["peak"],
            "published_time_s": pub["time_s"], "noprune_time_s": npr["time_s"],
            "published_halted": pub["halted"], "noprune_halted": npr["halted"],
            "published_formula": pub["formula"], "noprune_formula": npr["formula"],
        })
        print(f"  {arm} a={alpha} unit{u}: true={g:.6f} pub={pub['iou']:.6f} "
              f"nopr={npr['iou']:.6f} {'MISS' if miss else 'ok'}  "
              f"visited {pub['visited']}->{npr['visited']}  "
              f"t {pub['time_s']}->{npr['time_s']}s {npr['halted']}", flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}")

    def med(xs):
        xs = sorted(xs)
        n = len(xs)
        return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

    print("\n--- VERDICTS ---")
    res = []
    still = [r for r in rows if r["noprune_missed"]]
    res.append(_v("N1", not still,
                  f"{len(rows) - len(still)}/{len(rows)} match the brute-force oracle"
                  + ("" if not still else
                     "; STILL MISSING: " + ", ".join(
                         f"{r['arm']} a={r['alpha']} {r['unit']} "
                         f"(true {float(r['in_grammar_max']):.6f} vs "
                         f"{float(r['noprune_iou']):.6f})" for r in still))))

    halts = [r for r in rows if r["noprune_halted"] != "no"]
    mp = med([float(r["published_time_s"]) for r in rows])
    mn = med([float(r["noprune_time_s"]) for r in rows])
    res.append(_v("N2", not halts and mn > mp,
                  f"{len(halts)} halts under the {TIME_BUDGET}s cap; median time "
                  f"{mp:.2f}s -> {mn:.2f}s ({mn / mp:.2f}x)"))

    worst = min(float(r["delta_iou"]) for r in rows)
    nworse = sum(1 for r in rows if float(r["delta_iou"]) < -1e-12)
    res.append(_v("N3", nworse == 0,
                  f"{nworse} pairs worse than published; most negative delta {worst:+.3e}"))

    same = [r for r in rows if int(r["noprune_visited"]) <= int(r["published_visited"])]
    res.append(_v("N4", not same,
                  f"{len(rows) - len(same)}/{len(rows)} pairs did strictly more work"
                  + ("" if not same else
                     "; unchanged on " + ", ".join(f"{r['arm']}/{r['unit']}"
                                                   for r in same[:5]))))

    print(f"\n  scored {sum(res)}/{len(res)} SUPPORTED")
    if still:
        print("\n  *** N1 FAILED WITH PRUNING FULLY OFF.")
        print("      The fault is NOT the ceiling estimate. It is in expansion, dedup, or")
        print("      the grammar walk. This is a LARGER finding. Stop and report before")
        print("      anything else -- do not recompute published figures against this run.")
    return 0


def _v(name, ok, detail):
    print(f"  {name}: {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    print(f"      {detail}")
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
