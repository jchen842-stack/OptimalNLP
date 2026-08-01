"""Falsification test: is the DISJOINT branch responsible for the 2/27 exact-search misses?

Run BEFORE any patch. The root-cause statement written up in experiment 2b says the misses
come from `utils/optimal_utils.estimate_label_quantities` taking its disjoint fork
(pinned upstream 70805299, `utils/optimal_utils.py:270-281`):

    disjoint -> heuristic.estimate_disjoint_label_info(label, left_quantities, right_quantities)
    else     -> heuristic.estimate_label_info(..., neuron_quantities=neuron_quantities)

The disjoint branch does not receive `neuron_quantities`, and
`compositional/optimal_sample_heuristic.py:93` can make it return `None` outright.

The test is one line: force `are_disjoint()` to return False unconditionally, so every node
takes the non-disjoint branch, and re-run exact search on all 27 pairs at length 3.

If the two known misses do NOT recover, the root-cause statement in METHOD_NOTES is WRONG
and nothing gets patched. That is the point of running this first.

Usage::

    python src/exp_disjoint_falsification.py --run
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

OUT = os.path.join(REPO, "results", "disjoint_falsification_L3.csv")

from exp_beam_width import PAIRS, K, LENGTH, MAX_SENTS, MIN_SUPPORT, CAP, TIME_BUDGET  # noqa: E402

# The two recorded misses, pasted from results/oracle_L3_all27.txt.
KNOWN_MISSES = {
    ("trained", "0.2", 88): (0.25454105110196174, 0.2522022213711222),
    ("trained", "0.05", 86): (0.21660649819494585, 0.20679723502304148),
}

# =====================================================================================
# PRE-REGISTERED PREDICTIONS — committed before are_disjoint was forced even once.
# =====================================================================================
PREDICTIONS = """
F1  BOTH KNOWN MISSES RECOVER.
    With are_disjoint() forced False, trained a=0.2 unit88 and trained a=0.05 unit86 both
    reach the true in-grammar optimum (0.25454105110196174 and 0.21660649819494585) to
    within 1e-12.
    This is the load-bearing test. If it fails, the disjoint branch is NOT the cause, the
    root-cause paragraph in METHOD_NOTES is wrong, and no patch is justified.
    SUPPORTED only if BOTH recover.

F2  NO PAIR GETS WORSE.
    No pair returns a LOWER IoU than the committed baseline run. Forcing the non-disjoint
    branch removes a pruning shortcut; it should never cost IoU.
    This can fail. If some pair gets worse, the two branches are not ordered by soundness
    and the fix is not "always take the non-disjoint branch".
    SUPPORTED if min over pairs of (forced_iou - baseline_iou) >= -1e-12.

F3  RUNTIME RISES BUT STAYS UNDER THE CAP.
    Every one of the 27 runs completes without hitting the 1500s soft cap, and the median
    runtime is HIGHER than the baseline median (the shortcut was doing real pruning work).
    Two-sided on purpose: an unchanged runtime would mean the forced branch is not actually
    being exercised, which would make F1 uninterpretable whichever way it came out.
    SUPPORTED if 0 halts AND median(forced) > median(baseline).

F4  CONTROL — THE 25 NON-MISSING PAIRS ARE UNCHANGED.
    The 25 pairs that already reached the in-grammar optimum still return exactly the same
    IoU. If forcing the branch perturbs pairs it should not touch, the diff is not isolating
    the disjoint path and F1's recovery could be incidental.
    SUPPORTED if all 25 match baseline to within 1e-12.
"""


def grammar_max_all(dense, neurons):
    """Exhaustive in-grammar length-3 max per neuron, integer popcounts."""
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


def exact_iou(rts, dense, concepts, neuron):
    """One exact run; full-precision IoU recovered from the integer counts run_one keeps."""
    res = rts.run_one(dense, neuron, LENGTH, CAP, TIME_BUDGET, None, concepts=concepts)
    iou = res["n_inter"] / (res["n_fires"] + int(neuron.sum()) - res["n_inter"])
    return iou, res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.parse_args()

    print(PREDICTIONS, flush=True)

    sys.path.insert(0, os.environ.get("OPTIMALCE_UPSTREAM",
                                      os.path.expanduser("~/projects/optimalce")))
    import real_token_masks as rtm
    import real_token_search as rts
    from utils import optimal_utils

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
    print(f"[falsification] {len(tokens)} tokens | K={K} length={LENGTH} | "
          f"{len(keys)} pairs\n", flush=True)

    print("in-grammar optima ...", flush=True)
    gram = grammar_max_all(dense, neurons)

    print("baseline (are_disjoint untouched) ...", flush=True)
    base = [exact_iou(rts, dense, concepts, n) for n in neurons]

    # ---- THE ONE LINE ----
    real_are_disjoint = optimal_utils.are_disjoint
    optimal_utils.are_disjoint = lambda *a, **k: False
    try:
        print("forced (are_disjoint -> False) ...", flush=True)
        forced = [exact_iou(rts, dense, concepts, n) for n in neurons]
    finally:
        optimal_utils.are_disjoint = real_are_disjoint

    rows = []
    for (arm, alpha, u), g, (bi, br), (fi, fr) in zip(keys, gram, base, forced):
        rows.append({
            "arm": arm, "alpha": alpha, "unit": f"unit{u}",
            "in_grammar_max": repr(g),
            "baseline_iou": repr(bi), "forced_iou": repr(fi),
            "baseline_missed": int(g - bi > 1e-12), "forced_missed": int(g - fi > 1e-12),
            "delta_iou": repr(fi - bi),
            "baseline_time_s": br["time_s"], "forced_time_s": fr["time_s"],
            "baseline_visited": br["visited"], "forced_visited": fr["visited"],
            "baseline_halted": br["halted"], "forced_halted": fr["halted"],
            "baseline_formula": br["formula"], "forced_formula": fr["formula"],
        })
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}")

    def med(xs):
        xs = sorted(xs)
        n = len(xs)
        return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

    print("\n--- DIFF vs the committed exact run ---")
    print(f"  {'pair':>30} {'in_gram':>10} {'baseline':>10} {'forced':>10} "
          f"{'dIoU':>11} {'t_base':>8} {'t_forced':>9}")
    for r in rows:
        tag = ""
        if r["baseline_missed"]:
            tag = "  RECOVERED" if not r["forced_missed"] else "  STILL MISSING"
        elif r["forced_missed"]:
            tag = "  NEWLY MISSING"
        print(f"  {r['arm'] + ' a=' + r['alpha'] + ' ' + r['unit']:>30} "
              f"{float(r['in_grammar_max']):>10.6f} {float(r['baseline_iou']):>10.6f} "
              f"{float(r['forced_iou']):>10.6f} {float(r['delta_iou']):>+11.2e} "
              f"{float(r['baseline_time_s']):>8.2f} {float(r['forced_time_s']):>9.2f}{tag}")

    print("\n--- VERDICTS ---")
    res = []
    rec = {k: not any(r["forced_missed"] for r in rows
                      if (r["arm"], r["alpha"], r["unit"]) == (k[0], k[1], f"unit{k[2]}"))
           for k in KNOWN_MISSES}
    got = {}
    for k in KNOWN_MISSES:
        r = next(r for r in rows
                 if (r["arm"], r["alpha"], r["unit"]) == (k[0], k[1], f"unit{k[2]}"))
        got[k] = (float(r["forced_iou"]), float(r["in_grammar_max"]))
    res.append(_v("F1", all(rec.values()),
                  "; ".join(f"{k[0]} a={k[1]} unit{k[2]}: forced={v[0]!r} "
                            f"in_grammar={v[1]!r} -> "
                            f"{'RECOVERED' if rec[k] else 'STILL MISSING'}"
                            for k, v in got.items())))

    worst = min(float(r["delta_iou"]) for r in rows)
    nworse = sum(1 for r in rows if float(r["delta_iou"]) < -1e-12)
    res.append(_v("F2", nworse == 0,
                  f"{nworse} pairs worse than baseline; most negative delta = {worst:+.3e}"))

    halts = [r for r in rows if r["forced_halted"] != "no"]
    mb = med([float(r["baseline_time_s"]) for r in rows])
    mf = med([float(r["forced_time_s"]) for r in rows])
    res.append(_v("F3", not halts and mf > mb,
                  f"{len(halts)} halts under the {TIME_BUDGET}s cap; median time "
                  f"{mb:.2f}s -> {mf:.2f}s ({mf / mb:.2f}x)"))

    unaffected = [r for r in rows if not r["baseline_missed"]]
    moved = [r for r in unaffected if abs(float(r["delta_iou"])) > 1e-12]
    res.append(_v("F4", not moved,
                  f"{len(unaffected)} previously-optimal pairs; {len(moved)} changed IoU"
                  + ("; " + ", ".join(f"{r['arm']}/{r['unit']}" for r in moved[:5])
                     if moved else "")))

    print(f"\n  scored {sum(res)}/{len(res)} SUPPORTED")
    if not all(rec.values()):
        print("\n  *** F1 FAILED: the disjoint branch is NOT the cause. The root-cause")
        print("      paragraph in METHOD_NOTES is wrong and must be retracted. DO NOT PATCH.")
    return 0


def _v(name, ok, detail):
    print(f"  {name}: {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    print(f"      {detail}")
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
