"""Experiment 1 — LENGTH LADDER at K=8: does exact search reach length 5?

Question. 5.9 established that length dominates cost far more than K does (a 15->50 K
increase at length 3 costs ~4x; +1 length at K=15 costs 100x or more). The obvious move is
to spend the K budget on length: drop K to 8 and ask whether exact search reaches length 5.

Design. Same corpus, same units, same search as the length-3 tractability grid in
METHOD_NOTES ("Length-3 tractability across K (M = 24,199, trained a=0.1, 5 units, exact)"):
M = 24,199 tokens, arm `all`, trained alpha=0.1, `--unit_ids 88 92 396 413 510`, exact
search, cap 200,000, soft time budget 1500s (Phase B's threshold, reused unchanged).

The alpha, unit set and time budget are NOT free choices here — they are the only values for
which comparable exact length-3 and length-4 numbers already exist, so any other value would
break the ladder this experiment is built to measure.

Controls, both of which could have failed:
  * lengths 3 and 4 are re-run AT K=8 in the same script, so the length-5 number is compared
    against its own K, not against the K=15 numbers.
  * the `lemma` arm is a disjoint vocabulary; at K=50 length 3 it terminated in 1-2 visited
    nodes. If it also explodes at length 5, the cost is not about concept overlap and the
    whole reading is void.

Usage::

    python src/exp_length_ladder.py --stage A     # lengths 3,4 at K=8 + lemma control
    python src/exp_length_ladder.py --stage B     # length 5 at K=8
    python src/exp_length_ladder.py --score       # print verdicts from whatever exists
"""

import argparse
import csv
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUTDIR = os.path.join(REPO, "results", "length_ladder")

# --- fixed design ------------------------------------------------------------------
UNIT_IDS = ["88", "92", "396", "413", "510"]
ACTS = "results/acts2k_trained_a0.1.npz"
K = 8
TIME_BUDGET = 1500.0   # Phase B's soft cap, reused unchanged
CAP = 200000
MAX_SENTS = 2000
MIN_SUPPORT = 5

# --- baseline, pasted from results/beam_vs_exact_K15.csv (exact side, trained alpha=0.1) ---
# unit396 687.02 no | unit413 200.69 no | unit510 1514.55 TIME | unit88 400.75 no | unit92 886.77 no
L4_K15_TIMES = {"unit396": 687.02, "unit413": 200.69, "unit510": 1514.55,
                "unit88": 400.75, "unit92": 886.77}
L4_K15_TIMEOUTS = 1
L4_K15_MEDIAN = 687.02

# In-grammar candidate counts, K * (3K)^(L-1) -- the space the search actually constructs
# (METHOD_NOTES: "the method's formula grammar is narrower than length-4-over-K-concepts").
SPACE = {(3, 15): 30375, (4, 15): 1366875, (3, 8): 4608, (4, 8): 110592,
         (5, 8): 2654208, (6, 8): 63700992}
SPACE_RATIO_L5K8_OVER_L4K15 = SPACE[(5, 8)] / SPACE[(4, 15)]   # 1.942

# =====================================================================================
# PRE-REGISTERED PREDICTIONS — written and committed before any length-5 run existed.
# Each is scored by score() below from the CSVs, not by reading.
# =====================================================================================
PREDICTIONS = """
P1  COST EXCEEDS THE SPACE-SIZE RATIO.
    The (L5,K=8) candidate space is 1.942x the (L4,K=15) space, which is the whole reason
    K=8 was chosen. I predict wall time overshoots that ratio: median time at (L5,K=8) over
    the same 5 units will exceed 1.942 x 687.02s = 1334.2s.
    Why: A*'s admissible bound loosens with remaining depth, so a deeper search prunes a
    smaller fraction of its space. Candidate count is a floor on cost, not an estimate.
    SUPPORTED if median(L5,K=8) time > 1334.2s (timeouts counted at their observed wall
    time, which understates them, so this test is biased AGAINST P1).

P2  PEAK FRONTIER GROWS WITH LENGTH AT FIXED K.
    peak_frontier at (L5,K=8) > peak_frontier at (L4,K=8) on at least 4 of 5 units.
    This is the frontier-explosion mechanism restated at constant branching factor: if it
    fails, the cost of length is not a frontier effect.

P3  CONTROL — K=8 IS GENUINELY CHEAPER THAN K=15 AT MATCHED LENGTH.
    median time at (L4,K=8) < 343.5s, i.e. under half the (L4,K=15) median of 687.02s.
    This could fail. If K=8 does not buy a real discount at length 4, the premise of the
    whole experiment -- spend the K budget on length -- is void and the length-5 number
    means nothing.

P4  CONTROL — THE DISJOINT ARM STAYS TRIVIAL AT LENGTH 5.
    lemma-only (disjoint vocabulary) at K=8, length 5 terminates on all 5 units, each in
    under 10s. At K=50 length 3 it terminated in 1-2 visited nodes. If a disjoint
    vocabulary also explodes at length 5, cost is not driven by concept overlap and every
    reading above is void.

P5  LENGTH 6 AT K=8 IS OUT OF REACH AND IS NOT ATTEMPTED BLIND.
    SPACE[(6,8)]/SPACE[(5,8)] = 24.0. Registered in advance: length 6 is only attempted if
    the measured median (L5,K=8) time is under 60s, which would put a 24x extrapolation
    inside the 1500s cap. Otherwise it is reported as not attempted, with the reason.
"""


def run(arm, length, out, k=K, time_budget=TIME_BUDGET):
    """One real_token_search.py invocation: all 5 units, one (arm, length)."""
    os.makedirs(OUTDIR, exist_ok=True)
    # Same interpreter resolution as verify/run_all.sh: the compexp env holds scipy/torch.
    py = os.environ.get("PY") or os.path.expanduser("~/miniconda3/envs/compexp/bin/python")
    if not os.path.exists(py):
        py = sys.executable
    cmd = [py, os.path.join(HERE, "real_token_search.py"),
           "--arms", arm, "--K", str(k), "--lengths", str(length),
           "--min_support", str(MIN_SUPPORT), "--max_sents", str(MAX_SENTS),
           "--cap", str(CAP), "--time_budget", str(time_budget), "--seed", "0",
           "--neuron", "real", "--beam_list", "none", "--acts", ACTS,
           "--unit_ids", *UNIT_IDS, "--dmin", "0", "--dmax", "1", "--min_fire", "200",
           "--out", out]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def load(pattern):
    rows = []
    for path in sorted(glob.glob(os.path.join(OUTDIR, pattern))):
        with open(path) as f:
            for r in csv.DictReader(l for l in f if not l.startswith("#")):
                r["_src"] = os.path.basename(path)
                rows.append(r)
    return rows


def sel(rows, arm_prefix, length):
    return [r for r in rows if r["arm"].startswith(arm_prefix) and int(r["length"]) == length]


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return float("nan")
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def verdict(name, ok, detail):
    tag = "SUPPORTED" if ok else "NOT SUPPORTED"
    print(f"  {name}: {tag}")
    print(f"      {detail}")
    return ok


def summarise(rows, label):
    if not rows:
        print(f"  {label}: (no runs)")
        return
    print(f"  {label}:")
    print(f"    {'unit':>9} {'peak':>9} {'visited':>9} {'expanded':>9} {'time_s':>9} "
          f"{'IoU':>8} {'cov':>7} {'halted':>7}")
    for r in sorted(rows, key=lambda r: r["neuron"]):
        print(f"    {r['neuron']:>9} {r['peak_frontier']:>9} {r['visited']:>9} "
              f"{r['expanded']:>9} {r['time_s']:>9} {str(r['best_iou']):>8} "
              f"{str(r['formula_cov']):>7} {r['halted']:>7}")
    to = [r for r in rows if r["halted"] != "no"]
    print(f"    timeouts/caps: {len(to)}/{len(rows)}"
          + (f"  ({', '.join(r['neuron'] + ':' + r['halted'] for r in to)})" if to else ""))


def score():
    all_rows = load("*.csv")
    l3 = sel(all_rows, "all categories", 3)
    l4 = sel(all_rows, "all categories", 4)
    l5 = sel(all_rows, "all categories", 5)
    l6 = sel(all_rows, "all categories", 6)
    lem5 = sel(all_rows, "lemma-only", 5)

    print("=" * 86)
    print("EXPERIMENT 1 — LENGTH LADDER at K=8 (M=24,199, arm=all, trained alpha=0.1,")
    print("               units 88/92/396/413/510, exact, cap=200k, soft budget 1500s)")
    print("=" * 86)
    print("\n--- PRE-REGISTERED PREDICTIONS ---")
    print(PREDICTIONS)

    print("--- RAW ---")
    summarise(l3, "length 3, K=8, all categories")
    summarise(l4, "length 4, K=8, all categories")
    summarise(l5, "length 5, K=8, all categories")
    if l6:
        summarise(l6, "length 6, K=8, all categories")
    summarise(lem5, "length 5, K=8, lemma-only (disjoint control)")

    print("\n--- BASELINE, pasted from results/beam_vs_exact_K15.csv ---")
    print(f"    length 4, K=15, same units: times {L4_K15_TIMES}")
    print(f"    median {L4_K15_MEDIAN}s, {L4_K15_TIMEOUTS} timeout")

    print("\n--- VERDICTS ---")
    results = []

    if l5:
        t5 = [float(r["time_s"]) for r in l5]
        m5 = median(t5)
        thr = round(SPACE_RATIO_L5K8_OVER_L4K15 * L4_K15_MEDIAN, 1)
        results.append(verdict(
            "P1", m5 > thr,
            f"median(L5,K=8) = {m5:.2f}s vs threshold {thr}s "
            f"(= {SPACE_RATIO_L5K8_OVER_L4K15:.3f} x {L4_K15_MEDIAN}s); "
            f"times {[round(t, 2) for t in sorted(t5)]}"))
    else:
        print("  P1: NOT SCORED (no length-5 runs)")

    if l5 and l4:
        p4 = {r["neuron"]: int(r["peak_frontier"]) for r in l4}
        p5 = {r["neuron"]: int(r["peak_frontier"]) for r in l5}
        shared = sorted(set(p4) & set(p5))
        grew = [u for u in shared if p5[u] > p4[u]]
        results.append(verdict(
            "P2", len(grew) >= 4,
            f"peak grew on {len(grew)}/{len(shared)} units; "
            + ", ".join(f"{u} {p4[u]}->{p5[u]}" for u in shared)))
    else:
        print("  P2: NOT SCORED (need both length 4 and length 5 at K=8)")

    if l4:
        t4 = [float(r["time_s"]) for r in l4]
        m4 = median(t4)
        results.append(verdict(
            "P3", m4 < 343.5,
            f"median(L4,K=8) = {m4:.2f}s vs threshold 343.5s (= 0.5 x {L4_K15_MEDIAN}s); "
            f"times {[round(t, 2) for t in sorted(t4)]}"))
    else:
        print("  P3: NOT SCORED (no length-4 K=8 runs)")

    if lem5:
        slow = [r for r in lem5 if float(r["time_s"]) >= 10.0]
        bad = [r for r in lem5 if r["halted"] != "no"]
        results.append(verdict(
            "P4", not slow and not bad and len(lem5) == 5,
            f"{len(lem5)} runs, {len(bad)} halted, {len(slow)} over 10s; "
            f"times {sorted(round(float(r['time_s']), 2) for r in lem5)}, "
            f"visited {sorted(int(r['visited']) for r in lem5)}"))
    else:
        print("  P4: NOT SCORED (no lemma length-5 runs)")

    if l5:
        m5 = median([float(r["time_s"]) for r in l5])
        attempted = bool(l6)
        should = m5 < 60.0
        results.append(verdict(
            "P5", attempted == should,
            f"median(L5,K=8) = {m5:.2f}s; gate is <60s -> length 6 "
            f"{'attempted' if should else 'NOT attempted'}; actually "
            f"{'attempted' if attempted else 'not attempted'}. "
            f"SPACE[(6,8)]/SPACE[(5,8)] = {SPACE[(6, 8)] / SPACE[(5, 8)]:.1f}x, so the "
            f"extrapolated length-6 median is {m5 * 24:.0f}s per unit."))
    else:
        print("  P5: NOT SCORED (no length-5 runs)")

    print(f"\n  scored {sum(results)}/{len(results)} SUPPORTED")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["A", "B", "C"], default=None,
                    help="A = lengths 3,4 at K=8 + lemma L5 control; "
                         "B = length 5 at K=8; C = length 6 at K=8 (gated by P5)")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    if args.stage == "A":
        print(PREDICTIONS, flush=True)
        run("all", 3, os.path.join(OUTDIR, "L3_all_K8.csv"))
        run("lemma", 5, os.path.join(OUTDIR, "L5_lemma_K8.csv"))
        run("all", 4, os.path.join(OUTDIR, "L4_all_K8.csv"))
    elif args.stage == "B":
        run("all", 5, os.path.join(OUTDIR, "L5_all_K8.csv"))
    elif args.stage == "C":
        run("all", 6, os.path.join(OUTDIR, "L6_all_K8.csv"))

    if args.score or args.stage:
        return score()
    return score()


if __name__ == "__main__":
    raise SystemExit(main())
