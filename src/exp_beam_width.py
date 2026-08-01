"""Experiment 2 — BEAM WIDTH vs ACCURACY at length 3.

Question. The paper uses beam size 5 (Appendix B); this project used beam 200, a 40x wider
beam, and got 20/27 exact agreement at length 3 with a ratio-of-averages of +0.96% over all
pairs / +5.05% restricted to differing pairs. Two things follow that were never measured:

  (a) At what beam width does agreement with exact collapse?
  (b) Does the paper's own beam width (5) reproduce the paper's reported gap of +5.1-6.5%?

Design. The exact same 27 (arm, alpha, unit) pairs as `results/beam_vs_exact_L3_K15.csv`,
K = 15, length 3, M = 24,199, min_support 5. Beam widths {5, 10, 25, 50, 100, 200} plus
exact, all run in-process against the same masks so the comparison is like-for-like.

Formula equality is EXTENSIONAL and exact here: the winning formula's token mask is captured
directly from the search and compared bit-for-bit against exact's. `phaseB_report.py` uses
(n_fires, n_inter) as an extensional proxy; this script uses the mask itself, and reports
whether the two tests ever disagree.

Stratification is within (arm, alpha) throughout -- alpha moves lift and density together,
so pooled statistics over alpha measure density.

Usage::

    python src/exp_beam_width.py --run
    python src/exp_beam_width.py --score
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

OUT = os.path.join(REPO, "results", "beam_width_L3.csv")

# --- fixed design: the 27 pairs, read off results/beam_vs_exact_L3_K15.csv ---------
PAIRS = [
    ("trained",   "0.2",   [88, 92, 413]),
    ("trained",   "0.1",   [88, 92, 396, 413, 510]),
    ("trained",   "0.05",  [86, 87, 91, 395, 412]),
    ("untrained", "0.2",   [88, 92, 396, 510]),
    ("untrained", "0.1",   [88, 92, 396, 413, 510]),
    ("untrained", "0.05",  [88, 92, 396, 413, 510]),
]
WIDTHS = [5, 10, 25, 50, 100, 200]
K = 15
LENGTH = 3
MAX_SENTS = 2000
MIN_SUPPORT = 5
CAP = 200000
TIME_BUDGET = 1500.0
VISION_BAND = (5.1, 6.5)

# --- baseline, pasted from results/phaseB_report_L3.txt (beam 200 vs exact, 27 pairs) ---
B200_AGREE = 20
B200_ROA_ALL = 0.96
B200_ROA_RESTRICTED = 5.05

# =====================================================================================
# PRE-REGISTERED PREDICTIONS — committed before any beam width below 200 was run.
# Scored by score() from the CSV, not by reading.
# =====================================================================================
PREDICTIONS = """
B1  AGREEMENT IS MONOTONE NON-DECREASING IN BEAM WIDTH.
    Across {5, 10, 25, 50, 100, 200}, the count of pairs whose returned mask is identical
    to exact never strictly decreases as width grows.
    This can fail. Frontier truncation keeps the top-N by bound, and the incumbent that
    survives truncation at width W is not guaranteed to survive at width 2W once a
    different node updates the incumbent first, so monotonicity is an assumption about
    the search, not a theorem about it.
    SUPPORTED if agree(5) <= agree(10) <= agree(25) <= agree(50) <= agree(100) <= agree(200).

B2  BEAM 5 DOES NOT REPRODUCE THE PAPER'S BAND ON ALL PAIRS.
    Ratio-of-averages (exact over beam) at width 5, over all 27 pairs, stays BELOW the
    bottom of the vision band: < 5.1%.
    Why: at length 3 the in-grammar space is only K*(3K)^2 = 30,375 formulas, and beam 200
    already agrees with exact on 20/27, which says most units here have an optimum that is
    easy to reach. A 40x narrower beam should lose ground but not 5 percentage points of it.
    SUPPORTED if roa_all(5) < 5.1. NOT SUPPORTED if it reaches or exceeds 5.1.

    Reported alongside but NOT the scored test: the restricted-to-differing statistic. Its
    denominator set CHANGES with width -- a narrower beam makes more pairs differ, pulling
    easy small-gap pairs into the restricted set -- so it is not comparable across widths
    and must not be read as a trend. The all-pairs set is fixed at 27 and is.

B3  NO COLLAPSE IN THE SWEPT RANGE.
    "Collapse" is defined here, in advance, as agreement falling to 13 or fewer of 27
    (below half). I predict this never happens, i.e. agreement at width 5 is still >= 14/27.
    SUPPORTED if min over widths of agree(w) >= 14.

B4  CONTROL — THE EXACT RE-RUN REPRODUCES THE COMMITTED EXACT NUMBERS.
    Re-running exact on all 27 pairs reproduces the exact_IoU column of
    results/beam_vs_exact_L3_K15.csv to within 1e-6 on every pair.
    This could fail. If it does, this sweep is not measured against the same thing the
    20/27 and +5.05% figures were, and every comparison to them is void.

B5  CONTROL — THE BEAM CAP BINDS WHERE IT SHOULD AND NOWHERE ELSE.
    On any pair whose EXACT peak frontier never exceeds the beam width W, beam-W must
    return exact's mask -- the cap was never reached, so the two searches are the same
    search. Zero violations expected.
    This could fail. A violation means MAX_FRONTIER_SIZE changes the result on runs where
    it was never active, i.e. the beam patch is not the no-op it is verified to be off the
    cap, and the whole beam-vs-exact series would need re-examining.
"""

# =====================================================================================
# AMENDMENT, WRITTEN AFTER THE DATA — recorded as after-the-fact, not slipped into the
# pre-registration above, which is left byte-for-byte as it was committed.
#
# The sweep produced an outcome none of B1-B5 anticipated: at narrow widths the search
# returns NO FORMULA AT ALL -- upstream's `best_results = (-1.0, None)` initialiser is
# never updated, so `visited=0`, `best_iou=-1.0`, `best_label=None`. `perform_search`
# documents this ("`best_label` is a formula object or `None` if no valid formula is
# found"), so it is a sanctioned upstream outcome, not a bug in our patch.
#
# Mechanism: `_apply_beam_cap` keeps the top-N nodes by ESTIMATED CEILING. An INDIVIDUAL
# node has been fully resolved and carries its true IoU, which sits below the optimistic
# ceilings of unexpanded nodes, so the cap preferentially evicts exactly the nodes that
# could have become the answer.
#
# Consequence for scoring: a -1.0 IoU in a sum makes ratio-of-averages meaningless (it
# produced -192.81% at width 5 and +793.54% at width 10). B2 was therefore scored on a
# poisoned statistic and its SUPPORTED verdict is VOID -- it passed a "< 5.1%" test only
# because the statistic went negative. B2 is re-scored below on the FIXED subset of pairs
# that returned a solution at EVERY width, which keeps the denominator set constant across
# widths (the confound B2 itself warned about for the restricted statistic).
#
# The no-solution runs are reported explicitly at every width and are never dropped from a
# mean silently, per the standing rule for timeouts.
# =====================================================================================
AMENDMENT_NOTE = "B2 as pre-registered is VOID (statistic poisoned by -1.0); re-scored as B2'"


def run():
    import real_token_search as rts
    import real_token_masks as rtm
    import env_info
    env_info.print_banner("beam-width-L3")

    feats = os.path.expanduser(
        "~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats")
    tokens = rtm.load_tokens(feats, MAX_SENTS)
    _, cats = rts.ARMS["all"]
    concepts = rtm.select_concepts(tokens, cats, K, MIN_SUPPORT)
    dense = rtm.build_dense(tokens, concepts)
    print(f"[beam-width] {len(tokens)} tokens | K={K} length={LENGTH} "
          f"widths={WIDTHS}+exact | {len(PAIRS)} strata\n", flush=True)

    # Capture the winning mask itself, so extensional equality is a bit-for-bit mask test
    # rather than the (n_fires, n_inter) proxy phaseB_report.py uses. Wrapping rather than
    # editing real_token_search.py keeps that file untouched.
    grabbed = {}
    real_formula_stats = rts.formula_stats

    def grabbing_formula_stats(f, concepts_, dense_, neuron_bits=None):
        grabbed["mask"] = rts.eval_formula(f, dense_)
        return real_formula_stats(f, concepts_, dense_, neuron_bits)

    rts.formula_stats = grabbing_formula_stats

    rows = []
    try:
        for arm, alpha, units in PAIRS:
            acts = os.path.join(REPO, "results", f"acts2k_{arm}_a{alpha}.npz")
            picked, untrained, alpha_f, _ = rts.load_real_neurons(
                acts, len(units), np.random.default_rng(999), dmin=0, dmax=1,
                min_fire=200, unit_ids=units)
            for uid, neuron_bits in picked:
                # Exact first: it is the reference every width on this pair is scored against.
                per_cond = {}
                for cond in ["none"] + WIDTHS:
                    beam_cap = None if cond == "none" else int(cond)
                    grabbed.clear()
                    res = rts.run_one(dense, neuron_bits, LENGTH, CAP, TIME_BUDGET,
                                      beam_cap, concepts=concepts)
                    per_cond[cond] = (res, grabbed.get("mask"))
                    print(f"  {arm} a={alpha} unit{uid} beam={cond}: "
                          f"IoU={res['best_iou']} peak={res['peak_frontier']} "
                          f"t={res['time_s']}s halted={res['halted']}", flush=True)

                ex_res, ex_mask = per_cond["none"]
                for w in WIDTHS:
                    b_res, b_mask = per_cond[w]
                    same_mask = (ex_mask is not None and b_mask is not None
                                 and bool(np.array_equal(ex_mask, b_mask)))
                    same_proxy = (b_res["n_fires"] == ex_res["n_fires"]
                                  and b_res["n_inter"] == ex_res["n_inter"])
                    rows.append({
                        "arm": arm, "alpha": alpha, "unit": f"unit{uid}",
                        "density_run": round(float(neuron_bits.mean()), 5),
                        "n_fire_neuron": int(neuron_bits.sum()),
                        "beam": w,
                        "beam_IoU": b_res["best_iou"], "exact_IoU": ex_res["best_iou"],
                        "beam_cov": b_res["formula_cov"], "exact_cov": ex_res["formula_cov"],
                        "same_mask": int(same_mask), "same_proxy": int(same_proxy),
                        "same_string": int(b_res["formula"] == ex_res["formula"]),
                        "beam_peak": b_res["peak_frontier"],
                        "exact_peak": ex_res["peak_frontier"],
                        "cap_bound": int(ex_res["peak_frontier"] > w),
                        "beam_visited": b_res["visited"], "exact_visited": ex_res["visited"],
                        "beam_time_s": b_res["time_s"], "exact_time_s": ex_res["time_s"],
                        "beam_halted": b_res["halted"], "exact_halted": ex_res["halted"],
                        "beam_formula": b_res["formula"], "exact_formula": ex_res["formula"],
                    })
    finally:
        rts.formula_stats = real_formula_stats

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}  ({len(rows)} rows = {len(rows) // len(WIDTHS)} pairs "
          f"x {len(WIDTHS)} widths)")


def score():
    with open(OUT) as f:
        rows = [r for r in csv.DictReader(f)]
    for r in rows:
        for k in ("beam_IoU", "exact_IoU", "beam_cov", "exact_cov"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else float("nan")
        for k in ("same_mask", "same_proxy", "same_string", "beam", "cap_bound",
                  "beam_peak", "exact_peak"):
            r[k] = int(float(r[k]))

    # A run that returned no formula: upstream's (-1.0, None) initialiser was never updated.
    for r in rows:
        r["nosol"] = int(r["beam_formula"] in ("", "None") or r["beam_IoU"] < 0)

    by_w = {w: [r for r in rows if r["beam"] == w] for w in WIDTHS}
    n_pairs = len(by_w[WIDTHS[0]])

    def key(r):
        return (r["arm"], r["alpha"], r["unit"])

    # FIXED like-for-like set: pairs that returned a solution at every width. Fixed across
    # widths, so ratio-of-averages is comparable along the sweep.
    dropped = {key(r) for r in rows if r["nosol"]}
    fixed = {key(r) for r in rows} - dropped

    def agree(w, pool=None):
        return sum(r["same_mask"] for r in (pool if pool is not None else by_w[w]))

    def roa(pool):
        se, sb = sum(r["exact_IoU"] for r in pool), sum(r["beam_IoU"] for r in pool)
        return (se / sb - 1.0) * 100 if sb else float("nan")

    def fx(w):
        return [r for r in by_w[w] if key(r) in fixed]

    print("=" * 88)
    print("EXPERIMENT 2 — BEAM WIDTH vs ACCURACY at length 3 "
          f"(K={K}, M=24,199, {n_pairs} pairs)")
    print("=" * 88)
    print("\n--- PRE-REGISTERED PREDICTIONS ---")
    print(PREDICTIONS)

    print("--- NO-SOLUTION RUNS (reported, never dropped silently) ---")
    print("    The search returned best_label=None / best_iou=-1.0 / visited=0 -- the beam")
    print("    cap evicted every node that could have become the answer.")
    print(f"    {'beam':>6} {'no_solution':>12}")
    for w in WIDTHS:
        print(f"    {w:>6} {sum(r['nosol'] for r in by_w[w]):>8}/{len(by_w[w])}")
    print(f"    pairs returning a solution at ALL {len(WIDTHS)} widths: "
          f"{len(fixed)}/{n_pairs}  (dropped {len(dropped)})")
    if dropped:
        print("    dropped: " + ", ".join(f"{a}/a={al}/{u}"
                                          for a, al, u in sorted(dropped)))

    print("\n--- FIXED-SET SWEEP (the like-for-like statistic; n constant across widths) ---")
    print(f"    {'beam':>6} {'n':>4} {'agree':>9} {'roa_all%':>9} {'band?':>7} {'medtime':>8}")
    for w in WIDTHS:
        pool = fx(w)
        ra = roa(pool)
        times = sorted(float(r["beam_time_s"]) for r in pool)
        print(f"    {w:>6} {len(pool):>4} {agree(w, pool):>4}/{len(pool):<4} {ra:>9.2f} "
              f"{('IN' if VISION_BAND[0] <= ra <= VISION_BAND[1] else 'out'):>7} "
              f"{times[len(times) // 2]:>8.2f}")

    print("\n--- ALL-PAIRS SWEEP (roa is MEANINGLESS wherever no_solution > 0) ---")
    print(f"    {'beam':>6} {'agree/n':>9} {'roa_all%':>9} {'roa_diff%':>10} {'n_diff':>7} "
          f"{'band?':>9} {'capbound':>9} {'medtime':>8}")
    for w in WIDTHS:
        pool = by_w[w]
        diff = [r for r in pool if not r["same_mask"]]
        ra, rd = roa(pool), (roa(diff) if diff else float("nan"))
        band = "IN" if VISION_BAND[0] <= ra <= VISION_BAND[1] else "out"
        times = sorted(float(r["beam_time_s"]) for r in pool)
        print(f"    {w:>6} {agree(w):>4}/{len(pool):<4} {ra:>9.2f} {rd:>10.2f} "
              f"{len(diff):>7} {band:>9} "
              f"{sum(r['cap_bound'] for r in pool):>4}/{len(pool):<4} "
              f"{times[len(times) // 2]:>8.2f}")
    print("    roa_diff% is NOT comparable across widths (its denominator set moves);"
          " shown for reference only.")

    print("\n--- STRATIFIED WITHIN (arm, alpha) — agreement count per stratum ---")
    strata = sorted({(r["arm"], r["alpha"]) for r in rows},
                    key=lambda s: (s[0], -float(s[1])))
    print(f"    {'stratum':>20} {'n':>3} " + " ".join(f"{('b' + str(w)):>6}" for w in WIDTHS))
    for arm, alpha in strata:
        sub = {w: [r for r in by_w[w] if r["arm"] == arm and r["alpha"] == alpha]
               for w in WIDTHS}
        n = len(sub[WIDTHS[0]])
        print(f"    {arm + ' a=' + alpha:>20} {n:>3} "
              + " ".join(f"{sum(r['same_mask'] for r in sub[w]):>6}" for w in WIDTHS))

    print("\n--- STRATIFIED WITHIN (arm, alpha) — ratio-of-averages %, all pairs ---")
    print(f"    {'stratum':>20} {'n':>3} " + " ".join(f"{('b' + str(w)):>7}" for w in WIDTHS))
    for arm, alpha in strata:
        sub = {w: [r for r in by_w[w] if r["arm"] == arm and r["alpha"] == alpha]
               for w in WIDTHS}
        n = len(sub[WIDTHS[0]])
        print(f"    {arm + ' a=' + alpha:>20} {n:>3} "
              + " ".join(f"{roa(sub[w]):>7.2f}" for w in WIDTHS))

    print("\n--- BASELINE, pasted from results/phaseB_report_L3.txt ---")
    print(f"    beam 200 vs exact: {B200_AGREE}/27 agree, "
          f"roa_all = +{B200_ROA_ALL}%, roa_restricted = +{B200_ROA_RESTRICTED}%")

    print("\n--- VERDICTS ---")
    results = []

    counts = [agree(w) for w in WIDTHS]
    mono = all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))
    results.append(_v("B1", mono,
                      "agreement by width " + ", ".join(f"{w}:{c}" for w, c in
                                                        zip(WIDTHS, counts))
                      + ("" if mono else "  <- NON-MONOTONE")))

    ra5 = roa(by_w[5])
    print(f"  B2: VOID as pre-registered — {AMENDMENT_NOTE}")
    print(f"      roa_all(beam 5) over all {n_pairs} pairs = {ra5:+.2f}%, which is negative "
          f"because {sum(r['nosol'] for r in by_w[5])} runs contributed IoU = -1.0. The "
          f"'< {VISION_BAND[0]}%' test passes on this number, but the number is not a gap. "
          f"Not counted as support.")
    ra5f = roa(fx(5))
    results.append(_v("B2'", ra5f < VISION_BAND[0],
                      f"[re-scored on the fixed {len(fixed)}-pair set] roa(beam 5) = "
                      f"{ra5f:+.2f}% vs band floor {VISION_BAND[0]}%; band is "
                      f"{VISION_BAND[0]}-{VISION_BAND[1]}%, so beam 5 "
                      + ("does NOT reach it" if ra5f < VISION_BAND[0] else
                         ("lands INSIDE it" if ra5f <= VISION_BAND[1] else "OVERSHOOTS it"))
                      + ". NOTE: this measures our frontier cap, NOT the paper's "
                        "beam_optimal algorithm — see experiment 2b."))

    results.append(_v("B3", min(counts) >= 14,
                      f"min agreement over widths = {min(counts)}/{n_pairs} at beam "
                      f"{WIDTHS[counts.index(min(counts))]}; collapse threshold is <=13"))

    ref = {}
    with open(os.path.join(REPO, "results", "beam_vs_exact_L3_K15.csv")) as f:
        for r in csv.DictReader(f):
            ref[(r["arm"], r["alpha"], r["unit"])] = float(r["exact_IoU"])
    bad, checked = [], 0
    for r in by_w[200]:
        key = (r["arm"], r["alpha"], r["unit"])
        if key in ref:
            checked += 1
            if abs(ref[key] - r["exact_IoU"]) > 1e-6:
                bad.append(f"{key} committed={ref[key]} rerun={r['exact_IoU']}")
    results.append(_v("B4", checked == n_pairs and not bad,
                      f"{checked}/{n_pairs} pairs matched against the committed exact_IoU; "
                      f"{len(bad)} mismatches" + (("; " + "; ".join(bad[:3])) if bad else "")))

    viol = [r for r in rows if not r["cap_bound"] and not r["same_mask"]]
    n_unbound = sum(1 for r in rows if not r["cap_bound"])
    results.append(_v("B5", not viol,
                      f"{n_unbound} of {len(rows)} (pair, width) runs never reached the cap "
                      f"(exact_peak <= W); {len(viol)} of those disagreed with exact"
                      + (("; e.g. " + str([(v['arm'], v['alpha'], v['unit'], v['beam'])
                                           for v in viol[:3]])) if viol else "")))

    dis = [r for r in rows if r["same_mask"] != r["same_proxy"]]
    print(f"\n  cross-check: mask test vs phaseB_report's (n_fires, n_inter) proxy "
          f"disagree on {len(dis)}/{len(rows)} runs"
          + (f"  <- the proxy is NOT safe here: {[(d['arm'], d['alpha'], d['unit'], d['beam']) for d in dis[:5]]}"
             if dis else "  (proxy is safe on this data)"))
    syn = sum(1 for r in by_w[200] if r["same_string"])
    print(f"  cross-check: at beam 200, {agree(200)}/{n_pairs} agree extensionally, "
          f"{syn}/{n_pairs} agree as strings")

    print(f"\n  scored {sum(results)}/{len(results)} SUPPORTED")
    return 0


def _v(name, ok, detail):
    print(f"  {name}: {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    print(f"      {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.run:
        print(PREDICTIONS, flush=True)
        run()
    return score()


if __name__ == "__main__":
    raise SystemExit(main())
