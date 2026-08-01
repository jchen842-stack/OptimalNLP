"""Experiment 2b — the PAPER'S beam (`beam_optimal.py`) vs exact, at length 3.

Experiment 2a swept `MAX_FRONTIER_SIZE`, which caps `optimal.py`'s A* frontier by estimated
ceiling. That is the mechanism this project has used everywhere, but it is NOT what the paper
means by "beam size 5". Upstream ships a separate algorithm:

    compositional/beam_optimal.py  ->  compute_beam_optimal_explanations

a level-wise beam over COMPLETE formulas ranked by EXACT IoU (`utils/search_utils.beam_search`
puts `(iou, label)` into a `PriorityQueue(beam_limit)`), seeded from the previous level's
best. Every leaf is a scored length-1 formula, so unlike the frontier cap it cannot return
`None` on a non-degenerate neuron.

This script runs that algorithm, unmodified, on the same 27 pairs, same corpus, same widths,
and answers the question experiment 2a could not: does the paper's own beam size 5 reproduce
the paper's own reported +5.1-6.5% gap?

Nothing upstream is edited. `BeamStubConfig` just adds the `get_beam_limit()` accessor that
`compute_beam_optimal_explanations` reads, alongside the accessors `StubConfig` already
provides.

Usage::

    python src/exp_beam_optimal.py --run
    python src/exp_beam_optimal.py --score
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

OUT = os.path.join(REPO, "results", "beam_optimal_L3.csv")

from exp_beam_width import PAIRS, WIDTHS, K, LENGTH, MAX_SENTS, MIN_SUPPORT, \
    CAP, TIME_BUDGET, VISION_BAND  # noqa: E402

# --- baselines, pasted from results/beam_width_L3_VERDICTS.txt (experiment 2a) ---------
# frontier cap, fixed 19-pair set:  beam 5 agree 0/19 roa +22.26% | beam 200 agree 16/19 roa +0.57%
# frontier cap no-solution counts:  5 -> 8/27, 10 -> 3/27, 25 -> 2/27, 50/100/200 -> 0/27
CAP_FIXEDSET_ROA = {5: 22.26, 10: 22.08, 25: 18.52, 50: 7.30, 100: 5.31, 200: 0.57}
CAP_FIXEDSET_AGREE = {5: 0, 10: 0, 25: 2, 50: 3, 100: 9, 200: 16}
CAP_NOSOL = {5: 8, 10: 3, 25: 2, 50: 0, 100: 0, 200: 0}

# =====================================================================================
# PRE-REGISTERED PREDICTIONS — committed before beam_optimal.py was run even once.
# =====================================================================================
PREDICTIONS = """
C1  THE PAPER'S BEAM ALWAYS RETURNS A FORMULA.
    Zero no-solution runs at every width including 5, against 8/27 for the frontier cap.
    Why: `beam_search` ranks COMPLETE formulas by exact IoU and every leaf is a valid
    length-1 formula, so the beam is non-empty after level 1 by construction.
    This is the structural claim that makes 2a's correction correct. If it fails, the two
    mechanisms are not as different as the correction says and that correction needs
    revisiting.
    SUPPORTED if no_solution == 0 at all six widths.

C2  BEAM 5 REPRODUCES THE PAPER'S BAND.
    Ratio-of-averages (exact over beam_optimal) at width 5, over all 27 pairs, lands
    INSIDE +5.1% to +6.5%.
    Why: this is the paper's own algorithm at the paper's own default beam size on the
    paper's own statistic. If the vision result transfers to NLI at all, this is where it
    should show up. The frontier cap gave +22.26% at width 5, so this predicts the
    mechanism difference accounts for essentially the whole discrepancy.
    SUPPORTED only if 5.1 <= roa_all(5) <= 6.5. Registered as a two-sided test on purpose:
    2a showed a one-sided threshold cannot tell "small" from "broken".

C3  THE PAPER'S BEAM DOMINATES THE FRONTIER CAP AT EVERY WIDTH.
    At each of the six widths, beam_optimal's ratio-of-averages is SMALLER (a tighter gap)
    than the frontier cap's on the same width, on the fixed 19-pair set from 2a.
    This can fail. Ranking by exact IoU is greedier and could plateau early, whereas the
    ceiling-ranked frontier is at least admissible-optimistic and might recover at wide
    settings. A failure at beam 200 in particular would mean our published beam-200 numbers
    are not the weaker approximation the correction assumes.
    SUPPORTED if roa_beamopt(w) < roa_cap(w) at all six widths.

C4  CONTROL — EXACT IS UNCHANGED.
    The exact reference re-run here reproduces the exact_IoU column of
    results/beam_vs_exact_L3_K15.csv to within 1e-6 on all 27 pairs, exactly as B4 did.
    If this drifts, the two experiments are not measured against the same reference and
    C1-C3 cannot be compared to 2a at all.

C5  CONTROL — BEAM WIDTH >= EXACT'S VISITED COUNT MUST GIVE EXACT'S ANSWER... IS NOT
    TESTABLE HERE, AND IS REGISTERED AS SUCH.
    2a's B5 passed vacuously because its antecedent was never satisfied. The analogous
    claim here would be that a beam wider than the whole level-wise search space returns
    the optimum; at K=15 length 3 that space is 30,375 formulas and the widest swept beam
    is 200, so the antecedent is again never satisfied. Registered in advance as UNTESTED
    rather than run and reported as a pass.
"""


class BeamStubConfig:
    """StubConfig plus the `get_beam_limit()` accessor beam_optimal.py reads.

    Kept here rather than added to synthetic_overlap_sweep.StubConfig so that no existing
    file changes and every published result keeps the exact config object it was run with.
    """

    def __init__(self, length, M, beam_limit):
        self._length, self._M, self._beam = length, M, beam_limit

    def get_length(self):
        return self._length

    def get_mask_shape(self):
        return (1, self._M)

    def get_device(self):
        return "cpu"

    def get_beam_limit(self):
        return self._beam


def run_beam_optimal(dense, neuron_bits, length, beam_limit, concepts, time_budget):
    """One beam_optimal run. Mirrors real_token_search.run_one's plumbing exactly, but
    dispatches to compute_beam_optimal_explanations instead of compute_optimal_explanations."""
    import heapq
    import scipy.sparse as sparse
    import torch
    from compositional import beam_optimal
    from synthetic_overlap_sweep import (HeapProbe, compute_disjoint_info,
                                         compute_quantities, _Halt)
    import real_token_search as rts

    _, M = dense.shape
    masks = [sparse.csr_matrix(dense[c].reshape(1, M)) for c in range(dense.shape[0])]
    common, unique, uncoverable, _ = compute_quantities(dense, M)
    disjoint_info = compute_disjoint_info(dense, dense.shape[0])
    bitmaps = torch.from_numpy(neuron_bits.reshape(1, M))

    probe = HeapProbe(CAP, time_budget=time_budget)
    saved = getattr(beam_optimal, "heapq", None)
    beam_optimal.heapq = probe
    cfg = BeamStubConfig(length, M, beam_limit)

    t0, halt, best_label = time.time(), "", None
    devnull = open(os.devnull, "w")
    saved_stdout = sys.stdout
    try:
        sys.stdout = devnull
        best_label, best_iou, visited, expanded, estimated = \
            beam_optimal.compute_beam_optimal_explanations(
                bitmaps=bitmaps, masks=masks, masks_info=(common, unique, uncoverable),
                disjoint_info=disjoint_info, config=cfg)
    except _Halt as h:
        halt = h.reason
        best_iou, visited, expanded, estimated = float("nan"), -1, -1, -1
    finally:
        sys.stdout = saved_stdout
        devnull.close()
        if saved is not None:
            beam_optimal.heapq = saved
        else:
            beam_optimal.heapq = heapq
        dt = time.time() - t0

    mask = None
    fstats = {"formula": None, "formula_cov": None, "n_fires": None, "n_inter": None}
    if best_label is not None:
        mask = rts.eval_formula(best_label, dense)
        s = rts.formula_stats(best_label, concepts, dense, neuron_bits)
        fstats = {k: s[k] for k in fstats}
    return {"best_iou": round(best_iou, 4) if best_iou == best_iou else None,
            "visited": visited, "expanded": expanded, "time_s": round(dt, 2),
            "peak_frontier": probe.peak, "halted": halt or "no", **fstats}, mask


def run():
    import real_token_search as rts
    import real_token_masks as rtm
    import env_info
    env_info.print_banner("beam-optimal-L3")

    feats = os.path.expanduser(
        "~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats")
    tokens = rtm.load_tokens(feats, MAX_SENTS)
    _, cats = rts.ARMS["all"]
    concepts = rtm.select_concepts(tokens, cats, K, MIN_SUPPORT)
    dense = rtm.build_dense(tokens, concepts)
    print(f"[beam-optimal] {len(tokens)} tokens | K={K} length={LENGTH} "
          f"widths={WIDTHS} | upstream compositional.beam_optimal, unmodified\n", flush=True)

    grabbed = {}
    real_formula_stats = rts.formula_stats

    def grabbing(f, c, d, nb=None):
        grabbed["mask"] = rts.eval_formula(f, d)
        return real_formula_stats(f, c, d, nb)

    rows = []
    try:
        for arm, alpha, units in PAIRS:
            acts = os.path.join(REPO, "results", f"acts2k_{arm}_a{alpha}.npz")
            picked, _, _, _ = rts.load_real_neurons(
                acts, len(units), np.random.default_rng(999), dmin=0, dmax=1,
                min_fire=200, unit_ids=units)
            for uid, neuron_bits in picked:
                # Exact reference, via the untouched optimal.py path.
                rts.formula_stats = grabbing
                grabbed.clear()
                ex = rts.run_one(dense, neuron_bits, LENGTH, CAP, TIME_BUDGET, None,
                                 concepts=concepts)
                ex_mask = grabbed.get("mask")
                rts.formula_stats = real_formula_stats

                for w in WIDTHS:
                    b, b_mask = run_beam_optimal(dense, neuron_bits, LENGTH, w,
                                                 concepts, TIME_BUDGET)
                    same = (ex_mask is not None and b_mask is not None
                            and bool(np.array_equal(ex_mask, b_mask)))
                    rows.append({
                        "arm": arm, "alpha": alpha, "unit": f"unit{uid}",
                        "density_run": round(float(neuron_bits.mean()), 5),
                        "beam": w, "algo": "beam_optimal",
                        "beam_IoU": b["best_iou"], "exact_IoU": ex["best_iou"],
                        "beam_cov": b["formula_cov"], "exact_cov": ex["formula_cov"],
                        "no_solution": int(b["formula"] is None),
                        "same_mask": int(same),
                        "beam_visited": b["visited"], "exact_visited": ex["visited"],
                        "beam_expanded": b["expanded"], "exact_expanded": ex["expanded"],
                        "beam_time_s": b["time_s"], "exact_time_s": ex["time_s"],
                        "beam_halted": b["halted"], "exact_halted": ex["halted"],
                        "beam_formula": b["formula"], "exact_formula": ex["formula"],
                    })
                    print(f"  {arm} a={alpha} unit{uid} beam={w}: IoU={b['best_iou']} "
                          f"exact={ex['best_iou']} same={int(same)} "
                          f"nosol={int(b['formula'] is None)} t={b['time_s']}s", flush=True)
    finally:
        rts.formula_stats = real_formula_stats

    with open(OUT, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {OUT}  ({len(rows)} rows)")


def score():
    with open(OUT) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["beam"] = int(r["beam"])
        r["no_solution"] = int(r["no_solution"])
        r["same_mask"] = int(r["same_mask"])
        for k in ("beam_IoU", "exact_IoU"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else float("nan")

    by_w = {w: [r for r in rows if r["beam"] == w] for w in WIDTHS}
    n = len(by_w[WIDTHS[0]])

    def roa(pool):
        se, sb = sum(r["exact_IoU"] for r in pool), sum(r["beam_IoU"] for r in pool)
        return (se / sb - 1.0) * 100 if sb else float("nan")

    print("=" * 92)
    print(f"EXPERIMENT 2b — THE PAPER'S BEAM (compositional/beam_optimal.py) vs EXACT, "
          f"length {LENGTH}, K={K}, {n} pairs")
    print("=" * 92)
    print("\n--- PRE-REGISTERED PREDICTIONS ---")
    print(PREDICTIONS)

    print("--- beam_optimal vs FRONTIER CAP (2a), side by side ---")
    print(f"    {'beam':>6} | {'bo_agree':>9} {'bo_roa%':>9} {'bo_nosol':>9} {'band?':>6} "
          f"| {'cap_agree':>10} {'cap_roa%':>9} {'cap_nosol':>10}")
    for w in WIDTHS:
        pool = by_w[w]
        ra = roa(pool)
        band = "IN" if VISION_BAND[0] <= ra <= VISION_BAND[1] else "out"
        print(f"    {w:>6} | {sum(r['same_mask'] for r in pool):>5}/{len(pool):<3} "
              f"{ra:>9.2f} {sum(r['no_solution'] for r in pool):>6}/{len(pool):<3} "
              f"{band:>6} | {CAP_FIXEDSET_AGREE[w]:>6}/19  {CAP_FIXEDSET_ROA[w]:>9.2f} "
              f"{CAP_NOSOL[w]:>7}/27")
    print("    cap columns are 2a's FIXED 19-pair set (its only valid statistic); "
          "bo columns are all 27.")

    print("\n--- STRATIFIED WITHIN (arm, alpha) — beam_optimal ratio-of-averages % ---")
    strata = sorted({(r["arm"], r["alpha"]) for r in rows},
                    key=lambda s: (s[0], -float(s[1])))
    print(f"    {'stratum':>20} {'n':>3} " + " ".join(f"{('b' + str(w)):>7}" for w in WIDTHS))
    for arm, alpha in strata:
        sub = {w: [r for r in by_w[w] if r["arm"] == arm and r["alpha"] == alpha]
               for w in WIDTHS}
        print(f"    {arm + ' a=' + alpha:>20} {len(sub[WIDTHS[0]]):>3} "
              + " ".join(f"{roa(sub[w]):>7.2f}" for w in WIDTHS))

    print("\n--- VERDICTS ---")
    res = []
    total_nosol = sum(r["no_solution"] for r in rows)
    res.append(_v("C1", total_nosol == 0,
                  f"{total_nosol} no-solution runs across {len(rows)} "
                  f"(pair, width) runs; frontier cap had {sum(CAP_NOSOL.values())}"))

    ra5 = roa(by_w[5])
    res.append(_v("C2", VISION_BAND[0] <= ra5 <= VISION_BAND[1],
                  f"roa_all(beam 5) = {ra5:+.2f}% vs band {VISION_BAND[0]}-{VISION_BAND[1]}%"
                  f" -> {'INSIDE' if VISION_BAND[0] <= ra5 <= VISION_BAND[1] else ('BELOW' if ra5 < VISION_BAND[0] else 'ABOVE')}"
                  f"; frontier cap at width 5 gave {CAP_FIXEDSET_ROA[5]:+.2f}%"))

    worse = [w for w in WIDTHS if not (roa(by_w[w]) < CAP_FIXEDSET_ROA[w])]
    res.append(_v("C3", not worse,
                  "beam_optimal roa vs cap roa by width: "
                  + ", ".join(f"{w}:{roa(by_w[w]):+.2f} vs {CAP_FIXEDSET_ROA[w]:+.2f}"
                              for w in WIDTHS)
                  + (f"; NOT tighter at {worse}" if worse else "; tighter at all six")))

    ref = {}
    with open(os.path.join(REPO, "results", "beam_vs_exact_L3_K15.csv")) as f:
        for r in csv.DictReader(f):
            ref[(r["arm"], r["alpha"], r["unit"])] = float(r["exact_IoU"])
    bad = [k for r in by_w[200]
           for k in [(r["arm"], r["alpha"], r["unit"])]
           if k in ref and abs(ref[k] - r["exact_IoU"]) > 1e-6]
    res.append(_v("C4", not bad and len(by_w[200]) == n,
                  f"{n} pairs checked against the committed exact_IoU; {len(bad)} mismatches"))

    print("  C5: UNTESTED BY CONSTRUCTION (registered as such in advance)")
    print(f"      widest swept beam is {max(WIDTHS)}; the level-wise space at K={K} "
          f"length {LENGTH} is {K * (3 * K) ** (LENGTH - 1):,} formulas, so the antecedent "
          f"is never satisfied. Not counted either way.")

    print(f"\n  scored {sum(res)}/{len(res)} SUPPORTED (C5 untested, excluded)")
    return 0


def _v(name, ok, detail):
    print(f"  {name}: {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    print(f"      {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.run:
        print(PREDICTIONS, flush=True)
        run()
    return score()


if __name__ == "__main__":
    raise SystemExit(main())
