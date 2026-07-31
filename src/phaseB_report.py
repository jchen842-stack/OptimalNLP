"""Phase B: beam vs exact, side by side, per (arm, alpha, unit).

Three things this file is careful about, each because getting them wrong has already
produced a wrong answer in this project:

1. Density is taken PER RUN, never per arm. tanh saturation puts ties at the quantile
   threshold, so a unit's realised density drifts below its nominal alpha (unit 86 sits at
   0.036 when alpha=0.05). Lift divides by density, so an arm mean makes lifts that cannot
   be reconciled with their own IoU.
2. Search cost is scored on VISITED / EXPANDED, not peak frontier. Peak frontier is how
   many nodes are held at once; visited is how many are actually explored. They point
   opposite ways here, and wall time tracks visited.
3. Comparisons are WITHIN UNIT across alpha. Comparing different units across alpha
   confounds unit-level difficulty with the alpha effect.
"""

import argparse
import csv
import glob
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corrected_metrics import metrics  # noqa: E402


def load(pattern):
    out = {}
    for path in sorted(glob.glob(pattern)):
        arm = "untrained" if os.path.basename(path).startswith("untrained") else "trained"
        for r in csv.DictReader(open(path)):
            f = lambda k: (float(r[k]) if r.get(k) not in (None, "", "None") else None)  # noqa: E731
            i = lambda k: (int(float(r[k])) if r.get(k) not in (None, "", "None") else None)  # noqa: E731
            r["arm_label"] = arm
            r.update(metrics(f("density"), f("formula_cov"), i("n_fires"),
                             i("n_inter"), f("best_iou"), i("n_fire_neuron")))
            out[(arm, round(float(r["alpha"]), 4), r["neuron"])] = r
    return out


def scatter(rows, xk, yk, title, w=56, h=16):
    """ASCII scatter, points labelled by arm so alpha stays a label not an axis."""
    xs = [r[xk] for r in rows]
    ys = [r[yk] for r in rows]
    if not xs:
        return
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    grid = [[" "] * w for _ in range(h)]
    for r, x, y in zip(rows, xs, ys):
        cx = int((x - x0) / (x1 - x0) * (w - 1)) if x1 > x0 else 0
        cy = int((y - y0) / (y1 - y0) * (h - 1)) if y1 > y0 else 0
        mark = "t" if r["arm"] == "trained" else "u"
        cur = grid[h - 1 - cy][cx]
        grid[h - 1 - cy][cx] = "#" if cur != " " else mark
    print(f"\n  {title}")
    print(f"  y={yk} [{y0:.3g} .. {y1:.3g}]   x={xk} [{x0:.3g} .. {x1:.3g}]   "
          f"(t=trained, u=untrained, #=overlap)")
    for i, line in enumerate(grid):
        lab = f"{y1:>9.3g}" if i == 0 else (f"{y0:>9.3g}" if i == h - 1 else " " * 9)
        print(f"  {lab} |{''.join(line)}|")
    print(f"  {' ' * 9} +{'-' * w}+")
    print(f"  {' ' * 10}{x0:<.3g}{' ' * max(w - 12, 1)}{x1:>.3g}")


def _norm_ranks(vals):
    """Within-stratum ranks, centred and scaled so strata of unequal size pool fairly.

    Raw ranks cannot be pooled across strata of different n (3, 4 and 5 here) -- a rank of
    3 means "middle" in one stratum and "top" in another. Centring on (n+1)/2 and dividing
    by n puts every stratum on the same [-0.5, 0.5] scale with mean 0.
    """
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:  # average ranks over ties
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return [(r - (n + 1) / 2.0) / n for r in ranks]


def score_prediction(rows, xk, yk, title, stratify=("arm", "alpha")):
    """Stratified rank correlation with an explicit SUPPORTED / NOT SUPPORTED verdict.

    Why stratified rather than pooled: alpha drives lift and density together
    (rho(density, lift) = -1.00 trained / -0.80 untrained in Phase A), so a POOLED
    rho(time, lift) is just rho(time, density) with the sign flipped and cannot separate
    the two. The registered mechanism -- a strong incumbent prunes harder -- is a claim
    about differences BETWEEN UNITS AT FIXED DENSITY, so the alpha axis has to be removed
    by construction. Ranking within each (arm, alpha) stratum does that; pooling the
    within-stratum ranks recovers the power that six separate n=5 tests would lose.
    """
    print(f"\n=== {title} ===")
    strata = {}
    for r in rows:
        strata.setdefault(tuple(r[k] for k in stratify), []).append(r)

    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("  scipy unavailable; correlation skipped")
        return

    print(f"  per-stratum rho({yk}, {xk}):")
    px, py = [], []
    for key in sorted(strata, key=lambda k: (k[0], -float(k[1]))):
        sub = strata[key]
        if len(sub) < 3:
            print(f"    {key[0]:>10} alpha={key[1]:<6} n={len(sub)}  (too small to rank)")
            continue
        rx = _norm_ranks([r[xk] for r in sub])
        ry = _norm_ranks([r[yk] for r in sub])
        px.extend(rx)
        py.extend(ry)
        srho, sp = spearmanr([r[xk] for r in sub], [r[yk] for r in sub])
        print(f"    {key[0]:>10} alpha={key[1]:<6} n={len(sub)}  rho={srho:+.3f}  p={sp:.3f}")

    if len(px) < 3:
        print("  too few strata to pool")
        return
    rho, _ = spearmanr(px, py)  # asymptotic p deliberately discarded, see below

    # p by permuting x WITHIN each stratum. The centring leaves unequal spread across
    # unequal strata (+-0.400 at n=5, +-0.375 at n=4, +-0.333 at n=3), so the small
    # stratum is down-weighted ~17%; permuting reproduces that weighting exactly under the
    # null instead of assuming it away. The unequal weighting is the reason -- NOT any
    # small-n failure of the asymptotic p, which matches the permutation p closely once
    # compared one-sided to one-sided.
    usable = [strata[k] for k in sorted(strata, key=lambda k: (k[0], -float(k[1])))
              if len(strata[k]) >= 3]
    rng = random.Random(0)
    n_perm, hits = 10000, 0
    for _ in range(n_perm):
        qx, qy = [], []
        for sub in usable:
            xv = [r[xk] for r in sub]
            rng.shuffle(xv)
            qx.extend(_norm_ranks(xv))
            qy.extend(_norm_ranks([r[yk] for r in sub]))
        prho, _ = spearmanr(qx, qy)
        if prho <= rho:
            hits += 1
    p = (hits + 1) / (n_perm + 1)  # add-one keeps p > 0 and is unbiased

    ok = rho < 0 and p < 0.05
    print(f"\n  STRATIFIED (headline): rho = {rho:+.3f}  n = {len(px)} "
          f"across {len(usable)} strata")
    # Only the one-sided permutation p is printed. Spearman's asymptotic p is TWO-SIDED,
    # so printing it alongside invites a like-for-like reading of two numbers that differ
    # by a factor of ~2 purely by convention -- and the loose-quotable one reads high.
    print(f"  p = {p:.4f}  (one-sided, {n_perm} within-stratum permutations of {xk})")
    print(f"  prediction rho < 0 -> {'SUPPORTED' if ok else 'NOT SUPPORTED'} "
          f"({'negative' if rho < 0 else 'positive'} rank correlation, "
          f"{'significant' if p < 0.05 else 'not significant'} at 0.05)")

    prho, _ = spearmanr([r[xk] for r in rows], [r[yk] for r in rows])
    print(f"  [CONFOUNDED, not the headline] pooled rho = {prho:+.3f} (no p reported) -- "
          f"alpha moves lift and density together, so this is rho({yk}, density) "
          f"sign-flipped and cannot separate the two.")

    scatter(rows, xk, yk, f"{yk} vs {xk} (pooled view; scoring is stratified)")


def num(r, k):
    if r is None or r.get(k) in (None, "", "None"):
        return None
    return float(r[k])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beam_glob", required=True)
    ap.add_argument("--exact_glob", required=True)
    ap.add_argument("--out", default="results/beam_vs_exact_K15.csv")
    args = ap.parse_args()

    beam, exact = load(args.beam_glob), load(args.exact_glob)

    rows = []
    for key in sorted(exact, key=lambda k: (k[0], -k[1], k[2])):
        arm, a, unit = key
        e, b = exact[key], beam.get(key)
        bl, el = num(b, "lift"), num(e, "lift")
        gain = ((el - bl) / bl * 100) if (bl and el) else None
        # The paper's Table 4 band (+5.1%..+6.5%) is an IoU improvement, so the IoU-based
        # gain is the like-for-like number to compare against. The lift-based gain is what
        # the registered predictions are scored on. They are not interchangeable: lift
        # divides by density, so the two differ whenever beam and exact pick formulas of
        # different coverage.
        bi, ei = num(b, "best_iou"), num(e, "best_iou")
        gain_iou = ((ei - bi) / bi * 100) if (bi and ei) else None
        rows.append({
            "arm": arm, "alpha": a, "unit": unit,
            "density_run": num(e, "density"), "n_fire_neuron": e.get("n_fire_neuron"),
            "beam_IoU": num(b, "best_iou"), "exact_IoU": num(e, "best_iou"),
            "beam_cov": num(b, "formula_cov"), "exact_cov": num(e, "formula_cov"),
            "beam_lift": bl, "exact_lift": el,
            "rel_gain_lift_pct": round(gain, 2) if gain is not None else None,
            "rel_gain_iou_pct": round(gain_iou, 2) if gain_iou is not None else None,
            # The entire cause of a negative lift-gain: exact can buy IoU by widening
            # coverage, trading precision for recall. Without this ratio the divergence
            # between the two gap definitions is invisible.
            # None on a timeout: an exact run that hit the cap has no formula, so it has no
            # coverage, no gap and no usable time. Those rows stay in the table as
            # explicit timeouts and are excluded from every derived statistic.
            "cov_ratio_exact_over_beam": (
                round(num(e, "formula_cov") / num(b, "formula_cov"), 4)
                if (b and num(b, "formula_cov") and num(e, "formula_cov")) else None),
            "beam_precision": num(b, "precision"), "exact_precision": num(e, "precision"),
            "beam_recall": num(b, "recall"), "exact_recall": num(e, "recall"),
            "beam_n_fires": num(b, "n_fires"), "exact_n_fires": num(e, "n_fires"),
            "beam_n_inter": num(b, "n_inter"), "exact_n_inter": num(e, "n_inter"),
            "exact_nfit": num(e, "normalised_fit"), "beam_nfit": num(b, "normalised_fit"),
            "visited": num(e, "visited"), "expanded": num(e, "expanded"),
            "peak_frontier": num(e, "peak_frontier"), "time_s": num(e, "time_s"),
            "halted": e.get("halted"), "exact_formula": e.get("formula"),
            "beam_formula": b.get("formula") if b else None,
        })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}  ({len(rows)} pairs)\n")

    print("=== BEAM vs EXACT (per run; density is per-run realised, not arm mean) ===")
    print(f"{'arm':>10} {'alpha':>6} {'unit':>8} {'d_run':>7} "
          f"{'beamIoU':>8} {'exIoU':>7} {'gIoU%':>7} {'gLift%':>7} {'covR':>6} "
          f"{'bPrec':>6} {'ePrec':>6} {'bRec':>6} {'eRec':>6} "
          f"{'visited':>8} {'expand':>7} {'time':>7} {'halt':>5}")
    for r in rows:
        g = lambda k, w=7, p=3: (f"{r[k]:>{w}.{p}f}" if r[k] is not None else f"{'-':>{w}}")  # noqa: E731
        print(f"{r['arm']:>10} {r['alpha']:>6} {r['unit']:>8} {g('density_run')} "
              f"{g('beam_IoU',8)} {g('exact_IoU')} {g('rel_gain_iou_pct',7,2)} "
              f"{g('rel_gain_lift_pct',7,2)} {g('cov_ratio_exact_over_beam',6,2)} "
              f"{g('beam_precision',6)} {g('exact_precision',6)} "
              f"{g('beam_recall',6)} {g('exact_recall',6)} "
              f"{str(int(r['visited'])) if r['visited'] else '-':>8} "
              f"{str(int(r['expanded'])) if r['expanded'] else '-':>7} "
              f"{g('time_s',7,1)} {str(r['halted']):>5}")

    # Magnitude is a headline in its own right, separate from the two ORDER predictions:
    # neither says anything about the SIZE of the gap.
    # RATIO OF AVERAGES is the only figure comparable to the paper's Table 4 band. Table 4
    # reports "Avg. IoU per category", i.e. (avg exact IoU)/(avg beam IoU) - 1. Mean and
    # median of per-unit ratios are AVERAGES OF RATIOS -- a different statistic, and with a
    # right tail they do not agree. The band verdict tests this one; picking the median
    # after seeing the mean fall outside the band would be post-hoc even though both were
    # pre-registered.
    print("\n=== BAND COMPARISON: ratio of averages (the like-for-like statistic) ===")
    paired = [r for r in rows
              if r["beam_IoU"] is not None and r["exact_IoU"] is not None]
    # Table 4's caption restricts to "all the units in the layer for which the optimal and
    # the beam search find two different solutions" -- agreeing units are excluded from
    # their average, so (b) is the like-for-like set. Formulas are compared as rendered
    # strings; a re-associated but logically equivalent formula would be miscounted as
    # differing, so the identical-IoU count is printed as a cross-check.
    # "Different solutions" means a different MASK, not a different parse tree. Two
    # formulas can be syntactically distinct yet fire on exactly the same tokens on this
    # corpus (untrained unit510 at a=0.1: the two differ only where lemma=. AND const=VP,
    # which no token satisfies). Extensional identity is tested on (n_fires, n_inter),
    # which pins |F| and |F&N| and hence IoU; the syntactic count is printed alongside.
    def same_solution(r):
        return (r["beam_n_fires"] == r["exact_n_fires"]
                and r["beam_n_inter"] == r["exact_n_inter"])

    differing = [r for r in paired if not same_solution(r)]
    syntactic_same = sum(1 for r in paired if r["exact_formula"] == r["beam_formula"])
    same_iou = sum(1 for r in paired if abs(r["exact_IoU"] - r["beam_IoU"]) < 1e-9)

    def roa_block(tag, pool):
        for label, sub in (("pooled", pool),
                           ("trained", [r for r in pool if r["arm"] == "trained"]),
                           ("untrained", [r for r in pool if r["arm"] == "untrained"])):
            if not sub:
                continue
            se, sb = sum(r["exact_IoU"] for r in sub), sum(r["beam_IoU"] for r in sub)
            roa = (se / sb - 1.0) * 100
            note = ""
            if label == "pooled":
                note = ("  <- WITHIN the vision band" if 5.1 <= roa <= 6.5
                        else "  <- OUTSIDE the vision band")
            print(f"  {tag} {label:>10} n={len(sub):>2}  sum(exact_IoU)={se:.4f} "
                  f"sum(beam_IoU)={sb:.4f}  ratio-of-averages = {roa:+.2f}%{note}")

    print("  (a) ALL pairs:")
    roa_block("     ", paired)
    print(f"\n  (b) RESTRICTED to pairs whose returned FORMULAS differ "
          f"[LIKE-FOR-LIKE with Table 4's caption] -- "
          f"dropped {len(paired) - len(differing)} of {len(paired)}:")
    roa_block("     ", differing)
    print(f"\n  cross-check: {len(paired)-len(differing)} extensionally identical "
          f"(same n_fires AND n_inter); {syntactic_same} syntactically identical; "
          f"{same_iou} with identical IoU to 1e-9")
    print("  paper Table 4 (vision): +5.1% .. +6.5%, max length 3, beam size 5 "
          "(Appendix B); ours is length 4, beam 200")

    print("\n=== OPTIMALITY GAP MAGNITUDE (averages of ratios; NOT band-comparable) ===")
    for key, lab, cmp_paper in (
            ("rel_gain_iou_pct", "IoU-based (per-unit ratios)", True),
            ("rel_gain_lift_pct", "lift-based (what predictions #1/#2 are scored on)", False)):
        g = [r[key] for r in rows if r[key] is not None]
        if not g:
            continue
        print(f"  {lab}, n={len(g)}: mean {statistics.mean(g):+.2f}%  "
              f"median {statistics.median(g):+.2f}%  "
              f"range {min(g):+.2f}% .. {max(g):+.2f}%")
        print(f"    per-unit: {', '.join(f'{v:+.1f}' for v in sorted(g))}")
        if cmp_paper:
            m = statistics.mean(g)
            med = statistics.median(g)
            print(f"    mean {m:+.2f}% vs median {med:+.2f}% -- "
                  f"{'mean is skewed by outliers' if abs(m-med) > 1.0 else 'mean and median agree'}")
            print(f"    NOT compared to the vision band here -- see the ratio-of-averages "
                  f"block above, which is the like-for-like statistic.")

    # Timeouts carry a censored time and no formula, so they are excluded from scoring
    # rather than treated as slow-but-finished runs.
    scored = [r for r in rows
              if r["rel_gain_lift_pct"] is not None and r["exact_lift"] is not None
              and r["time_s"] is not None and r["halted"] != "time"]
    if len(scored) >= 3:
        # Both pre-registered predictions share a mechanism: a strong incumbent found early
        # prunes harder. If it holds, the gap AND the time both fall as lift rises.
        # #1 is scored TWICE. IoU-gain is a correction to how the registered concept was
        # OPERATIONALISED (it is the actual optimality gap and what the paper reports);
        # lift-gain is the literal registered quantity. Both verdicts are printed because
        # the substitution cannot be claimed as original intent. Disagreement between them
        # is itself a result: it means beam and exact differ in the KIND of formula found.
        score_prediction([r for r in scored if r["rel_gain_iou_pct"] is not None],
                         "exact_lift", "rel_gain_iou_pct",
                         "PRE-REGISTERED #1 PRIMARY: IoU optimality gap vs lift")
        score_prediction(scored, "exact_lift", "rel_gain_lift_pct",
                         "PRE-REGISTERED #1 SECONDARY (as registered): lift gap vs lift")
        score_prediction(scored, "exact_lift", "time_s",
                         "PRE-REGISTERED #2: exact search time vs lift")

    # Realised density, not nominal alpha, is the quantity alpha is a knob for; they come
    # apart per unit (unit 88 realises 0.129 at a nominal 0.2). Plot against the quantity.
    # Flagged for the write-up, NOT for inference from a single unit: exact widening
    # coverage relative to beam is the alpha=0.5 blanket problem in miniature, and exact is
    # MORE exposed to it than beam because it actually reaches the IoU optimum. If this is
    # systematically > 1 across all 27 it is a finding about the OBJECTIVE, not our units.
    cr = [r["cov_ratio_exact_over_beam"] for r in rows
          if r["cov_ratio_exact_over_beam"] is not None]
    if cr:
        above = sum(1 for v in cr if v > 1.0)
        print(f"\n=== COVERAGE RATIO exact/beam (write-up flag, n={len(cr)}) ===")
        print(f"  mean {statistics.mean(cr):.3f}  median {statistics.median(cr):.3f}  "
              f"range {min(cr):.3f} .. {max(cr):.3f}")
        print(f"  above 1.0: {above}/{len(cr)}  "
              f"-- systematic widening requires the full n=27, do not infer from one unit")

    dens = [r for r in rows if r["density_run"] is not None
            and r["exact_lift"] is not None and r["halted"] != "time"]
    if len(dens) >= 3:
        scatter(dens, "density_run", "exact_lift",
                "lift vs REALISED DENSITY (alpha is a label, not the x-axis)")
        scatter([r for r in dens if r["time_s"] is not None], "density_run", "time_s",
                "exact time vs REALISED DENSITY")

    print("\n=== WITHIN-UNIT SERIES across alpha (Q3 scored on visited/expanded) ===")
    for arm in ("untrained", "trained"):
        units = sorted({r["unit"] for r in rows if r["arm"] == arm})
        alphas = sorted({r["alpha"] for r in rows if r["arm"] == arm}, reverse=True)
        for u in units:
            series = [next((r for r in rows if r["arm"] == arm and r["unit"] == u
                            and r["alpha"] == a), None) for a in alphas]
            if sum(x is not None for x in series) < 2:
                continue
            cells = []
            for a, r in zip(alphas, series):
                if r is None:
                    cells.append(f"a={a}: --")
                elif r["halted"] == "time" or r["visited"] is None:
                    cells.append(f"a={a}: TIMEOUT at t={r['time_s']:.0f}s")
                else:
                    cells.append(f"a={a}: vis={int(r['visited']):>5} "
                                 f"exp={int(r['expanded']):>5} t={r['time_s']:>6.1f}s "
                                 f"lift={r['exact_lift']:.2f}")
            print(f"  {arm:>10} {u:>8} | " + " | ".join(cells))


if __name__ == "__main__":
    raise SystemExit(main())
