"""Merge the alpha sweep runs and apply the corrected metrics.

Emits results/alpha_sweep_K15.csv (per-run) and prints the alpha x arm verdict table.
Timeouts are carried through as rows with a null formula and counted, never dropped.
"""

import argparse
import csv
import glob
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corrected_metrics import metrics  # noqa: E402

ANCHORS = ("const=NP", "const=VP")


def load(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        arm = "untrained" if base.startswith("untrained") else "trained"
        for r in csv.DictReader(open(path)):
            r["arm_label"] = arm
            rows.append(r)
    return rows


def enrich(r):
    f = lambda k: float(r[k]) if r.get(k) not in (None, "", "None") else None  # noqa: E731
    i = lambda k: int(float(r[k])) if r.get(k) not in (None, "", "None") else None  # noqa: E731
    d = f("density")
    m = metrics(d, f("formula_cov"), i("n_fires"), i("n_inter"), f("best_iou"))
    r.update(m)
    r["timed_out"] = "yes" if r.get("halted") == "time" else "no"
    fml = r.get("formula") or ""
    r["anchored"] = "yes" if any(a in fml for a in ANCHORS) else ("" if not fml else "no")
    return r


def agg(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "None")]
    return round(statistics.mean(vals), 4) if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", default="results/alpha_sweep_K15.csv")
    ap.add_argument("--phase", default="A-beam")
    args = ap.parse_args()

    rows = [enrich(r) for r in load(args.glob)]
    if not rows:
        raise SystemExit(f"no rows matched {args.glob}")
    for r in rows:
        r["phase"] = args.phase

    cols = list(rows[0].keys())
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}  ({len(rows)} runs)\n")

    print("=== PER-RUN ===")
    hdr = f"{'arm':>10} {'alpha':>6} {'unit':>8} {'dens':>7} {'fires':>6} {'IoU':>6} " \
          f"{'cov':>6} {'lift':>6} {'nfit%':>6} {'IoUind':>7} {'IoU/d':>6} " \
          f"{'peak':>7} {'vis':>6} {'time':>7} {'TO':>3} {'anch':>4}"
    print(hdr)
    for r in sorted(rows, key=lambda x: (x["arm_label"], -float(x["alpha"]), x["neuron"])):
        g = lambda k, w=6, p=3: (f"{float(r[k]):>{w}.{p}f}"  # noqa: E731
                                 if r.get(k) not in (None, "", "None") else f"{'-':>{w}}")
        nf = (f"{float(r['normalised_fit'])*100:>6.1f}"
              if r.get("normalised_fit") not in (None, "", "None") else f"{'-':>6}")
        print(f"{r['arm_label']:>10} {r['alpha']:>6} {r['neuron']:>8} {g('density',7,4)} "
              f"{r.get('n_fire_neuron',''):>6} {g('best_iou')} {g('formula_cov')} "
              f"{g('lift')} {nf} {g('iou_indep',7)} {g('iou_over_d')} "
              f"{r.get('peak_frontier',''):>7} {r.get('visited',''):>6} "
              f"{g('time_s',7,1)} {r['timed_out']:>3} {r['anchored']:>4}")

    # Nominal alpha is the knob; realised density is the quantity it controls, and the two
    # come apart per unit (tanh ties at the quantile threshold). Plot against density.
    pts = [r for r in rows if r.get("lift") not in (None, "", "None")]
    if len(pts) >= 3:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from phaseB_report import scatter
        for r in pts:
            r["arm"] = r["arm_label"]
            r["density_run"] = float(r["density"])
            r["beam_lift"] = float(r["lift"])
            r["beam_cov"] = float(r["formula_cov"])
        scatter(pts, "density_run", "beam_lift",
                "beam lift vs REALISED DENSITY (alpha is a label, not the x-axis)")
        scatter(pts, "density_run", "beam_cov",
                "beam formula coverage vs REALISED DENSITY")

    print("\n=== VERDICT TABLE: alpha x arm ===")
    print(f"{'arm':>10} {'alpha':>6} {'n':>3} {'density':>8} {'mean cov':>9} "
          f"{'mean lift':>10} {'mean nfit%':>11} {'med time':>9} {'timeouts':>9} "
          f"{'anchored':>9}")
    keys = sorted({(r["arm_label"], float(r["alpha"])) for r in rows},
                  key=lambda k: (k[0], -k[1]))
    for arm, a in keys:
        sub = [r for r in rows if r["arm_label"] == arm and float(r["alpha"]) == a]
        times = [float(r["time_s"]) for r in sub if r.get("time_s")]
        to = sum(1 for r in sub if r["timed_out"] == "yes")
        anch = sum(1 for r in sub if r["anchored"] == "yes")
        done = [r for r in sub if r["anchored"] != ""]
        nfit = agg(sub, "normalised_fit")
        print(f"{arm:>10} {a:>6} {len(sub):>3} {agg(sub,'density'):>8} "
              f"{str(agg(sub,'formula_cov')):>9} {str(agg(sub,'lift')):>10} "
              f"{(f'{nfit*100:.1f}' if nfit is not None else '-'):>11} "
              f"{(f'{statistics.median(times):.1f}' if times else '-'):>9} "
              f"{f'{to}/{len(sub)}':>9} {f'{anch}/{len(done)}':>9}")


if __name__ == "__main__":
    raise SystemExit(main())
