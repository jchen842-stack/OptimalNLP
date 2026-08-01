"""Score the length-4 per-sentence partition run against P5, registered before it finished.

Reads results/partition_L4.csv only. No searching.

## Flat (one-sample) length-4 baseline, pasted from results/beam_vs_exact_K15.csv

    27 pairs
    timeouts/caps : 4   (unit510 a=0.1, unit87 a=0.05, unit396 a=0.2-untrained, unit510 a=0.05-untrained)
    median peak_frontier : 17,714
    median time          : 718.6s
    max time             : 2,085.1s   (soft 1500s cap overshoot)

## PRE-REGISTERED — committed 2026-08-01 while the run was still in flight,
## before results/partition_L4.csv existed.

  V0  CONTROL, read FIRST. IoU is partition-invariant (Lemma 3.6). Every pair that
      TERMINATES must return the same IoU the flat run returned, to float64 equality --
      except the pairs the flat run lost to the inadmissible aggregated ceiling, which may
      now return HIGHER. A terminating pair returning LOWER means the repartition is wrong
      and nothing else may be read.

  P5  THE LENGTH-4 WALL MOVES.
      Either termination where there previously was none, or a large drop in peak frontier.
      Fixed in advance so neither disjunct can be stretched after the fact:
        (a) timeouts/caps < 4, OR
        (b) median peak_frontier < 8,857  (half the flat median of 17,714)
      SUPPORTED if (a) OR (b). Reported separately so a split result is visible.

      If P5 holds, the founding observation of diary D5.0 -- the frontier explosion that
      motivated this whole line of work -- is substantially a SAMPLE-REPRESENTATION
      ARTIFACT rather than a property of token-level concept structure. D5.0 is NOT to be
      rewritten; a superseding entry is appended.

  P6  NOT A PREDICTION, A GUARD. Any pair that terminates under the partition but returned
      a TIMEOUT flat has no flat IoU to compare, and is counted in its own bucket rather
      than scored as agreement or disagreement. Sentinels never reach a comparison
      (standing rule, this file's third instance).
"""
import csv, os, statistics as st, math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PART = os.path.join(REPO, "results", "partition_L4.csv")
FLAT = os.path.join(REPO, "results", "beam_vs_exact_K15.csv")

FLAT_TIMEOUTS = 4
FLAT_MEDIAN_PEAK = 17714.0
FLAT_MEDIAN_TIME = 718.6
PEAK_HALVED = FLAT_MEDIAN_PEAK / 2


def main():
    if not os.path.exists(PART):
        print(f"{PART} not found -- the length-4 partition run has not written yet.")
        return 2
    part = list(csv.DictReader(open(PART)))
    flat = {(r["arm"], r["alpha"], r["unit"]): r for r in csv.DictReader(open(FLAT))}

    print("=" * 92)
    print("LENGTH-4 PER-SENTENCE PARTITION vs FLAT ONE-SAMPLE  (K=15, M=24,199, 27 pairs)")
    print("=" * 92)
    print(__doc__[__doc__.index("## Flat"):__doc__.index("  P6")])

    term = [r for r in part if r["halted"] == "no"]
    halted = [r for r in part if r["halted"] != "no"]

    print(f"\n{'pair':>28} {'flat_peak':>10} {'part_peak':>10} {'flat_t':>9} {'part_t':>9} "
          f"{'flat_halt':>10} {'part_halt':>10} {'IoU':>10}")
    for r in part:
        k = (r["arm"], r["alpha"], r["unit"])
        f = flat.get(k)
        fp = float(f["peak_frontier"]) if f else float("nan")
        ft = float(f["time_s"]) if f else float("nan")
        fh = f["halted"] if f else "?"
        iou = r["part_iou"]
        print(f"{r['arm'] + ' a=' + r['alpha'] + ' ' + r['unit']:>28} {fp:>10.0f} "
              f"{float(r['peak']):>10.0f} {ft:>9.1f} {float(r['time_s']):>9.1f} "
              f"{fh:>10} {r['halted']:>10} {str(iou)[:10]:>10}")

    # ---- V0: only pairs that terminated in BOTH runs are comparable -------------------
    cmp_ok, higher, lower, nocmp = [], [], [], []
    for r in part:
        k = (r["arm"], r["alpha"], r["unit"])
        f = flat.get(k)
        pi = float(r["part_iou"]) if r["part_iou"] not in ("", "None", "nan") else float("nan")
        if f is None or f["halted"] != "no" or r["halted"] != "no" or math.isnan(pi):
            nocmp.append(k)
            continue
        fi = float(f["exact_IoU"])
        if abs(pi - fi) <= 5e-5:
            cmp_ok.append(k)
        elif pi > fi:
            higher.append((k, fi, pi))
        else:
            lower.append((k, fi, pi))

    print("\n--- V0 CONTROL ---")
    print(f"  comparable pairs (terminated in BOTH): {len(cmp_ok) + len(higher) + len(lower)}")
    print(f"    same IoU     : {len(cmp_ok)}")
    print(f"    HIGHER       : {len(higher)}  {[(k, round(a, 6), round(b, 6)) for k, a, b in higher]}")
    print(f"    LOWER        : {len(lower)}   {[(k, round(a, 6), round(b, 6)) for k, a, b in lower]}")
    print(f"  P6 bucket (not comparable -- timeout on one side, or no label): {len(nocmp)}")
    v0 = not lower
    print(f"  V0: {'PASS' if v0 else 'FAIL -- a terminating pair returned LOWER; repartition is wrong'}")
    if not v0:
        print("  Nothing below may be read.")
        return 1

    # ---- P5 --------------------------------------------------------------------------
    peaks = [float(r["peak"]) for r in part]
    med_peak = st.median(peaks)
    a = len(halted) < FLAT_TIMEOUTS
    b = med_peak < PEAK_HALVED
    print("\n--- P5 ---")
    print(f"  (a) timeouts/caps {len(halted)} < {FLAT_TIMEOUTS} ............ {'YES' if a else 'NO'}"
          + (f"   still halted: {[r['unit'] for r in halted]}" if halted else ""))
    print(f"  (b) median peak {med_peak:,.0f} < {PEAK_HALVED:,.0f} ...... {'YES' if b else 'NO'}"
          f"   (flat median {FLAT_MEDIAN_PEAK:,.0f}, ratio {med_peak / FLAT_MEDIAN_PEAK:.2f}x)")
    print(f"  P5: {'SUPPORTED' if (a or b) else 'NOT SUPPORTED'}")

    times = [float(r["time_s"]) for r in part]
    print(f"\n  median time {st.median(times):.1f}s vs flat {FLAT_MEDIAN_TIME}s "
          f"({st.median(times) / FLAT_MEDIAN_TIME:.2f}x); max {max(times):.1f}s")

    if a or b:
        print("\n  D5.0's founding frontier-explosion observation is substantially a")
        print("  SAMPLE-REPRESENTATION ARTIFACT. Append a superseding diary entry.")
        print("  DO NOT rewrite D5.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
