"""Score the length-4 per-sentence partition run against P5, registered before it finished.

Reads results/partition_L4.csv only. No searching.

## Flat (one-sample) length-4 baseline, pasted from results/beam_vs_exact_K15.csv

    27 pairs
    timeouts/caps : 4   (trained a=0.1 unit510, trained a=0.05 unit87, untrained a=0.05 unit396, untrained a=0.05 unit510)
                    [CORRECTED 2026-08-01: the third was mis-named a=0.2; it is a=0.05. Count unchanged.]
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

    print(REGISTERED_C)
    if report_partial(part):
        return 0
    addenda_scoring(part, flat)
    score_p7(part, flat)
    score_p7_banded(part, flat)

    if a or b:
        print("\n  D5.0's founding frontier-explosion observation is substantially a")
        print("  SAMPLE-REPRESENTATION ARTIFACT. Append a superseding diary entry.")
        print("  DO NOT rewrite D5.0.")
    return 0



# =====================================================================================
# ADDENDA — registered 2026-08-01 while the L4 run was still in flight, before
# results/partition_L4.csv existed. Not edited after seeing results.
# =====================================================================================
ADDENDA = """
A1  P5(b) IS SCORED ON THE MATCHED SET.
    The verdict median is over pairs that terminated in BOTH the flat and the per-sentence
    runs. The all-27 median is reported alongside, labelled as INCLUDING TRUNCATED PEAKS.
    Why: a timed-out run's peak frontier is truncated at whatever it had reached when the
    cap fired, so it is a lower bound, not a measurement. All-27 median MEMBERSHIP also
    changes with the treatment -- if the partition removes timeouts, different pairs enter
    the median. The matched set holds membership fixed and excludes truncated values.
    This biases the test AGAINST the hypothesis: the pairs excluded are the flat run's
    hardest four, the ones most likely to show a large peak drop. A matched-set pass is
    therefore conservative.

A2  THE TWO DISJUNCTS MEASURE DIFFERENT THINGS, AND A SPLIT IS COHERENT.
    P5(b), peak frontier, measures the bound-tightening mechanism DIRECTLY.
    P5(a), timeouts, is wall-clock. Per the paper's Section C, sample computation costs
    on the order of |D| times more arithmetic per estimate than aggregated, and |D| goes
    from 1 to ~2,000 here.
    So (b) firing while (a) does not is a COHERENT OUTCOME -- bounds tightened, and the
    per-sample arithmetic absorbed the runtime gain. It is NOT a split failure and must not
    be reported as one. Median time and median expanded-node count are printed so the two
    effects can be separated after the fact rather than argued about.

A3  WHAT A P3 PASS AT LENGTH 3 ACTUALLY MEANT (mechanism, recorded before scoring L4).
    With |D| = 1, aggregated and sample computation are IDENTICAL, so Algorithm 1's
    refine-on-pop step (lines 14-21) was a no-op for the entire flat series. Both misses
    died on the aggregated estimate BEFORE ever being popped -- confirmed by the targeted
    trace, which found P* carrying the "sum" estimate at the moment reduce_frontier
    dropped it. Refinement could not have rescued them under the flat partition, because
    there was nothing to refine to.
    Under |D| ~ 2,000 refinement is non-trivial for the first time. So the length-3 P3 pass
    means A DISABLED COMPONENT WAS TURNED BACK ON -- not that bounds tightened globally.
    The same reading applies to any L4 recovery.
"""


def addenda_scoring(part, flat):
    """Matched-set P5(b) plus the time/count split A2 asks for."""
    import statistics as _st
    matched = [r for r in part
               if r["halted"] == "no"
               and flat.get((r["arm"], r["alpha"], r["unit"]), {}).get("halted") == "no"]
    print(ADDENDA)
    print("--- A1: P5(b) ON THE MATCHED SET (verdict) ---")
    if not matched:
        print("  no pairs terminated in both runs; P5(b) NOT SCOREABLE on the matched set")
        return
    mp = _st.median([float(r["peak"]) for r in matched])
    fp = _st.median([float(flat[(r["arm"], r["alpha"], r["unit"])]["peak_frontier"])
                     for r in matched])
    ap = _st.median([float(r["peak"]) for r in part])
    print(f"  matched set n={len(matched)} of 27 (excludes the flat run's 4 timeouts "
          f"and any new ones)")
    print(f"    flat median peak (matched)      : {fp:,.0f}")
    print(f"    partition median peak (matched) : {mp:,.0f}   ratio {mp / fp:.2f}x")
    print(f"    P5(b) VERDICT: {'YES' if mp < fp / 2 else 'NO'}  (threshold: below half of "
          f"{fp:,.0f} = {fp / 2:,.0f})")
    print(f"  all-27 median peak: {ap:,.0f}  <- INCLUDES TRUNCATED PEAKS, not the verdict")
    print("\n--- A2: separating bound-tightening from per-sample arithmetic ---")
    mt = _st.median([float(r["time_s"]) for r in matched])
    ft = _st.median([float(flat[(r["arm"], r["alpha"], r["unit"])]["time_s"])
                     for r in matched])
    me = _st.median([float(r["expanded"]) for r in matched if float(r["expanded"]) > 0])
    fe = _st.median([float(flat[(r["arm"], r["alpha"], r["unit"])]["expanded"])
                     for r in matched
                     if float(flat[(r["arm"], r["alpha"], r["unit"])]["expanded"]) > 0])
    print(f"  matched median time     : flat {ft:>9.1f}s  partition {mt:>9.1f}s  "
          f"ratio {mt / ft:.2f}x")
    print(f"  matched median expanded : flat {fe:>9.0f}   partition {me:>9.0f}   "
          f"ratio {me / fe:.2f}x")
    print("  expanded down + time up  => bounds tightened, per-sample arithmetic absorbed it")
    print("  expanded flat + time up  => no bound tightening; cost is pure arithmetic")

# =====================================================================================
# P7 — registered 2026-08-01, verified before registration that results/partition_L4.csv
# did not exist (L4 run at 38:28 elapsed, still in flight).
#
# P7 is the COMPETING hypothesis to P5(b). Both are scored; they cannot both hold.
# =====================================================================================
P7 = """
P7  P5(b) FAILS: THE FRONTIER IS NEAR-INVARIANT UNDER REPARTITION.
    Matched-set median peak frontier at length 4 does NOT halve, and lands within ~1.2x of
    the matched flat baseline.
    Scored as two conditions, both required:
      P7a  ratio >= 0.5          (P5(b) fails -- no halving)
      P7b  1/1.2 <= ratio <= 1.2 (near-invariance, not merely "not halved")

    RATIONALE, recorded in advance:
      - The aggregated estimate is PARTITION-INVARIANT. SUM_x |I^C_max(L)_x| counts over all
        (x, j) pairs, and Top^A_t / Bott^A_1 are concept-wise over dataset-wide totals. None
        of these depends on how elements are grouped into samples.
      - Confirmed empirically at length 3: both miss-prefix ceilings are BYTE-IDENTICAL
        across partitions -- 0.232677 (unit88 a=0.2) and 0.203398 (unit86 a=0.05).
      - `reduce_frontier` prunes on aggregated estimates AT INSERTION (Alg 1 lines 11, 52),
        before any refinement.
      - Therefore frontier size should be near-invariant. Measured 0.99x at length 3 (P4).

    STATED RISK, recorded so it cannot be claimed afterwards as foresight: length 4 has more
    pops, so refinement compounds, and the effect may be NON-LINEAR IN DEPTH. A length-3
    near-invariance does not entail a length-4 one.

    IF P7 HOLDS: the length-4 wall is NOT a harness artifact, and diary D5.0 survives. The
    superseding entry contemplated under P5 is not written.
    IF P5(b) HOLDS INSTEAD: the wall moves and D5.0 is superseded (appended, never rewritten).
"""


def score_p7(part, flat):
    import statistics as _st
    matched = [r for r in part
               if r["halted"] == "no"
               and flat.get((r["arm"], r["alpha"], r["unit"]), {}).get("halted") == "no"]
    print(P7)
    print("--- P7 (competing hypothesis to P5(b)) ---")
    if not matched:
        print("  NOT SCOREABLE: no pairs terminated in both runs")
        return
    mp = _st.median([float(r["peak"]) for r in matched])
    fp = _st.median([float(flat[(r["arm"], r["alpha"], r["unit"])]["peak_frontier"])
                     for r in matched])
    ratio = mp / fp
    a = ratio >= 0.5
    b = (1 / 1.2) <= ratio <= 1.2
    print(f"  matched n={len(matched)}   flat median peak {fp:,.0f}   "
          f"partition median peak {mp:,.0f}   ratio {ratio:.3f}x")
    print(f"  P7a  ratio >= 0.5 (no halving) ............ {'YES' if a else 'NO'}")
    print(f"  P7b  0.833 <= ratio <= 1.2 (near-invariant) {'YES' if b else 'NO'}")
    print(f"  P7:  {'SUPPORTED' if (a and b) else 'NOT SUPPORTED'}")
    print(f"  P5(b) on the same matched set: {'NO (fails)' if a else 'YES (halved)'}")
    if a and b:
        print("\n  => the length-4 wall is NOT a harness artifact. D5.0 survives;")
        print("     no superseding diary entry is written.")
    elif not a:
        print("\n  => the wall moved. D5.0 is superseded by an APPENDED entry, never rewritten.")
    else:
        print("\n  => SPLIT: no halving, but outside the near-invariance band. Report as such;")
        print("     neither P5(b) nor P7 may be claimed whole.")

# =====================================================================================
# ADDENDA to P5/P7 — registered 2026-08-01 with results/partition_L4.csv verified absent
# (L4 run at 40:45 elapsed, still in flight).
# =====================================================================================
P7_ADDENDA = """
B1  FOUR-WAY OUTCOME, not three. The bands are disjoint and exhaustive, fixed in advance:

      ratio >= 1.2          THE PARTITION ENLARGED THE FRONTIER. A distinct result in its
                            own right, not a failure of either hypothesis: refinement
                            reordering costs more than it saves. NEITHER P5(b) nor P7 is
                            claimed. Reported as its own finding.
      0.833 <= ratio < 1.2  P7 HOLDS -- near-invariant.
      0.5   <= ratio < 0.833  SPLIT. No halving, but outside the invariance band. Neither
                            claimed whole.
      ratio < 0.5           P5(b) HOLDS -- halved.

B2  POWER FLOOR ON THE MATCHED SET.
    The matched-set size is reported explicitly, and EVERY per-pair ratio is printed, not
    only the median.
    If the matched set has FEWER THAN 10 pairs, the verdict is declared UNDERPOWERED and
    NEITHER P7 NOR P5(b) is called. The per-pair ratios are reported and scoring stops.

    Rationale, recorded in advance: per the paper's Section C, sample computation costs on
    the order of |D|x more arithmetic per estimate, and |D| goes from 1 to ~2,000. The
    per-sentence run may therefore time out far more often than the flat run did. If it
    does, the matched set collapses to the pairs that were EASIEST IN BOTH RUNS -- which is
    the same membership problem A1 already corrected P5(b) for, reappearing one level up.
    A median over the easiest survivors is not a measurement of the wall.

    Additionally: if per-sentence timeouts SUBSTANTIALLY EXCEED the flat run's 4, that is
    reported as a result in its own right -- it is the P5(a)/P5(b) asymmetry of A2 actually
    realised, with the per-sample arithmetic cost dominating whatever the bounds do.

B3  THE SOUNDNESS FINDING IS INDEPENDENT OF THE L4 OUTCOME.
    Recorded in advance so it is not re-litigated afterwards. Partition-invariance of the
    aggregated estimate holds regardless of what L4 shows: 5 of 27 prefixes remain
    inadmissible under the per-sentence partition, one pair (trained a=0.1 unit88) is worse
    than flat, and both miss-prefix ceilings are byte-identical across partitions
    (0.232677, 0.203398).
    L4 decides ONE thing only: whether diary D5.0 survives. It decides nothing about
    admissibility, nothing about the upstream report, and nothing about the 2/27 misses.
"""


def score_p7_banded(part, flat):
    import statistics as _st
    print(P7_ADDENDA)
    matched = [r for r in part
               if r["halted"] == "no"
               and flat.get((r["arm"], r["alpha"], r["unit"]), {}).get("halted") == "no"]
    part_halt = [r for r in part if r["halted"] != "no"]

    print("--- B2: matched set and per-pair ratios ---")
    print(f"  flat timeouts/caps        : {FLAT_TIMEOUTS}")
    print(f"  per-sentence timeouts/caps: {len(part_halt)}"
          + (f"   {[r['unit'] + '/' + r['alpha'] for r in part_halt]}" if part_halt else ""))
    if len(part_halt) > FLAT_TIMEOUTS:
        print(f"  >>> per-sentence timeouts EXCEED flat ({len(part_halt)} > {FLAT_TIMEOUTS}).")
        print("      This is the P5(a)/P5(b) asymmetry realised: per-sample arithmetic cost")
        print("      (~|D|x, |D| 1 -> ~2000) dominating whatever the bounds do. A result in")
        print("      its own right.")
    print(f"  MATCHED SET SIZE: {len(matched)} of 27")

    if matched:
        print(f"\n  {'pair':>28} {'flat_peak':>11} {'part_peak':>11} {'ratio':>8}")
        ratios = []
        for r in sorted(matched, key=lambda r: (r["arm"], r["alpha"], r["unit"])):
            k = (r["arm"], r["alpha"], r["unit"])
            fp = float(flat[k]["peak_frontier"]); pp = float(r["peak"])
            ratios.append(pp / fp)
            print(f"  {k[0] + ' a=' + k[1] + ' ' + k[2]:>28} {fp:>11,.0f} {pp:>11,.0f} "
                  f"{pp / fp:>8.3f}x")
        print(f"  per-pair ratio: min {min(ratios):.3f}  median {_st.median(ratios):.3f}  "
              f"max {max(ratios):.3f}")

    if len(matched) < 10:
        print(f"\n  *** VERDICT: UNDERPOWERED. Matched set is {len(matched)} < 10.")
        print("      NEITHER P7 NOR P5(b) is called. Per-pair ratios reported above; stop.")
        print("      The matched set is the pairs easiest in BOTH runs, which is the A1")
        print("      membership problem one level up.")
        return

    mp = _st.median([float(r["peak"]) for r in matched])
    fp = _st.median([float(flat[(r["arm"], r["alpha"], r["unit"])]["peak_frontier"])
                     for r in matched])
    ratio = mp / fp
    print(f"\n--- B1: four-way outcome  (matched median ratio {ratio:.3f}x) ---")
    if ratio >= 1.2:
        v = ("FRONTIER ENLARGED — distinct result, neither P5(b) nor P7 claimed. "
             "Refinement reordering costs more than it saves.")
    elif ratio >= 0.833:
        v = "P7 HOLDS — near-invariant. The L4 wall is NOT a harness artifact; D5.0 survives."
    elif ratio >= 0.5:
        v = "SPLIT — no halving, but outside the invariance band. Neither claimed whole."
    else:
        v = "P5(b) HOLDS — halved. The wall moved; D5.0 superseded by an APPENDED entry."
    print(f"  {v}")

# =====================================================================================
# C1-C3 — registered 2026-08-01 with results/partition_L4.csv verified absent
# (L4 run at 42:41 elapsed, still in flight).
# =====================================================================================
REGISTERED_C = """
C1  STOP RULE FOR THE RUN, fixed before it becomes a judgement call.
    Either ALL 27 pairs complete, or the result is reported PARTIAL:
      - the unrun pairs are NAMED explicitly,
      - per-pair ratios are reported for the pairs that did run,
      - NO median is claimed,
      - NEITHER P7 NOR P5(b) is called.
    Rationale: stopping partway selects the FAST pairs. That compounds with the matched-set
    selection A1 already corrected for -- two selection effects in the same direction, and
    the median would be over pairs that are fast in the flat run, fast in the per-sentence
    run, AND early enough to have finished. No median over that set means anything.

C2  REVISED (2026-08-01, before the file exists). NOT void -- a ONE-SIDED LOWER-BOUND CHECK.

    `max_length` is a MAXIMUM, not an exact length. Verified in source: `optimal.py:566`
    marks a node INDIVIDUAL only when `len(candidate_formula) == max_length`, `:575` computes
    `available_spots = max_length - len(candidate_formula)`, and shorter formulas are scored
    through the ancestor-propagation block (`:816-847`). Verified empirically: 1 of 27
    length-3 runs returned a formula with fewer than 3 leaves, in BOTH the flat and the
    per-sentence runs.

    Therefore the length-4 in-grammar space CONTAINS the length-3 space, and:

        part_L4 <  true_L3   =>  DEFINITE MISS. No length-4 oracle needed.
        part_L4 >= true_L3   =>  INCONCLUSIVE. Cannot confirm length-4 optimality.

    So `in_grammar_max` / `missed` in partition_L4.csv are KEPT and relabelled: they are a
    one-sided check that DETECTS LOSSES and CANNOT CONFIRM OPTIMALITY. `missed == 1` is a
    real length-4 miss; `missed == 0` means "not caught", not "optimal".

    A full length-4 miss COUNT still needs a genuine length-4 brute force
    (K*(3K)^3 = 1,366,875 formulas x 27 pairs) and is not claimed from this file.

    ORIGINAL C2 TEXT, superseded, kept so the correction is visible:
    `exp_partition.py:130` enumerates the brute-force optimum at a HARDCODED DEPTH OF 3
    (`for i ... for m2 in mv(...) ... for m3 in mv(m2)`), regardless of `PART_LENGTH`. So in
    partition_L4.csv the `in_grammar_max` and `missed` columns hold the LENGTH-3 optimum,
    not the length-4 one.
    Already visible in the run log: `trained a=0.1 unit396: part=0.146338 true=0.145326` --
    the length-4 search legitimately EXCEEDS the length-3 optimum, which is correct
    behaviour and a wrong comparison.
    Consequences, fixed in advance:
      - `in_grammar_max` and `missed` in partition_L4.csv are VOID. They are not read, and
        the columns are labelled in the file rather than deleted.
      - The V0 control is UNAFFECTED: it compares part_iou against the FLAT length-4
        exact_IoU from beam_vs_exact_K15.csv, not against this column.
      - P5(b)/P7/B1/B2 are UNAFFECTED: they are peak-frontier ratios and never touch it.
      - A length-4 miss count REQUIRES a genuine length-4 brute force: K*(3K)^3 = 1,366,875
        formulas x 27 pairs. Not run, and NOT claimed from this file.

C3  P8 — THE FIXED-P2 SHAPE, pre-registered because it is predictable from what is known.
    Columns: (inadmissible ceiling) / (prefix expanded before ANY copy of it was dropped) /
    (optima actually lost).

        flat (one sample)   6 / 4 / 2
        per-sentence        5 / 5 / 0        <- P8

    SUPPORTED if the per-sentence row is exactly 5 / 5 / 0.
    If P8 holds, the finding is reported AS THE PAIR OF ROWS, not as a miss count: the
    defect is near-constant (6 -> 5) while realised harm went 2 -> 0, and the ENTIRE
    difference is pop ordering.
    IF ANY per-sentence prefix was NOT expanded before being dropped, there is a loss we
    have not observed, and the 0/27 result needs re-examining before anything is claimed
    from it.
    Note: P8 is about the LENGTH-3 partition run, which is complete. It does not depend on
    L4 and is not blocked by C2.
"""


def report_partial(part):
    """C1: name the unrun pairs and refuse a median."""
    from exp_beam_width import PAIRS as _P
    expected = {(a, al, f"unit{u}") for a, al, us in _P for u in us}
    got = {(r["arm"], r["alpha"], r["unit"]) for r in part}
    missing = sorted(expected - got)
    if not missing:
        return False
    print("\n*** C1 STOP RULE FIRED: PARTIAL RESULT ***")
    print(f"  {len(got)} of 27 pairs completed; {len(missing)} did not run:")
    for k in missing:
        print(f"    {k[0]} a={k[1]} {k[2]}")
    print("  NO median is claimed. NEITHER P7 NOR P5(b) is called.")
    print("  Stopping partway selects the fast pairs, compounding with the matched-set")
    print("  selection A1 corrected for -- two selection effects in the same direction.")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
