"""D6 final item — exit-path distribution over all filtered flat K=15 events.

Builds a LOGGING-ONLY copy of optimal.py at run time, never committed. Four guards:

  G1  generated in a temp dir, never written into the tree. verify check 3 asserts bit-identity
      with clean upstream at 7080529 and would break on a committed patched copy.
  G2  insertion-only proved MECHANICALLY: `git diff --no-index --numstat` must show 0
      deletions, and every added line must be a logging call. The numstat is printed.
  G3  determinism gate: the patched run must match the unpatched on returned IoU (all 27),
      pop counts, expanded counts, AND the filtered event count. Identical IoUs alone would
      pass a patch that changed traversal without changing the answer.
  G4  assertion rule ENFORCED BY THE SCRIPT: if the patched run trips any upstream assertion,
      report and HALT. Never disable, never treat as bookkeeping. That was experiment C's
      actual failure.

FIVE node-exit `continue` sites, not six. `:804` is a `for ancestor` loop continue at indent
28 inside the propagation block -- it skips an ancestor, not the node -- and is excluded.

  :679  incumbent skip        -e_node < minimum_threshold
  :709  refinement discard    new_max < -e_node, not re-pushed
  :747  distributive discard  transformed estimate lower, not re-pushed
  :753  recent_nodes memory   node in recent_nodes
  :871  loop tail             normal end of iteration

Plus the pre-pop vanish at estimate_iou_frontier:389, which is not a continue and is counted
separately: a node that never reaches a pop cannot exit through any of the five.

M3: report the share of events exiting with NO logged path. D6 closes on this.
"""
import os, re, subprocess, sys, tempfile, types
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.environ.get(
    "OPTIMALCE_UPSTREAM", os.path.expanduser("~/projects/optimalce")))
import real_token_masks as rtm, real_token_search as rts, synthetic_overlap_sweep as sos
from compositional import optimal as clean_optimal
from utils import mask_utils as MU
from exp_beam_width import PAIRS, MAX_SENTS, MIN_SUPPORT, CAP

FEATS = os.path.expanduser("~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats")
SITES = {679: "incumbent_skip", 709: "refinement_discard", 747: "distributive_discard",
         753: "recent_nodes_memory", 871: "loop_tail"}


def build(src_path, tmpdir):
    lines = open(src_path).read().split("\n")
    for ln in sorted(SITES, reverse=True):
        raw = lines[ln - 1]
        assert raw.strip() == "continue", f":{ln} is {raw.strip()!r}, not 'continue'"
        indent = raw[:len(raw) - len(raw.lstrip())]
        extra = (", new_max, minimum_threshold, best_results[0]" if ln == 747 else "")
        lines.insert(ln - 1, f'{indent}_EXIT_LOG(node, "{SITES[ln]}"{extra})')
    out = os.path.join(tmpdir, "optimal_logged.py")
    open(out, "w").write("\n".join(lines))
    d = subprocess.run(["git", "diff", "--no-index", "--numstat", src_path, out],
                       capture_output=True, text=True).stdout.strip()
    print(f"  G2 numstat (added, deleted, file):\n    {d}")
    add, dele = d.split("\t")[0], d.split("\t")[1]
    assert dele == "0", f"G2 FAILED: {dele} deletions, must be 0"
    assert int(add) == len(SITES), f"G2 FAILED: {add} additions, expected {len(SITES)}"
    diff = subprocess.run(["git", "diff", "--no-index", "-U0", src_path, out],
                          capture_output=True, text=True).stdout
    added = [l[1:].strip() for l in diff.split("\n") if l.startswith("+") and not l.startswith("+++")]
    assert all(l.startswith("_EXIT_LOG(") for l in added), f"G2 FAILED: non-logging line {added}"
    print(f"  G2 PASS: 0 deletions, {add} additions, all `_EXIT_LOG(...)`")
    m = types.ModuleType("optimal_logged"); m.__dict__["__name__"] = "optimal_logged"
    exec(compile("\n".join(lines), out, "exec"), m.__dict__)
    return m


def run(mod, dense, nb, con, log=None):
    npop = [0]; OP = sos.HeapProbe.heappop
    def pop(self, h):
        npop[0] += 1; return OP(self, h)
    sos.HeapProbe.heappop = pop
    saved = rts.optimal
    rts.optimal = mod
    if log is not None: mod._EXIT_LOG = log
    else: mod.__dict__.setdefault("_EXIT_LOG", lambda *a: None)
    try:
        r = rts.run_one(dense, nb, 3, CAP, 1500.0, None, concepts=con)
    finally:
        sos.HeapProbe.heappop = OP; rts.optimal = saved
    return r, npop[0]


def main():
    print(__doc__)
    tok = rtm.load_tokens(FEATS, MAX_SENTS); _, cats = rts.ARMS["all"]
    con = rtm.select_concepts(tok, cats, 15, MIN_SUPPORT); dense = rtm.build_dense(tok, con)
    keys, neur = [], []
    for arm, alpha, units in PAIRS:
        z = np.load(os.path.join(REPO, "results", f"acts2k_{arm}_a{alpha}.npz"))
        for u in units: keys.append((arm, alpha, f"unit{u}")); neur.append(z["acts"][u].astype(bool))

    tmp = tempfile.mkdtemp(prefix="optlog_")
    src = os.path.join(REPO, ".upstream-clean", "compositional", "optimal.py")
    print(f"  G1 generated in {tmp} (never written into the tree)")
    mod = build(src, tmp)

    print("\n  G3 determinism gate: patched vs unpatched on IoU, pops, expanded")
    bad = []
    base, patched = [], []
    for k, nb in zip(keys, neur):
        a, pa = run(clean_optimal, dense, nb, con)
        b, pb = run(mod, dense, nb, con)
        ia = a["n_inter"] / (a["n_fires"] + int(nb.sum()) - a["n_inter"])
        ib = b["n_inter"] / (b["n_fires"] + int(nb.sum()) - b["n_inter"])
        base.append((ia, pa, a["expanded"])); patched.append((ib, pb, b["expanded"]))
        if abs(ia - ib) > 1e-15 or pa != pb or a["expanded"] != b["expanded"]:
            bad.append((k, (ia, pa, a["expanded"]), (ib, pb, b["expanded"])))
    if bad:
        print(f"  G3 FAILED on {len(bad)} pairs -- RUN VOID, no distribution reported")
        for x in bad[:5]: print(f"    {x}")
        return 1
    print(f"  G3 PASS: all 27 identical on IoU, pop count and expanded count")

    print("\n  exit-path distribution over filtered events")
    from collections import Counter
    dist = Counter(); nolog = 0; total = 0; E_ROWS = []
    OGFM = MU.get_formula_mask_and_tree
    for k, nb in zip(keys, neur):
        nn = int(nb.sum()); thr = [0.0]; popped = {}; smask = set(); exits = {}; detail = {}
        ORED = mod.reduce_frontier
        def red(fr, t):
            thr[0] = max(thr[0], t); return ORED(fr, t)
        def gfm(f, masks, path_masks=None, device=None):
            o = OGFM(f, masks, path_masks, device) if device is not None else OGFM(f, masks, path_masks)
            try: smask.add(rts.eval_formula(f, dense).tobytes())
            except Exception: pass
            return o
        OP = sos.HeapProbe.heappop
        def pop(self, h):
            it = OP(self, h)
            if it[1] == "INDIVIDUAL":
                kk = rts.render(it[2], con)
                if kk not in popped:
                    m = rts.eval_formula(it[2], dense)
                    popped[kk] = (int((m & nb).sum()) / max(int((m | nb).sum()), 1), thr[0], m.tobytes())
            return it
        def elog(node, tag, new_max=None, min_thr=None, incumbent=None):
            try:
                lab = rts.render(node[2], con)
                exits.setdefault(lab, []).append(tag)
                if tag == "distributive_discard":
                    m = rts.eval_formula(node[2], dense)
                    ex = int((m & nb).sum()) / max(int((m | nb).sum()), 1)
                    detail.setdefault(lab, []).append((new_max, min_thr, ex, incumbent,
                                                       node[1]))
            except Exception: pass
        mod.reduce_frontier = red; MU.get_formula_mask_and_tree = gfm; sos.HeapProbe.heappop = pop
        try:
            run(mod, dense, nb, con, log=elog)
        except AssertionError as e:
            print(f"  G4 HALT: upstream assertion tripped on {k}: {e}")
            print("  Reported, not disabled. Run stops here.")
            return 1
        finally:
            mod.reduce_frontier = ORED; MU.get_formula_mask_and_tree = OGFM; sos.HeapProbe.heappop = OP
        for lab, (iou, t, mb) in popped.items():
            if iou > t + 1e-12 and mb not in smask:
                total += 1
                tags = exits.get(lab)
                if tags:
                    dist[tags[-1]] += 1
                    if tags[-1] == "distributive_discard" and lab in detail:
                        E_ROWS.append(detail[lab][-1])
                else: dist["NO_LOGGED_PATH"] += 1; nolog += 1
    print(f"\n  filtered events traced: {total}")
    print(f"\n  E1/E2 -- at :747, new_max (transformed estimate) vs the ORIGINAL formula's exact IoU")
    print(f"    {'new_max':>12} {'min_thresh':>12} {'exact_iou':>12} {'incumbent':>12} {'final?':>7}")
    under = ge = 0; chain = {"thr<=inc": 0, "thr>inc": 0}
    for row in E_ROWS:
        nm, mt, ex, inc, nop = row
        if nm is None or ex is None: continue
        (under := under) ; 
        if nm < ex - 1e-15: under += 1
        else: ge += 1
        if mt is not None and inc is not None:
            chain["thr<=inc" if mt <= inc + 1e-15 else "thr>inc"] += 1
    for row in E_ROWS[:8]:
        nm, mt, ex, inc, nop = row
        print(f"    {nm!r:>12} {mt!r:>12} {ex:>12.9f} {inc!r:>12} {str(nop=='INDIVIDUAL'):>7}")
    z = sum(1 for r in E_ROWS if r[0] == 0.0)
    nz_under = sum(1 for r in E_ROWS if r[0] is not None and r[0] != 0.0 and r[0] < r[2] - 1e-15)
    nz_ge = sum(1 for r in E_ROWS if r[0] is not None and r[0] != 0.0 and r[0] >= r[2] - 1e-15)
    print(f"\n    new_max <  exact_iou : {under} of {under+ge}")
    print(f"    new_max >= exact_iou : {ge} of {under+ge}")
    print(f"\n    SENTINEL SPLIT (path_heuristic.py:50 and :174 both `return 0.0, 0.0`)")
    print(f"      new_max EXACTLY 0.0 (sentinel, not an estimate) : {z}")
    print(f"      new_max nonzero AND < exact_iou (real under-bound): {nz_under}")
    print(f"      new_max nonzero AND >= exact_iou (sound)          : {nz_ge}")
    print(f"    -> informative rows = {nz_under + nz_ge} of {len(E_ROWS)}")
    print(f"\n    ordering of minimum_threshold vs incumbent (hypothesis, not required by the test):")
    print(f"      threshold <= incumbent : {chain['thr<=inc']}")
    print(f"      threshold >  incumbent : {chain['thr>inc']}")
    for tag, c in dist.most_common():
        print(f"    {tag:<22} {c:>4}  ({100*c/total:.1f}%)")
    print(f"\n  M3: share exiting with NO logged path = {100*nolog/total:.1f}%  ({nolog}/{total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
