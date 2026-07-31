"""Run the OptimalCE optimal search on REAL SNLI token concept masks.

This is the synthetic sweep's measurement core (`synthetic_overlap_sweep.HeapProbe` /
`StubConfig` / quantity helpers) pointed at real concept masks built by
`real_token_masks.py`, so peak-frontier / visited / time / IoU numbers are directly
comparable to `results/overlap_sweep.csv` and `results/beam_sweep.csv`.

The concept masks are real. The **neuron is a proxy** — an OR of three real concepts plus
label noise, mirroring `synthetic_overlap_sweep.make_neuron`. That is deliberate and it
bounds the claim: this measures how the SEARCH behaves under real token concept structure
(which is what the frontier explosion is about — the combinatorics live on the concept
side), NOT what real neurons in a trained model encode. Real activations require the
encoder checkpoint and are a separate step.

Usage::

    python src/real_token_search.py --arms tag all --lengths 3 4
"""

import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

# Upstream `compositional` package (patched with MAX_FRONTIER_SIZE) and our synthetic harness.
UPSTREAM = os.environ.get("OPTIMALCE_UPSTREAM", os.path.expanduser("~/projects/optimalce"))
sys.path.insert(0, UPSTREAM)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importing the synthetic harness installs its meta-path stub finder for the vision-only
# deps (detectron2, cityscapesscripts) as a side effect, which is what lets `compositional`
# import on a machine that has neither.
from synthetic_overlap_sweep import (  # noqa: E402
    HeapProbe, StubConfig, compute_disjoint_info, compute_quantities, _Halt,
)
import env_info  # noqa: E402
import real_token_masks as rtm  # noqa: E402

import heapq  # noqa: E402
import time  # noqa: E402
from compositional import formula as F  # noqa: E402
from compositional import optimal  # noqa: E402

ARMS = {
    # Single-valued categories are disjoint by construction: a token has exactly one POS
    # tag, one dep label, one lemma. `tag`/`dep` are closed-class and saturate (~23 tags
    # meet min_support), so they cannot supply a K-matched control at K=30+. `lemma` is
    # open-class and disjoint, so it scales to any K -- use it as the high-K control.
    "tag": ("tag-only (disjoint control)", ["tag"]),
    "dep": ("dep-only", ["dep"]),
    "lemma": ("lemma-only (disjoint control, scales with K)", ["lemma"]),
    "tagdep": ("tag+dep", ["tag", "dep"]),
    "all": ("all categories (token regime)", rtm.CATEGORIES),
}


def load_real_neurons(path, n_units, rng, dmin=0.15, dmax=0.85, min_fire=0, unit_ids=None):
    """Pick `n_units` real unit masks from a `real_activations.py` dump.

    Units are filtered to a sensible density band — a unit that fires on ~everything or
    ~nothing has a degenerate optimum and would measure thresholding, not search behaviour.
    Selection is seeded so runs are reproducible.
    """
    z = np.load(path)
    acts, density = z["acts"], z["density"]
    fires = acts.sum(axis=1)
    # Statistical floor: a unit with too few firing tokens cannot support a length-4
    # formula drawn from ~1.4M candidates -- best-of-N would make noise look explanatory.
    # Fixed in advance and applied before any scoring.
    in_band = (density >= dmin) & (density <= dmax)
    eligible = np.where(in_band & (fires >= min_fire))[0]
    n_excluded = int((in_band & (fires < min_fire)).sum())
    alpha = float(z["alpha"]) if "alpha" in z else -1.0
    print(f"[units] {path}: {len(eligible)} eligible, {n_excluded} excluded by "
          f"min_fire={min_fire} (fires min={int(fires.min())} median={int(np.median(fires))})")
    if len(eligible) == 0:
        raise SystemExit(
            f"NO ELIGIBLE UNITS in {path}: every unit in band [{dmin}, {dmax}] fires on "
            f"fewer than {min_fire} tokens (max={int(fires.max())}). At alpha={alpha} this "
            f"corpus (M={acts.shape[1]}) is too small; M >= {int(min_fire/max(alpha,1e-9))} "
            f"would be required.")
    if unit_ids:
        # Phase B targets specific (unit, alpha) pairs selected by Phase A, so the unit set
        # is given explicitly rather than resampled.
        missing = [u for u in unit_ids if u not in set(eligible.tolist())]
        if missing:
            raise SystemExit(f"units {missing} are not eligible in {path}")
        pick = np.array(sorted(unit_ids))
    else:
        pick = rng.choice(eligible, size=min(n_units, len(eligible)), replace=False)
    return ([(int(u), acts[u].astype(bool)) for u in sorted(pick)], bool(z["untrained"]),
            alpha, n_excluded)


def make_proxy_neuron(dense, rng, noise, n_target=3):
    """Proxy neuron: OR of three real concepts + label noise.

    Concepts are chosen from the middle of the frequency order rather than the top, so the
    target is neither near-empty nor near-universal (a trivially coverable target would
    terminate the search early and hide the frontier behaviour we are measuring).
    """
    K, M = dense.shape
    coverage = dense.sum(axis=1)
    order = np.argsort(-coverage)
    mid = order[K // 3: K // 3 + n_target] if K >= n_target * 2 else order[:n_target]
    target = np.zeros(M, dtype=bool)
    for k in mid:
        target |= dense[k]
    flip = rng.random(M) < noise
    return np.where(flip, ~target, target), [int(k) for k in mid]


def _is_leafish(f):
    """Leaf, or NOT of a leaf -- the two things that carry a single concept."""
    return isinstance(f, F.Leaf) or (isinstance(f, F.Not) and isinstance(f.val, F.Leaf))


def _concept_of(f):
    """Concept index carried by a leafish node."""
    return f.val if isinstance(f, F.Leaf) else f.val.val


def eval_formula(f, dense):
    """Token mask the formula fires on. Mirrors the operator semantics in formula.py."""
    if isinstance(f, F.Leaf):
        return dense[f.val]
    if isinstance(f, F.Not):
        return ~eval_formula(f.val, dense)
    if isinstance(f, F.Or):
        return eval_formula(f.left, dense) | eval_formula(f.right, dense)
    if isinstance(f, F.And):
        return eval_formula(f.left, dense) & eval_formula(f.right, dense)
    raise TypeError(f"unsupported formula node: {type(f)}")


def count_ops(f):
    """(n_and, n_or, n_not) over the whole tree."""
    if isinstance(f, F.Leaf):
        return (0, 0, 0)
    if isinstance(f, F.Not):
        a, o, n = count_ops(f.val)
        return (a, o, n + 1)
    la, lo, ln = count_ops(f.left)
    ra, ro, rn = count_ops(f.right)
    is_or = isinstance(f, F.Or)
    return (la + ra + (0 if is_or else 1), lo + ro + (1 if is_or else 0), ln + rn)


def _or_chains(f, chains):
    """Collect the operands of every maximal OR-chain in the tree.

    An OR-chain is a run of adjacent Or nodes; its operands are the non-Or subtrees
    hanging off it. Recursion continues into those operands so nested chains under an
    AND are counted as their own chain.
    """
    if _is_leafish(f):
        return
    if isinstance(f, F.Or):
        operands, stack = [], [f]
        while stack:
            node = stack.pop()
            if isinstance(node, F.Or):
                stack.extend((node.left, node.right))
            else:
                operands.append(node)
        chains.append(operands)
        for op in operands:
            _or_chains(op, chains)
        return
    if isinstance(f, F.Not):
        _or_chains(f.val, chains)
        return
    _or_chains(f.left, chains)
    _or_chains(f.right, chains)


def render(f, concepts):
    """Readable formula string.

    Hand-rolled rather than `F.to_str` because upstream's `BinaryNode.to_str` forwards a
    `sort=` kwarg that `Leaf.to_str` does not accept, so it raises on any leaf child.
    """
    if isinstance(f, F.Leaf):
        return f"{concepts[f.val][0]}={concepts[f.val][1]}"
    if isinstance(f, F.Not):
        return f"(NOT {render(f.val, concepts)})"
    return f"({render(f.left, concepts)} {f.op} {render(f.right, concepts)})"


def formula_stats(f, concepts, dense, neuron_bits=None):
    """Readable string plus the OR-degeneracy diagnostics for one winning formula."""
    n_and, n_or, n_not = count_ops(f)

    chains = []
    _or_chains(f, chains)
    or_cats, max_same_cat = [], 0
    for operands in chains:
        cats = [concepts[_concept_of(o)][0] for o in operands if _is_leafish(o)]
        or_cats.extend(cats)
        if cats:
            max_same_cat = max(max_same_cat, max(Counter(cats).values()))

    fires = eval_formula(f, dense)
    out = {
        "formula": render(f, concepts),
        "formula_cov": round(float(fires.mean()), 4),
        "n_and": n_and, "n_or": n_or, "n_not": n_not,
        "or_categories": "+".join(sorted(Counter(or_cats).elements())),
        "max_same_cat_or": max_same_cat,
        "n_fires": int(fires.sum()), "n_inter": None,
    }
    if neuron_bits is not None:
        out["n_inter"] = int((fires & neuron_bits).sum())
    return out


def run_one(dense, neuron_bits, length, cap, time_budget, beam_cap, concepts=None,
            expand_budget=None):
    """One search run; mirrors synthetic_overlap_sweep.run_level but takes masks directly.

    `concepts` is diagnostic-only: it names the winning label for the formula columns and
    does not enter the search.
    """
    import scipy.sparse as sparse
    import torch

    K, M = dense.shape
    masks = [sparse.csr_matrix(dense[c].reshape(1, M)) for c in range(K)]
    common, unique, uncoverable, _ = compute_quantities(dense, M)
    disjoint_info = compute_disjoint_info(dense, K)
    bitmaps = torch.from_numpy(neuron_bits.reshape(1, M))

    probe = HeapProbe(cap, time_budget=time_budget)
    optimal.heapq = probe

    # Fixed-work mode: stop generating NEW expansions after `expand_budget`, but let the
    # search drain its queue and return its incumbent normally. Scoring is untouched --
    # this truncates exploration only, which is what a work budget means.
    saved_expand = optimal.expand_node
    if expand_budget is not None:
        counter = {"n": 0}

        def capped_expand(node, candidate_labels, max_length):
            if counter["n"] >= expand_budget:
                return []
            counter["n"] += 1
            return saved_expand(node, candidate_labels=candidate_labels,
                                max_length=max_length)

        optimal.expand_node = capped_expand
    optimal.MAX_FRONTIER_SIZE = beam_cap
    cfg = StubConfig(length, M)

    t0 = time.time()
    halt = ""
    best_label = None
    # optimal.py streams a per-pop progress line; silence it so the sweep output stays readable.
    devnull = open(os.devnull, "w")
    saved_stdout = sys.stdout
    try:
        sys.stdout = devnull
        best_label, best_iou, visited, expanded, estimated = optimal.compute_optimal_explanations(
            bitmaps=bitmaps, masks=masks, masks_info=(common, unique, uncoverable),
            disjoint_info=disjoint_info, config=cfg,
        )
    except _Halt as h:
        halt = h.reason
        best_iou, visited, expanded, estimated = float("nan"), -1, -1, -1
    finally:
        sys.stdout = saved_stdout
        devnull.close()
        optimal.heapq = heapq
        optimal.expand_node = saved_expand
        optimal.MAX_FRONTIER_SIZE = None
        dt = time.time() - t0

    # Formula columns are permanent: a score without the label it came from is what let a
    # near-universal winner sit undetected across two diary entries.
    fstats = {"formula": None, "formula_cov": None, "n_and": None, "n_or": None,
              "n_not": None, "or_categories": None, "max_same_cat_or": None,
              "n_fires": None, "n_inter": None}
    if best_label is not None and concepts is not None:
        fstats = formula_stats(best_label, concepts, dense, neuron_bits)

    return {
        "peak_frontier": probe.peak, "halted": halt or "no", "visited": visited,
        "expanded": expanded, "estimated": estimated,
        "best_iou": round(best_iou, 4) if best_iou == best_iou else None,
        "time_s": round(dt, 2), **fstats,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feats", default=os.path.expanduser(
        "~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats"))
    ap.add_argument("--max_sents", type=int, default=200)
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--min_support", type=int, default=5)
    ap.add_argument("--arms", nargs="+", default=["tag", "all"], choices=list(ARMS))
    ap.add_argument("--lengths", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--beam_list", nargs="+", default=["none"],
                    help="beam widths; 'none' = exact optimal")
    ap.add_argument("--cap", type=int, default=200000)
    ap.add_argument("--time_budget", type=float, default=60.0)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neuron", choices=["proxy", "real"], default="proxy",
                    help="proxy = OR of 3 concepts + noise; real = units from real_activations.py")
    ap.add_argument("--acts", default="results/real_activations.npz")
    ap.add_argument("--units", type=int, default=5, help="how many real units to run")
    ap.add_argument("--expand_budget", type=int, default=None,
                    help="fixed-work mode: cap the number of node expansions")
    ap.add_argument("--unit_ids", type=int, nargs="+", default=None,
                    help="explicit unit indices (Phase B); overrides random selection")
    ap.add_argument("--min_fire", type=int, default=0,
                    help="statistical floor: drop units firing on fewer than this many tokens")
    ap.add_argument("--dmin", type=float, default=0.15, help="min unit density to consider")
    ap.add_argument("--dmax", type=float, default=0.85, help="max unit density to consider")
    ap.add_argument("--out", default="results/real_token_search.csv")
    args = ap.parse_args()
    env_info.print_banner('real-search')

    tokens = rtm.load_tokens(args.feats, args.max_sents)
    print(f"[real-search] {len(tokens)} tokens | K={args.K} cap={args.cap} "
          f"budget={args.time_budget}s\n", flush=True)

    rows = []
    for arm in args.arms:
        name, cats = ARMS[arm]
        concepts = rtm.select_concepts(tokens, cats, args.K, args.min_support)
        dense = rtm.build_dense(tokens, concepts)
        diag = rtm.diagnostics(dense)
        if args.neuron == "proxy":
            bits, target_ks = make_proxy_neuron(
                dense, np.random.default_rng(args.seed + 999), args.noise)
            neurons = [("proxy", bits)]
            alpha, untrained, n_excluded = -1.0, False, 0
            label = f"OR{[f'{concepts[k][0]}={concepts[k][1]}' for k in target_ks]}"
        else:
            picked, untrained, alpha, n_excluded = load_real_neurons(
                args.acts, args.units, np.random.default_rng(args.seed + 999),
                dmin=args.dmin, dmax=args.dmax, min_fire=args.min_fire,
                unit_ids=args.unit_ids)
            if len(picked[0][1]) != len(tokens):
                raise SystemExit(
                    f"activation/mask token mismatch: {len(picked[0][1])} vs {len(tokens)} "
                    "-- regenerate the .npz with the same --max_sents")
            neurons = [(f"unit{u}", b) for u, b in picked]
            label = f"{len(neurons)} real units ({'untrained' if untrained else 'trained'})"

        print(f"=== {name} === overlap={diag['mean_overlap']} "
              f"common_frac={diag['common_frac']} disjoint_pairs={diag['disjoint_pairs']} "
              f"| neuron={label}\n", flush=True)

        for nid, neuron_bits in neurons:
            for length in args.lengths:
                for bw in args.beam_list:
                    beam_cap = None if str(bw).lower() == "none" else int(bw)
                    print(f"--- {arm} {nid} length={length} beam={bw} "
                          f"density={neuron_bits.mean():.3f} ---", flush=True)
                    res = run_one(dense, neuron_bits, length, args.cap,
                                  args.time_budget, beam_cap, concepts=concepts,
                                  expand_budget=args.expand_budget)
                    row = {"arm": name, "categories": "+".join(cats), "neuron": nid,
                           "alpha": alpha, "n_tokens": len(tokens),
                           "n_fire_neuron": int(neuron_bits.sum()),
                           "units_excluded_min_fire": n_excluded,
                           "untrained": untrained,
                           "density": round(float(neuron_bits.mean()), 5),
                           "length": length, "beam": bw,
                           "mean_overlap": diag["mean_overlap"],
                           "common_frac": diag["common_frac"],
                           "disjoint_pairs": diag["disjoint_pairs"], **res}
                    rows.append(row)
                    print(row, "\n", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")

    cols = ["alpha", "neuron", "density", "length", "beam", "mean_overlap", "disjoint_pairs",
            "peak_frontier", "visited", "best_iou", "formula_cov", "n_or", "n_and",
            "max_same_cat_or", "time_s", "halted"]
    print("\n=== SUMMARY ===")
    print(" ".join(f"{c:>14}" for c in cols))
    for r in rows:
        print(" ".join(f"{str(r.get(c, ''))[:14]:>14}" for c in cols))


if __name__ == "__main__":
    raise SystemExit(main())
