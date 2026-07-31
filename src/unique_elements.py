"""Unique-element fraction of the SNLI token corpus, at two corpus sizes.

This is the OptimalCE paper's own §4.3 variable ("Importance of Unique Elements"): the
fraction of ELEMENTS carrying exactly one admitted concept. Their heuristic degrades as it
falls, because an element covered by several concepts gives the bound nothing to separate.

It is not the same statistic as `disjoint_pairs`, which we had been reporting. That counts
CONCEPT PAIRS that never co-occur -- a property of the vocabulary. A vocabulary can be
mostly pairwise-disjoint while almost every element still carries several concepts, which
is exactly the case here: ~51% of pairs are disjoint, but only ~14% of tokens are unique.

Needs no search and no model; it is a property of the annotation alone.
"""

import argparse
import csv
import os
import sys

from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_info  # noqa: E402
import real_token_masks as rtm  # noqa: E402


def analyse(feats, max_sents, K, min_support):
    tokens = rtm.load_tokens(feats, max_sents)

    counts = Counter()
    for f in tokens:
        for cat in rtm.CATEGORIES:
            for val in f.get(cat, []):
                counts[(cat, val)] += 1
    eligible = [c for c, n in counts.items() if n >= min_support]
    by_cat_support = Counter(c[0] for c in eligible)

    concepts = rtm.select_concepts(tokens, rtm.CATEGORIES, K, min_support)
    dense = rtm.build_dense(tokens, concepts)
    diag = rtm.diagnostics(dense)

    # For tokens carrying exactly one admitted concept, which category supplies it.
    sums = dense.sum(axis=0)
    unique_by_cat = Counter()
    for m in (sums == 1).nonzero()[0]:
        k = int(dense[:, m].argmax())
        unique_by_cat[concepts[k][0]] += 1

    return {
        "max_sents": max_sents, "n_tokens": len(tokens), "K": K,
        "distinct_pairs": len(counts), "pairs_over_min_support": len(eligible),
        "support_by_cat": dict(sorted(by_cat_support.items())),
        "unique_by_cat": dict(sorted(unique_by_cat.items(), key=lambda kv: -kv[1])),
        **{k: diag[k] for k in ("M", "active_tokens", "coverage", "unique_tokens",
                                "unique_frac_all", "unique_frac_active", "mean_overlap",
                                "common_frac", "disjoint_pairs", "disjoint_frac",
                                "max_overlap")},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feats", default=os.path.expanduser(
        "~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 2000])
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--min_support", type=int, default=5)
    ap.add_argument("--out", default="results/unique_elements.csv")
    args = ap.parse_args()
    env_info.print_banner('unique-elements')

    rows = [analyse(args.feats, n, args.K, args.min_support) for n in args.sizes]

    flat = []
    for r in rows:
        row = {k: v for k, v in r.items() if not isinstance(v, dict)}
        for cat, n in r["support_by_cat"].items():
            row[f"support_{cat}"] = n
        for cat, n in r["unique_by_cat"].items():
            row[f"uniqtok_{cat}"] = n
        flat.append(row)
    cols = sorted({k for r in flat for k in r},
                  key=lambda c: (c.startswith("support_"), c.startswith("uniqtok_"), c))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in flat:
            w.writerow(r)
    print(f"wrote {args.out}\n")

    for r in rows:
        print(f"=== {r['max_sents']} sentences | {r['n_tokens']} tokens | K={r['K']} ===")
        print(f"  concepts clearing min_support={args.min_support}: "
              f"{r['pairs_over_min_support']} of {r['distinct_pairs']} distinct")
        print(f"    by category: {r['support_by_cat']}")
        print(f"  UNIQUE ELEMENTS (paper §4.3): {r['unique_tokens']}/{r['M']} = "
              f"{r['unique_frac_all']:.1%} of all tokens, "
              f"{r['unique_frac_active']:.1%} of active tokens")
        print(f"    category supplying the single concept: {r['unique_by_cat']}")
        print(f"  vocabulary-level (NOT the same statistic): disjoint_pairs "
              f"{r['disjoint_pairs']} = {r['disjoint_frac']:.1%} of pairs")
        print(f"  mean_overlap {r['mean_overlap']} | max_overlap {r['max_overlap']} | "
              f"coverage {r['coverage']}\n")

    a, b = rows[0], rows[-1]
    print(f"TREND with corpus size: unique-element fraction "
          f"{a['unique_frac_all']:.1%} at {a['n_tokens']} tokens -> "
          f"{b['unique_frac_all']:.1%} at {b['n_tokens']} tokens "
          f"({'DOWN' if b['unique_frac_all'] < a['unique_frac_all'] else 'UP'}); "
          f"disjoint_pairs {a['disjoint_frac']:.1%} -> {b['disjoint_frac']:.1%}")


if __name__ == "__main__":
    raise SystemExit(main())
