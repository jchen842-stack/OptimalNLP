"""Real SNLI token concept masks — does real NLP data actually sit in the "token regime"
that the synthetic sweep predicted would explode the optimal-search frontier?

Source: `snli_1.0_dev.feats` from the Compositional Explanations of Neurons (Mu & Andreas,
NeurIPS 2020) NLI codebase — spaCy/benepar/WordNet annotations, pre-computed and checked in,
so this needs no spaCy, no benepar, no GPU, no model.

Line format (one sentence per line), whitespace-separated tokens::

    text|lemma|tag|dep|ent|synset|const

`const` is multi-valued (";"-separated constituent labels); the rest are single-valued and
may be empty. A concept is a (category, value) pair, e.g. ("tag", "NNS").

The point of the contrast
-------------------------
Real data supplies BOTH regimes depending only on which concept categories you admit:

* ``--categories tag``  — a token has exactly one POS tag, so tag-concepts are mutually
  exclusive: a genuinely disjoint, vision-like partition made of real data.
* ``--categories all``  — a token simultaneously has a lemma AND a tag AND a dep AND a
  synset AND several constituents, so overlap is structural, not a knob we turned.

That makes the regime shift an observation about NLP rather than an artifact of the
synthetic generator, and it isolates overlap while holding the masks real in both arms.
"""

import argparse
import csv
from collections import Counter

import numpy as np

# Field order after the token text itself, per the .feats header line.
CATEGORIES = ["lemma", "tag", "dep", "ent", "synset", "const"]
MULTI = {"const"}  # ";"-separated, a token may belong to several at once


def load_tokens(path, max_sents=None):
    """Parse the .feats file into a flat list of per-token {category: [values]} dicts.

    The header line is skipped. Empty fields contribute no concept (an absent entity tag
    is not the concept "no entity"), which keeps `uncoverable` meaningful downstream.
    """
    tokens = []
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for i, line in enumerate(f):
            if max_sents is not None and i >= max_sents:
                break
            line = line.strip()
            if not line:
                continue
            for tok in line.split(" "):
                parts = tok.split("|")
                if len(parts) != len(CATEGORIES) + 1:
                    continue  # malformed / stray token, skip rather than guess
                feats = {}
                for cat, val in zip(CATEGORIES, parts[1:]):
                    if not val:
                        continue
                    feats[cat] = val.split(";") if cat in MULTI else [val]
                tokens.append(feats)
    return tokens


def select_concepts(tokens, categories, K, min_support):
    """Pick the K most frequent (category, value) concepts, restricted to `categories`.

    Frequency ranking mirrors how the upstream analysis filters to concepts with enough
    support to be estimable; `min_support` drops the long tail of hapax lemmas/synsets
    that can never form a useful explanation.
    """
    counts = Counter()
    for feats in tokens:
        for cat in categories:
            for val in feats.get(cat, []):
                counts[(cat, val)] += 1
    eligible = [(c, n) for c, n in counts.items() if n >= min_support]
    eligible.sort(key=lambda cn: (-cn[1], cn[0]))
    return [c for c, _ in eligible[:K]]


def build_dense(tokens, concepts):
    """(K, M) bool concept-membership matrix over the token axis."""
    index = {c: i for i, c in enumerate(concepts)}
    dense = np.zeros((len(concepts), len(tokens)), dtype=bool)
    for m, feats in enumerate(tokens):
        for cat, vals in feats.items():
            for val in vals:
                k = index.get((cat, val))
                if k is not None:
                    dense[k, m] = True
    return dense


def diagnostics(dense):
    """The same overlap diagnostics the synthetic harness reports, so the numbers are
    directly comparable to results/overlap_sweep.csv."""
    K, M = dense.shape
    sum_elements = dense.astype(np.int32).sum(axis=0)
    active = int((sum_elements > 0).sum())
    common_frac = float((sum_elements > 1).sum()) / max(active, 1)
    mean_overlap = float(sum_elements[sum_elements > 0].mean()) if active else 0.0

    # Pairwise disjointness: concepts a,b are disjoint iff they never co-occur on a token.
    # dense @ dense.T counts co-occurrences; zero off-diagonal entries are disjoint pairs.
    co = dense.astype(np.int32) @ dense.astype(np.int32).T
    np.fill_diagonal(co, 1)  # a concept is never "disjoint with itself"
    disjoint_pairs = int((co == 0).sum())
    total_pairs = K * (K - 1)

    return {
        "K": K,
        "M": M,
        "active_tokens": active,
        "coverage": round(active / max(M, 1), 3),
        "mean_overlap": round(mean_overlap, 3),
        "common_frac": round(common_frac, 3),
        "disjoint_pairs": disjoint_pairs,
        "disjoint_frac": round(disjoint_pairs / max(total_pairs, 1), 3),
        "max_overlap": int(sum_elements.max()) if M else 0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feats", default="../neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats")
    ap.add_argument("--max_sents", type=int, default=200,
                    help="sentences to read; ~11 tokens each, so 200 ~= 2048 tokens (synthetic M)")
    ap.add_argument("--K", type=int, default=15, help="concepts to keep (matches synthetic default)")
    ap.add_argument("--min_support", type=int, default=5, help="drop concepts rarer than this")
    ap.add_argument("--out", default="results/real_token_stats.csv")
    args = ap.parse_args()

    tokens = load_tokens(args.feats, args.max_sents)
    print(f"[real] loaded {len(tokens)} tokens from {args.max_sents} sentences\n", flush=True)

    # Each arm is a concept vocabulary; only the vocabulary changes, never the token set.
    arms = [
        ("tag-only (disjoint control)", ["tag"]),
        ("dep-only", ["dep"]),
        ("tag+dep", ["tag", "dep"]),
        ("lemma+synset", ["lemma", "synset"]),
        ("all categories (token regime)", CATEGORIES),
    ]

    rows = []
    for name, cats in arms:
        concepts = select_concepts(tokens, cats, args.K, args.min_support)
        if not concepts:
            print(f"--- {name}: no concepts met min_support={args.min_support}, skipped ---")
            continue
        dense = build_dense(tokens, concepts)
        row = {"arm": name, "categories": "+".join(cats)}
        row.update(diagnostics(dense))
        rows.append(row)
        print(f"--- {name} ---")
        print(f"    concepts: {', '.join(f'{c}={v}' for c, v in concepts[:6])}"
              f"{' ...' if len(concepts) > 6 else ''}")
        print(f"    {row}\n", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")

    print("\n=== vs SYNTHETIC baselines (results/overlap_sweep.csv) ===")
    print(f"{'arm':<32} {'mean_overlap':>13} {'common_frac':>12} {'disjoint_pairs':>15}")
    print(f"{'synthetic disjoint (p_add=0)':<32} {1.0:>13} {0.0:>12} {210:>15}")
    print(f"{'synthetic token   (p_add=1)':<32} {2.863:>13} {1.0:>12} {0:>15}")
    for r in rows:
        print(f"{r['arm']:<32} {r['mean_overlap']:>13} {r['common_frac']:>12} {r['disjoint_pairs']:>15}")


if __name__ == "__main__":
    raise SystemExit(main())
