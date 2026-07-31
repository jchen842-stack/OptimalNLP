# Research Diary — Summer D5.4: The IoUs Were Measuring Density, Not Explanation

*Revises D5.2 and D5.3. Both entries reported search IoUs for real units — 0.46–0.65 — and
read them as weak-but-real explanation quality, concluding that the concept vocabulary rather
than the search was the binding constraint. That reading does not survive retaining the
winning formula. At the activation range we were using, the trivial "fire on everything"
formula scores IoU equal to the neuron's density, our densities were ~0.50, and our IoUs were
~1.01x that. The number we were interpreting was the density.*

---

## 0. Headline

Every IoU in D5.2 and D5.3 sits within about 1% of what a formula that fires on **every
token** would score. The search was not finding weak explanations; it was finding
high-coverage blankets, and IoU at density 0.5 cannot tell those apart from real ones.

The instrumentation that caught it is one line: the search returns `(best_label, best_iou,
...)` and `src/real_token_search.py` was discarding `best_label` with `_`. Two diary entries
reported a score with no record of the formula that produced it.

---

## 1. The suspicion

Best IoU for every real unit almost exactly equals that unit's own activation density:

| unit | arm | density | IoU |
|---|---|---|---|
| 109 | trained | 0.536 | 0.530 |
| 115 | trained | 0.529 | 0.533 |
| 369 | trained | 0.565 | 0.591 |
| 389 | trained | 0.494 | 0.500 |
| 506 | trained | 0.453 | 0.464 |
| 396 | untrained | 0.623 | 0.607 |
| 92 | untrained | 0.626 | 0.615 |
| 510 | untrained | 0.552 | 0.563 |
| 88 | untrained | 0.640 | 0.649 |

r = 0.98, mean |IoU − density| = 0.011.

A formula F that fires everywhere gives |F∩N|/|F∪N| = d/1 = d. So IoU ≈ density is the
signature of a near-universal formula carrying no information about the neuron.

The proxy neuron, on identical masks and identical code, scores IoU 0.905 at density 0.43 —
2.1x its density. So the pipeline was sound and the problem was specific to real targets.

---

## 2. Instrumentation (`results/formula_dump_K15.csv`)

Added to the per-unit output, downstream of the search and touching nothing it reads:
`formula`, `formula_cov`, `n_and`/`n_or`/`n_not`, `or_categories`, `max_same_cat_or`.
`formula` and `formula_cov` are now **permanent columns** — their absence is why this went
undetected across two entries.

Every IoU reproduced the D5.2/D5.3 CSVs to four decimal places, so this is the same search.

One incidental find: upstream's `BinaryNode.to_str` forwards a `sort=` kwarg that
`Leaf.to_str` does not accept, so it raises on any leaf child. Formulas are rendered by hand.

---

## 3. What the formulas actually are

```
tr 109  (((const=NP OR const=VP) OR dep=nsubj) OR dep=punct)          cov 0.956
tr 115  (((const=NP AND (NOT const=PP)) OR const=VP) OR tag=IN)       cov 0.788
tr 369  (((const=VP OR const=PP) AND (NOT dep=nsubj)) OR dep=punct)   cov 0.756
tr 389  (((const=NP AND (NOT tag=DT)) OR const=VP) OR dep=punct)      cov 0.859
tr 506  (((const=NP OR const=VP) AND (NOT dep=prep)) OR dep=punct)    cov 0.829
un  88  (((const=NP AND (NOT tag=DT)) OR const=VP) OR const=PP)       cov 0.839
un  92  (((const=NP OR const=VP) OR const=PP) OR dep=punct)           cov 0.954
un 396  (((const=NP OR const=VP) OR dep=nsubj) OR dep=punct)          cov 0.956
un 510  (((const=NP OR dep=punct) AND (NOT lemma=a)) OR const=VP)     cov 0.880
proxy   ((dep=det OR tag=IN) OR dep=pobj)                             cov 0.424
```

**The mechanism I predicted was wrong.** I expected an OR over a near-partition of one
annotation category — mutually exclusive concepts unioning additively with no interaction
penalty. That is not what happened: `max_same_cat_or` is 1–3 of ≤4 terms and the OR terms mix
`const` with `dep`/`tag`/`lemma`. Six of nine formulas spend an operator on an `AND (NOT …)`,
a narrowing move the degenerate story says should never pay.

The actual mechanism is that `const` spans **nest** — a token is inside an NP *and* a PP at
once — so `const` is precisely not disjoint, and it supplies the three highest-coverage
concepts in the K=15 vocabulary: `const=NP` 0.661, `const=VP` 0.543, `const=PP` 0.428.
Nothing else clears 0.23. Every winning formula is anchored on NP or VP.

Two further signs this is an attractor in the concept set rather than anything about the
neurons: untrained 396 and trained 109 returned the **identical formula** despite different
activations and densities; and the search's advantage over the trivial baseline is

| | IoU / density |
|---|---|
| real units | 0.974 – 1.047, mean 1.008 |
| proxy | 2.090 |

Three real units score **below** the all-firing baseline. The single best individual concept
does worse still (0.355–0.541). The AND-NOT is real optimization buying 1–5%, not an
explanation.

---

## 4. Withdrawal

The IoU-based conclusions in D5.2 §3 and D5.3 §3–§5 are withdrawn. Specifically:

- Any reading of real-unit IoU 0.46–0.65 as "somewhat explainable" is measuring density.
- The D5.3 trained-vs-untrained IoU comparison tracks the density gap and survives no
  correction for it. That entry already flagged the difference as unestablished at n=5; the
  reason is worse than sample size.
- D5.3 §5's conclusion that the **vocabulary** is the binding constraint is unsupported. The
  binding constraint was the activation range, which D5.5 corrects.

**Untouched:** the frontier-explosion and search-cost results in D5.1–D5.3. They do not
depend on what the winning formula was.

---

## 5. Why it took two entries

A score with no artifact attached cannot be sanity-checked. `best_iou` is a float that always
looks plausible; `formula_cov = 0.956` is obviously wrong the moment you see it. The fix is
not vigilance, it is retaining the object alongside the score, which is now permanent.

---

## 6. Artifacts

- Instrumentation: `src/real_token_search.py` (formula columns, now permanent)
- Dump: `results/formula_dump_K15.csv` (M=2,547, alpha=0 — superseded by D5.5, retained for
  provenance)
- Next: D5.5, which identifies the activation range as the cause and re-runs the protocol.
