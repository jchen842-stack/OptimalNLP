# results/ manifest

One line per file: what produced it, which diary entry it belongs to, and whether it is
current or superseded.

Superseded CSVs carry two leading `#` comment lines naming their replacement. Read them with
`pandas.read_csv(path, comment='#')` — a plain `csv.reader` will treat the comment as a row.

## Current

| file | produced by | diary | notes |
|---|---|---|---|
| `ENVIRONMENT.md` | `src/env_info.py` | D5.5 | python/numpy/scipy/torch/platform versions for these runs |
| `METHOD_NOTES.md` | hand-written | D5.4–D5.5 | pre-registrations, outcomes, recurring error shapes. **Read before quoting any number.** |
| `MANIFEST.md` | hand-written | — | this file |
| `unique_elements.csv` | `src/unique_elements.py` | D5.5 | unique-element fraction (paper §4.3) at M=2,547 and M=24,199 |
| `alpha_sweep_K15.csv` | `src/real_token_search.py` → `src/alpha_sweep_report.py` | D5.5 | Phase A. 40 beam-200 runs, alpha ∈ {0.5, 0.2, 0.1, 0.05}, both arms, M=24,199. No timeouts |
| `beam_vs_exact_K15.csv` | `src/real_token_search.py` → `src/phaseB_report.py` | D5.5 | Phase B, **length 4**. 27 pairs, 4 timed out (excluded from derived stats, n=23) |
| `phaseB_report.txt` | `src/phaseB_report.py` | D5.5 | full printed output for the above, including the pre-registered verdicts |
| `beam_vs_exact_L3_K15.csv` | `src/real_token_search.py` → `src/phaseB_report.py` | D5.5 | Phase B at **length 3**, matched to the paper's max length. 27 pairs, no timeouts |
| `oracle_L3.txt` | `tests/test_bruteforce_oracle.py` | D5.5 | brute-force oracle at length 3. In-grammar max == search on all 3 cases; expressiveness gap +0.0000% |
| `oracle_L4.txt` | `ORACLE_LENGTH=4 tests/test_bruteforce_oracle.py` | D5.5 | same at length 4. In-grammar max == search on all 3; expressiveness gap +0.1586% on untrained unit92 — the method's grammar cannot express `OR NOT` |
| `phaseB_report_L3.txt` | `src/phaseB_report.py` | D5.5 | full printed output for the length-3 grid; contains the band comparison |

## Synthetic baselines — still valid, not superseded

These measure the search on synthetic masks and are the comparison points the real-data runs
are quoted against. Different experiment, not replaced.

| file | produced by | diary | notes |
|---|---|---|---|
| `overlap_sweep.csv` | `src/synthetic_overlap_sweep.py` | D5.0–D5.1 | frontier size vs concept overlap; the disjoint→token regime shift |
| `beam_sweep.csv` | `src/synthetic_overlap_sweep.py` | D5.1 | beam width sweep on synthetic masks |
| `scale_curve.csv` | `src/synthetic_overlap_sweep.py` | D5.1 | cost vs M |
| `scale_beam.csv` | `src/synthetic_overlap_sweep.py` | D5.1 | cost vs M under a beam cap |

## Superseded

All of these predate two corrections: the corpus was M=2,547 tokens (now 24,199), and the
activation range was alpha=0 — a threshold at 0, which for a tanh-bounded LSTM state splits
~50/50 and gives density ~0.50. At density 0.5 the all-firing formula scores IoU = 0.5 by
construction, so IoUs from that era are ~1.01x chance and are **not** explanation-quality
measurements. Retained because the diary cites them.

| file | superseded by | diary | why |
|---|---|---|---|
| `real_units_K15.csv` | `alpha_sweep_K15.csv` | D5.2 | M=2,547, alpha=0. Untrained real units |
| `trained_units_K15.csv` | `alpha_sweep_K15.csv` | D5.3 | M=2,547, alpha=0, **and** not density-matched — its trained-vs-untrained gap tracks density, not training |
| `trained_units_K15_matched.csv` | `alpha_sweep_K15.csv` | D5.3 | density-matched fix of the above, still at the wrong activation range |
| `formula_dump_K15.csv` | `beam_vs_exact_K15.csv` | D5.4 | first run to retain the winning FORMULA; showed high-coverage blanket formulas. Superseded because the activation range, not the formula, was the root cause |
| `real_token_search.csv` | `alpha_sweep_K15.csv` | D5.1 | M=2,547, proxy neuron (OR of 3 concepts + noise), not real units |
| `real_K30.csv` | length-3 grid | D5.1 | M=2,547, proxy neuron, K=30 |
| `real_K50.csv` | length-3 grid | D5.1 | M=2,547, proxy neuron, K=50 |
| `real_beam_K50.csv` | — | D5.1 | M=2,547, proxy neuron, beam sweep at K=50 |
| `real_beam_units_K50.csv` | — | D5.2 | M=2,547, alpha=0 real units, beam sweep at K=50 |
| `real_lemma_control_K50.csv` | lemma K=50 control in the length-3 grid | D5.1 | M=2,547, proxy neuron, disjoint control |
| `real_token_stats.csv` | `unique_elements.csv` **in part** | D5.1 | M=2,547 mask diagnostics across 5 concept arms. `unique_elements.csv` covers only the all-categories arm; the tag / dep / tag+dep / lemma+synset arms here have **no replacement** |

## Not in the repo

`results/*.npz` (activations) and `models/*.pth` (checkpoints) are gitignored. See
`REPRODUCE.md` for regeneration and `models/README.md` for the training provenance.
