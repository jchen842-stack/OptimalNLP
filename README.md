# OptimalCE — NLP Extension

Applying **OptimalCE** (optimal compositional explanations of neurons) to NLP: the elements
are SNLI **tokens** rather than image pixels, and the concepts are spaCy/benepar/WordNet
annotations (`lemma`, `tag`, `dep`, `ent`, `synset`, `const`).

> **Status as of 2026-07-31.** Diagnosis complete and verified: the near-chance IoUs reported
> in D5.1–D5.4 were an artifact of the activation range, not a property of the neurons, and
> the correctness audit passes 10/10. **No heuristic has been modified yet.** Outstanding:
> formula lengths 5–6, the paper's 0.005 activation range (needs M ≥ 40,000 tokens), and the
> `OR-NOT` expressiveness gap found in the method's formula grammar.

| directory | what is in it |
|---|---|
| `src/` | the harness — masks, activations, search, reporting (see the table below) |
| `results/` | every CSV and report. **Start at [`results/MANIFEST.md`](results/MANIFEST.md)** |
| `diary/` | the research narrative, D5 through D5.5 — start at [`diary/README.md`](diary/README.md) for the index and what is superseded |
| `verify/` | the correctness audit scripts, plus `run_all.sh` |
| `tests/` | the brute-force oracle over the formula space |
| `patches/` | the one upstream diff (`MAX_FRONTIER_SIZE`), against `70805299` |
| `infra/`, `scripts/` | Nautilus pod/PVC manifests and the pod setup script |

**Verify it yourself:** `./verify/run_all.sh` runs all ten checks and prints a
PASS / FAIL / CANNOT VERIFY table, exiting nonzero on failure. Checks needing gitignored
inputs (the 54 MB checkpoint, the `.npz` activations) report CANNOT VERIFY and name the file
rather than crashing. Full detail in [`VERIFICATION.md`](VERIFICATION.md); how to regenerate
the inputs in [`REPRODUCE.md`](REPRODUCE.md).


### The four scripts

| script | what it does |
|---|---|
| `src/real_token_masks.py` | parses `snli_1.0_dev.feats` into (K, M) concept masks over tokens; overlap and unique-element diagnostics |
| `src/real_activations.py` | runs the SNLI encoder for per-token activations, binarised at one or more activation ranges (`--alphas`) |
| `src/real_token_search.py` | runs the optimal (or beam) search against those masks and neurons — the main experiment driver |
| `src/train_snli_encoder.py` | trains the Bowman SNLI classifier behind the trained arm (provenance in `models/README.md`) |

Reporting: `src/alpha_sweep_report.py`, `src/phaseB_report.py`, `src/corrected_metrics.py`,
`src/unique_elements.py`, `src/env_info.py`.

### Where results live

- **`results/MANIFEST.md`** — one line per results file: what produced it, which diary entry
  it belongs to, and whether it is current or **superseded**. Several CSVs cited by D5.1–D5.4
  are superseded and carry a `#` header saying so; read those with
  `pandas.read_csv(path, comment='#')`.
- **`results/METHOD_NOTES.md`** — pre-registered predictions and their outcomes, plus the
  recurring methodological errors found in this project. **Read this before quoting any
  number from `results/`.**
- `VERIFICATION.md` — the correctness audit. `REPRODUCE.md` — running it from a clean clone,
  and every input that is not in the repo. `results/ENVIRONMENT.md` — pinned versions.
- `tests/test_bruteforce_oracle.py` — exhaustive oracle over the formula space.
  `verify/` — the audit scripts.

### Status of the numbers

The D5.1–D5.4 IoU results are **superseded**. They were measured at an activation range of
alpha=0 (threshold at 0), which for a tanh-bounded LSTM state gives density ~0.50; at that
density the all-firing formula scores IoU = 0.5 by construction, so those IoUs sit ~1.01x
above chance and are not explanation-quality measurements. The corpus also grew from 2,547
to 24,199 tokens. Current results are the alpha sweep and the beam-vs-exact grids listed in
`results/MANIFEST.md`.

### What has and has not been verified

Eight independent checks pass (`VERIFICATION.md`), each recomputing a quantity a second way
rather than re-reading the code that produced it: token-order alignment against a separate
parser; padding correctness against unpadded single-sentence runs, with a working negative
control; the `MAX_FRONTIER_SIZE` patch being bit-identical to clean upstream when disabled;
our IoU matching upstream's `metrics.iou` to full float precision; masks matching hand-parsed
raw annotation lines; no vision-stub code executing on the NLP path; the checkpoint
re-evaluating to its stored 0.7934 dev accuracy exactly; and per-unit binarisation. On top of
that, `tests/test_bruteforce_oracle.py` exhaustively enumerates the formula space and
confirms the search returns the true optimum **of the space the method can construct** —
validating the quantity helpers and the heuristic end to end without a second
implementation.

That oracle also surfaced a property of the upstream method: `expand_node` only ever appends
a bare literal, and negation only ever appears as AND-NOT, so `OR NOT` and compound right
children are unreachable. At length 3 this costs nothing measurable; at length 4 it costs
+0.16% IoU on one of the units tested. No reported number changes, but "exact search" means
*optimal within the method's formula grammar*. See `VERIFICATION.md` check 10 and
`results/METHOD_NOTES.md`.

> **CORRECTION, 2026-08-01 — "the search is optimal in its own space" is FALSE.**
> That clause stood here until 2026-08-01 and has been struck. Exhaustively enumerating all
> 30,375 in-grammar formulas (K=15, length 3, M=24,199, min_support 5) over all 27 pairs,
> upstream's exact search **misses its own in-grammar optimum on 2 of 27**: trained a=0.2
> unit88 (+0.9274%) and trained a=0.05 unit86 (+4.7434%). Found by running upstream's own
> `beam_optimal.py`, which returns a *higher* IoU than "exact" on both.
> **The cause is not yet identified.** A first attribution (the disjoint branch of
> `estimate_label_quantities`) was published and then falsified the same day — see the
> RETRACTION section of `results/METHOD_NOTES.md`. Check 10 passed on 3 cases, all of which
> happen to be among the 25 that tie; it is now `10a` + `10b`, the latter covering all 27.
> Length-4 exact numbers have never been oracle-checked and should be read as **unverified**.

> **`CODE_WALKTHROUGH.md` citation status.** Restored 2026-08-01 after deletion in `733ff0c`.
> Of its **99** `file:line` citations (its own header said 98): **39 verified at the cited
> line, 35 resolve with content moved, 25 unverifiable by any tooling** (no extractable
> anchor — a permanent blind spot, settleable only by a human). Re-measure with
> `python verify/check_walkthrough_citations.py`; output in `results/walkthrough_citations.txt`.

**Not verified:** the beam path (`MAX_FRONTIER_SIZE = 200`) has no independent reference and
is checked only at the `None` setting; the expressiveness gap above is measured on three
cases at one length, not characterised; the original training seed is unrecoverable from
surviving artifacts; untrained-arm weights depend on the torch RNG and hence the torch
version, which was not recorded for the original runs (it is now); and which length-4 runs
hit the time budget is machine-dependent, so the exact n for those statistics will differ on
other hardware.

---

# Original README (D5–D5.3)

Unchanged. Some findings below have been superseded; see the status note above and
`results/METHOD_NOTES.md`.

Extending **OptimalCE** (optimal compositional explanations of neurons) from vision to
NLP/tokens. This repo holds *our* work — experiment harness, results, research diary, the
fix as a patch, and the infra manifests — kept separate from the upstream method code so it
is portable across Nautilus namespaces and PVCs.

Upstream method (not vendored here): <https://github.com/aiea-lab/optimal-compositional-explanations>
pinned at the commit in [`UPSTREAM`](UPSTREAM) (`70805299`).

## Why this repo is standalone

The compute + data (Cityscapes, activation cache) currently live on a Nautilus PVC
(`optimalce-data`, RWO). To avoid being locked to one namespace/PVC, everything reproducible
lives here and can be re-applied to any fresh checkout:

- **Our code** (`src/`) is self-contained and imports the upstream package at runtime.
- **The fix** (`patches/`) is a diff against the pinned upstream commit — no upstream source
  is copied into this repo.
- **Infra** (`infra/`) has the pod/PVC manifests so a new namespace is a `kubectl apply` away.

## Layout

```
src/synthetic_overlap_sweep.py   synthetic harness (overlap / scale / beam modes)
src/real_token_masks.py          real SNLI token concept masks + overlap statistics
src/real_token_search.py         optimal search on real token masks (wall + beam fix)
src/real_activations.py          real per-token unit activations (Bowman encoder)
src/train_snli_encoder.py        train the Bowman SNLI classifier (0.7934 dev)
src/unique_elements.py           unique-element fraction (the paper's 4.3 variable)
src/corrected_metrics.py         lift / normalised fit / coverage-matched IoU null
src/alpha_sweep_report.py        Phase A report over the activation-range sweep
src/phaseB_report.py             beam-vs-exact join, band comparison, prediction scoring
src/env_info.py                  environment capture -> results/ENVIRONMENT.md
verify/*.py, verify/run_all.sh   correctness audit; run_all.sh prints a PASS/FAIL table
tests/test_bruteforce_oracle.py  exhaustive oracle over the formula space
VERIFICATION.md                  the audit
REPRODUCE.md                     running it from a clean clone
requirements.txt                 pinned dependency versions
patches/0001-frontier-beam-fallback.patch   the fix, vs upstream 70805299
results/*.csv                    recorded experiment outputs
diary/README.md                  diary index: every entry, what it revises, what is superseded
diary/summer_d5.md               research diary (synthetic reproduction + fix)
diary/summer_d5.1.md             research diary (real SNLI token masks; revises D5)
diary/summer_d5.2.md             research diary (real neuron activations; revises D5.1)
diary/summer_d5.3.md             research diary (trained neurons; revises D5.2)
diary/summer_d5.4.md             research diary (IoU was tracking density; revises D5.2/D5.3)
diary/summer_d5.5.md             research diary (activation range corrected + audit; revises D5.4)
infra/{pvc,pod,pod-cpu}.yaml      Nautilus manifests
scripts/setup_pod.sh             clone upstream + apply patch + sync harness into a pod
UPSTREAM                         pinned upstream commit
```

## The result in one paragraph

⚠️ **Throughout this file, "optimal search" means optimal *within the method's formula
grammar*** (D5.5 §4). `expand_node` only ever appends a bare literal, so formulas are
left-deep, negation appears only as AND-NOT, and there is no `OR NOT`. A brute-force oracle
confirms the search attains the in-grammar optimum exactly; what the grammar cannot express
costs **+0.0000% at length 3** and **+0.1586% at length 4** on the cases tested
(`VERIFICATION.md` check 10).

Under vision's disjoint concepts the optimal search is cheap at any scale; under token-like
concept **overlap** the disjoint fast-path vanishes, admissible ceilings stop separating
candidates, and the best-first frontier balloons to OOM (wall at formula length ≥ 4). The
fix is the paper's §3.3 escape hatch — a **size-bounded beam fallback** (`MAX_FRONTIER_SIZE`
in `compositional/optimal.py`). Key finding: a size cap bounds *memory* but the beam must be
*tight* to also bound *time* — a wide cap (2000) still times out, while beam 100–500 turns a
non-terminating OOM search into 1.4–30 s with a usable explanation. See
[`diary/summer_d5.md`](diary/summer_d5.md) for the full write-up and tables.

## Real SNLI token data

The synthetic sweep turned overlap with a knob. This section removes the knob: the masks are
real spaCy/benepar/WordNet annotations of SNLI dev tokens (`snli_1.0_dev.feats`, pre-computed
and checked into the Compositional Explanations of Neurons NLI codebase, so no spaCy, benepar,
GPU, or model checkpoint is needed).

**The controlled contrast comes free with the data.** Only the admitted concept *vocabulary*
changes; the token set is identical in every arm:

- single-valued categories (`tag`, `dep`, `lemma` alone) are **disjoint by construction** — a
  token has exactly one POS tag, one dep label, one lemma;
- all categories at once is **overlapping by construction** — a token simultaneously has a
  lemma AND a tag AND a dep AND a synset AND several constituents.

So the regime shift is a property of NLP annotation, not of our generator.

### Findings

1. **The disjoint arms reproduce the synthetic control exactly** — `mean_overlap` 1.0,
   `common_frac` 0.0, every concept pair disjoint, and a flat, cheap search at every size.
   This validates the synthetic generator against real masks.
2. **Real overlap is *worse* than the synthetic worst case**: `mean_overlap` 3.18 (K=15) to
   4.20 (K=50), against 2.863 for synthetic `p_add=1`, with tokens in up to 7 concepts at once.
3. **But real overlap is block-structured, and that matters.** Concepts within a category are
   mutually exclusive, so ~half the concept pairs stay disjoint (110/210 at K=15) where the
   synthetic generator drove disjoint pairs to 0, so the disjoint fast-path partially survives.
   ⚠️ **Qualified by D5.5 §1.** `disjoint_pairs` is a property of the *vocabulary*. The
   paper's §4.3 variable is the fraction of *elements* carrying exactly one concept, which is
   **13.9%** (not ~51%) and *falls* as the corpus grows, while disjoint_pairs stays flat. The
   two statistics point opposite ways; the element-level one is the binding statistic.
   ⚠️ **Partly superseded by D5.2.** This point originally continued: "and at K=15/length=4 the
   real search terminates in 6.5 s where synthetic predicted a timeout." That held only for the
   *proxy* neuron. With real unit activations the same case runs 14 s → non-termination
   (1 of 5 units), so the tractability gain was substantially the easy target, not the block
   structure. The block structure is real; its share of the effect was overstated. See
   [`diary/summer_d5.2.md`](diary/summer_d5.2.md).
4. ⚠️ **Superseded by D5.5 §3 — the wall is on the LENGTH axis, not the K axis.** At
   M=24,199, exact search at length 3 terminates in 4.7–8.5 s even at K=50 all-categories,
   while length 4 at K=15 takes 200–1,500 s with 4 timeouts. K=15→50 at length 3 costs ~4x;
   length 3→4 at K=15 costs 100x or more. The table below is at M=2,547 with a proxy neuron
   and stands as measured; its *reading* — that K is the binding axis — does not.

   **The wall is nonetheless real**, and it arrives on the K axis (`results/real_K{30,50}.csv`):

   | K (all categories, length 4) | peak frontier | time | terminates |
   |---|---|---|---|
   | 15 | 3,085 | 6.5 s | yes |
   | 30 | 15,027 | 115.7 s | barely |
   | 50 | 48,014 | 220 s | **no** |

   Against a **K-matched** disjoint control (`lemma`-only, K=50, all 2450 pairs disjoint):
   257 frontier, 2 nodes visited, 0.25 s. That is 187× the frontier, and the qualitative gap
   is starker than any ratio — at identical K and length the disjoint search visits two nodes
   and finishes, while the overlapping one never terminates.
5. **The beam fallback rescues the real wall** (`results/real_beam_K50.csv`), and more cleanly
   than on synthetic data — quality plateaus early instead of wobbling. (Proxy neuron,
   M=2,547; the IoUs here are not affected by the density problem in the note above, because
   the proxy's density is 0.43, but they are proxy numbers and not real-unit numbers.)

   | beam | time | IoU | terminates |
   |---|---|---|---|
   | none | 220 s | — | no |
   | 100 | 0.54 s | 0.6875 | yes |
   | 200 | 2.3 s | 0.738 | yes |
   | 500 | 9.0 s | 0.738 | yes |
   | 1000 | 18.6 s | 0.7386 | yes |

   Beam 200 buys full attainable IoU for 2.3 s against a search that otherwise never returns.

### Scope and caveats

- **Neurons: proxy, untrained, and trained.** Results 1–5 above use a *proxy* neuron (OR of
  three concepts + 5% noise). D5.2 added **real per-token unit activations** via
  `src/real_activations.py`, which builds the encoder directly and reads
  `TextEncoder.get_states()` — the upstream `load_for_analysis` cannot be used here (it needs a
  checkpoint for weights *and* vocab) and upstream NLI neurons live on the *example* axis, not
  the token axis. D5.3 then trained a Bowman SNLI classifier to **0.7934** dev accuracy
  (`src/train_snli_encoder.py`) and reran with trained units. At K=15/length=4:

  ⚠️ **Superseded by D5.4 — the "best IoU" column below is WITHDRAWN, and so is the
  conclusion drawn from it two paragraphs down.** These IoUs were measured with the
  activation range at alpha=0 (threshold at 0), giving density ~0.50. At that density the
  "fire on everything" formula scores IoU = density by construction, and every value below
  sits ~1.01x that: the number is the density, not the fit. The winning formulas were
  high-coverage blankets, e.g. `(((const=NP OR const=VP) OR dep=nsubj) OR dep=punct)` at
  coverage 0.956. **The frontier and time columns are unaffected** — they do not depend on
  the IoU. Corrected at a proper activation range (D5.5 §2), trained lift reaches 3.41 and
  the trained arm *separates* from untrained.

  | neuron | frontier (med) | time (med) | best IoU (med) | timeouts |
  |---|---|---|---|---|
  | proxy | 3,085 | 6.5 s | 0.905 | 0/1 |
  | untrained | 18,760 | 145.6 s | 0.611 | 1/5 |
  | trained (density-matched) | 15,491 | 86.5 s | 0.530 | 0/5 |

  **Trained neurons land next to untrained, not next to the proxy** — the proxy is a 5×
  frontier / 13× time outlier against a properly trained model. That gap is an order of
  magnitude larger than the noise and is the robust finding here.
  ⚠️ **Superseded by D5.4/D5.5 — the paragraph below is right about the conclusion and wrong
  about the reason, and its own supporting numbers are withdrawn.** The 600-unit
  density-matched test (trained 0.431 vs untrained 0.420, p=0.008) compares IoUs at
  density ~0.50 and is therefore comparing densities. And the final claim — that the concept
  **vocabulary** is the binding constraint — is **unsupported**: the binding constraint was
  the activation range. At alpha=0.05 the trained arm reaches lift 3.41 against untrained
  2.07, where at alpha=0.5 the two are indistinguishable. The D5.3 question is reopened with
  an instrument that works (D5.5 §2, §5).

  **The trained-vs-untrained difference is not established.** At n=5 per arm neither the 17%
  frontier gap nor the search-IoU gap (0.530 vs 0.611) survives scrutiny: the untrained arm has
  one censored timeout, which biases its IoU mean upward, and a density-matched test over all
  600 eligible units *reverses* the IoU sign (trained 0.431 vs untrained 0.420, p=0.008,
  d=0.20 — a small effect the other way). Also robust: **both** arms sit near IoU 0.4–0.6, so
  real neurons of either kind are poorly described by this concept vocabulary, which points at
  the vocabulary rather than the search as the binding constraint. See
  [`diary/summer_d5.3.md`](diary/summer_d5.3.md) §4.
  (The OpenNMT en-de BiLSTM in `nli/models/README.md` belongs to the MT half of that
  codebase and is *not* used by the NLI path.)
- **Closed-class controls saturate.** `tag`/`dep` cap out at ~23 concepts meeting
  `min_support`, so they cannot supply a K-matched control at K≥30; use the `lemma` arm there.
  An earlier 649× figure compared K=50 overlap against a K=23 control and is superseded by the
  187× K-matched number above.
- 200 sentences (~2,547 tokens) were used to keep M near the synthetic M=2048. **Superseded:
  the current corpus is 2,000 sentences / 24,199 tokens** (D5.5 §1), which is what makes the
  paper's 0.005 activation range approachable at all.

```bash
python src/real_token_masks.py                      # overlap statistics, all arms
python src/real_token_search.py --arms lemma all \
    --lengths 4 --K 50 --beam_list none 200         # the wall and the fix
```

## Reproduce (current or a new namespace)

*For local reproduction the canonical guide is now [`REPRODUCE.md`](REPRODUCE.md), which
lists every input that is not in the repo and was tested from a clean clone. The pod/PVC
workflow below remains accurate.*

Everything is **CPU-only** — the optimal search is Python `heapq` + estimation over tiny
boolean tensors; no GPU is needed (GPU only ever computed the vision activations, which the
synthetic harness bypasses).

```bash
# 0. (new namespace) provision storage + a CPU pod
kubectl apply -f infra/pvc.yaml
kubectl apply -f infra/pod-cpu.yaml        # optimalce-cpu; use infra/pod.yaml if you need a GPU

# 1. clone upstream onto the PVC, apply our patch, sync the harness
POD=optimalce-cpu ./scripts/setup_pod.sh

# 2. run the experiments (see modes below)
kubectl exec $POD -- bash -lc '
  cd /workspace/data/optimal-compositional-explanations &&
  PYTHONPATH=. python -u nlp_extension/synthetic_overlap_sweep.py --mode scale \
    --cap 200000 --time_budget 30 --out nlp_extension/results/scale_curve.csv'
```

### Harness modes
- `--mode overlap` — hold size fixed, sweep concept overlap (the mechanism).
- `--mode scale` — sweep K×length, contrasting a disjoint control vs the token regime (the wall).
- `--mode beam --beam_list none 100 200 500 1000 2000` — fix a hard case, sweep beam width
  (validate the fix; the tradeoff curve).

The harness stubs the upstream vision-only imports (`detectron2`, `cityscapesscripts`) via a
meta-path finder, so it runs on a bare CPU container without those installed. The stubs now
return objects that raise `ImportError` on any use, rather than the builtin `object` — so a
stubbed symbol that is actually called fails loudly instead of silently returning a plain
instance. Nothing on the NLP path touches them (`VERIFICATION.md` check 6).

## Cleanup / moving off the PVC

`kubectl delete pod optimalce-cpu` frees compute (no GPU is held by the CPU pod anyway).
Keep the PVC only for the Cityscapes data + activation cache; none of *this* repo depends on
it. To migrate: push this repo to your own GitHub, then `setup_pod.sh` against a pod in the
new namespace.
