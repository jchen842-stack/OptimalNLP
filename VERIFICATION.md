# Correctness audit

Adversarial audit of the alpha-sweep / Phase B pipeline. Every check re-computes the
quantity a **second, independent way** and compares, rather than re-reading the code that
produced it. Scripts live in `verify/` and are re-runnable.

Environment: `~/miniconda3/envs/compexp/bin/python`, run from the repo root.

| # | check | result |
|---|---|---|
| 1 | token-order alignment | **PASS** |
| 2 | padding does not offset short sentences | **PASS** |
| 3 | `MAX_FRONTIER_SIZE` patch is a no-op when `None` | **PASS** |
| 4 | our IoU matches upstream's `metrics.iou` | **PASS** |
| 5 | masks match the raw `.feats` file | **PASS** |
| 6 | vision stubs never touched on the NLP path | **PASS** |
| 7 | checkpoint reproduces 0.7934 | **PASS**, but see the finding below |
| 8 | binarisation is per-unit | **PASS** |
| 9 | someone else can run this | see `REPRODUCE.md` §6b — untrained path reproduces; trained path needs the gitignored checkpoint |
| 10 | brute-force oracle over the whole formula space | **PASS** (`tests/test_bruteforce_oracle.py`) |

**Nothing failed. Two things changed as a result of the audit** (both in §7 and §Surprises):
`models/README.md` asserted the wrong training invocation, and it has been corrected.

---

## 1. Alignment — PASS

`verify_alignment` only compares **row counts**, which cannot detect a reordering. Tested
order directly instead.

Method: the token stream is re-derived by a hand-written parser inside `verify/check_alignment.py`
that never calls `real_token_masks`. The activation-side row→token map is rebuilt by
replicating the batching in `real_activations.extract_states` and emitting the token text
behind each row. Both arms, both corpus sizes.

```sh
python verify/check_alignment.py
```

```
### M=2547: independent parser 2547 tokens, real_token_masks 2547 tokens
  token streams identical: True
  [untrained] activation row->token vs independent: 2547/2547 match
  [trained]   activation row->token vs independent: 2547/2547 match
### M=24199: independent parser 24199 tokens, real_token_masks 24199 tokens
  token streams identical: True
  [untrained] activation row->token vs independent: 24199/24199 match
  [trained]   activation row->token vs independent: 24199/24199 match
RESULT: PASS
```

20 random indices per arm per size printed by the script; all matched.

**Limitation, stated plainly:** this proves the *token stream* and my replication of the
batching agree. Check 2 is what tests the actual tensor.

## 2. Padding — PASS

`TextEncoder.get_states` uses `pack_padded_sequence(..., enforce_sorted=False)`, so torch
sorts internally and `pad_packed_sequence` restores the original order — no inverse
permutation is needed on our side. Verified empirically rather than by reading:

Method: run a mixed-length batch (shortest and longest sentence in the corpus, **17.7x**
ratio) and compare each sentence's slice against running that sentence **alone** in a batch
of 1, where no padding exists at all. Plus a negative control that must fail.

```sh
python verify/check_padding.py
```

```
shortest sentence idx 19 len 3; longest idx 1592 len 53  (ratio 17.7x)
batched rows 56 (expect 56); alone 3 + 53
max |batched - alone|  short: 5.960e-08   long: 6.519e-08
reversed batch order    short: 5.960e-08   long: 6.519e-08
negative control (deliberate wrong slice): 4.797e-01  <- must be LARGE
RESULT: PASS
```

Differences are float32 round-off. Reversing batch order changes nothing. The negative
control is 7 orders of magnitude larger, so the test can fail.

## 3. Is the patch actually a no-op? — PASS

```sh
cd /tmp && git clone https://github.com/aiea-lab/optimal-compositional-explanations upstream_clean
cd upstream_clean && git checkout 70805299fc0758951a650197bffcc792d0ccca20

cd ~/projects/optimalce-nlp
OPTIMALCE_UPSTREAM=$HOME/projects/optimalce                python verify/check_patch_noop.py
OPTIMALCE_UPSTREAM=/tmp/upstream_clean                     python verify/check_patch_noop.py
```

The diff between our tree and clean `7080529` is exactly `0001-frontier-beam-fallback.patch`
and nothing else. One identical configuration (trained a=0.2 unit 413, K=15, length 4, exact):

| field | PATCHED (`MAX_FRONTIER_SIZE=None`) | CLEAN 7080529 |
|---|---|---|
| formula | `(((const=NP AND (NOT const=VP)) AND (NOT const=PP)) AND (NOT dep=punct))` | *identical* |
| `best_iou` repr | `0.48937119072394825` | `0.48937119072394825` |
| `best_iou` hex | `0x1.f51db8af0e455p-2` | `0x1.f51db8af0e455p-2` |
| visited | 2063 | 2063 |
| expanded | 4982 | 4982 |
| estimated | 48937 | 48937 |
| peak_frontier | 8239 | 8239 |

Identical to the bit, not merely close. The "algorithm ran unmodified" claim holds for
`MAX_FRONTIER_SIZE = None`.

## 4. Does our IoU match upstream's? — PASS

```sh
python verify/check_iou.py
```

10 (formula, neuron) pairs from the actual Phase B results, via **three independent paths**:
our `n_inter/union` arithmetic, upstream `src/metrics.py:iou` on re-evaluated masks, and the
value the search itself returned. Formulas are re-parsed from their rendered strings by a
parser written for this check, so the mask is rebuilt rather than reused.

```
              unit   ours (n_inter/union)   upstream metrics.iou    search best_iou  match
   tr unit413 a0.2    0.48937119072394825    0.48937119072394825             0.4894     OK
    tr unit88 a0.2     0.2609853528628495     0.2609853528628495             0.2610     OK
    tr unit92 a0.2    0.26438502673796793    0.26438502673796793             0.2644     OK
   tr unit396 a0.1    0.14633804789245203    0.14633804789245203             0.1463     OK
   tr unit413 a0.1    0.09615384615384616    0.09615384615384616             0.0962     OK
    tr unit88 a0.1    0.19102002503128912    0.19102002503128912             0.1910     OK
    tr unit92 a0.1    0.16892479801118707    0.16892479801118707             0.1689     OK
  tr unit395 a0.05    0.10794341675734494    0.10794341675734494             0.1079     OK
  tr unit412 a0.05    0.09845330160618679    0.09845330160618679             0.0985     OK
   tr unit86 a0.05     0.2202944269190326     0.2202944269190326             0.2203     OK
RESULT: PASS
```

Agreement is to full float precision.

**Reimplementation flagged.** `synthetic_overlap_sweep.compute_quantities` and
`compute_disjoint_info` reimplement quantities upstream computes in `utils/mask_utils.py` and
`utils/optimal_utils.py`. They are not exercised against upstream equivalents here — check 3
covers them indirectly, since patched and clean produce identical `visited`/`expanded`
through the same helpers, but that is a consistency argument, not an equivalence proof
against upstream's own versions. **Closed by check 10**, which validates them end to end
against an exhaustive oracle.

## 5. Are the masks right? — PASS

```sh
python verify/check_masks.py
```

10 (concept, token) pairs, 5 positive and 5 negative, each checked against the raw `|`-delimited
chunk read straight from `snli_1.0_dev.feats` with no project code.

```
  k      m                concept  dense              hand-parsed raw field   ok
  6  21329                 tag=IN   True                               'IN'   OK
  0   2373               const=NP   True                      'VP;VP;PP;NP'   OK
  1  11982               const=VP   True                   'VP;VP;PP;NP;NP'   OK
  1  14209               const=VP   True                   'NP;VP;PP;NP;PP'   OK
  0  18528               const=NP   True                         'VP;PP;NP'   OK
  5   4943                dep=det  False                            'punct'   OK
 13  17559              dep=nsubj  False                            'punct'   OK
  9   1900                lemma=a  False                               'in'   OK
 14  16627                lemma=.  False                             'with'   OK
  3   1228                 tag=NN  False                              'VBZ'   OK

multi-valued const example: token 'embracing' const field 'VP;VP' -> ['VP', 'VP']
empty-field example: token 'women' ent field is '' ; ent concepts in K=15 vocab: [] ;
                     row sum over all concepts = 2
any concept with empty value in vocabulary: False (must be False)

independent mean overlap over active tokens: 3.189  (reported 3.189)
independent unique fraction: 0.139  (reported 0.139)
RESULT: PASS
```

`;`-splitting works, empty fields contribute no concept, and both the reported mean overlap
(3.189) and unique-element fraction (0.139) reproduce from an independent recomputation.

## 6. Are the vision stubs ever called? — PASS

Two methods, because the first has a blind spot.

**(a) Runtime attribute spy.** Wrap `_AnyModule.__getattr__` after import, then run a full
search.

```sh
python verify/check_stubs.py
```

```
stub modules present after import: ['detectron2', 'detectron2.data', 'detectron2.data.dataset_mapper']
search completed, best_iou=0.181630, visited=419
stub attribute accesses during the NLP path: 0
RESULT: PASS (no stub touched)
```

Blind spot: names bound at **import** time (`DatasetMapper = object`) are direct references
afterwards and would not show up in a `__getattr__` spy.

**(b) Execution trace.** `sys.settrace` records every file executed during a search;
cross-referenced against every upstream file that mentions the vision deps.

```sh
python verify/check_stub_calltrace.py
```

```
upstream files importing/mentioning vision deps (3):
    experiments/synthetic_overlap_sweep.py
    utils/dataset_utils.py
    src/model_wrapper.py

upstream files EXECUTED during the search (9):
    compositional/formula.py            compositional/optimal.py
    compositional/optimal_sample_heuristic.py
    compositional/optimal_sum_heuristic.py
    compositional/path_heuristic.py
    utils/general_utils.py  utils/mask_utils.py  utils/optimal_utils.py
    (+ our src/synthetic_overlap_sweep.py)

intersection (executed AND vision-tainted): []
RESULT: PASS (no tainted file executes)
```

No file that touches the vision deps executes during a search.

## 7. Does the model reproduce? — PASS, with a correction

```sh
python verify/check_model.py
```

```
checkpoint: val_acc=0.7934362934362934 epoch=2 emb=300 hid=512 vocab=33671

  --max_data 100000 (DEFAULT)      pairs=  99889 vocab= 16669  matches 33671: False
  --max_data 0 (full corpus)       pairs= 549367 vocab= 33671  matches 33671: True

RE-EVALUATED dev accuracy: 0.7934362934362934  (7809/9842)
stored val_acc           : 0.7934362934362934
match to 1e-9            : True

OOV with CHECKPOINT stoi (33671 types): 2137/24199 = 8.8%
OOV with CORPUS-REBUILT stoi (1918 types): 0/24199 = 0.0%
```

Dev accuracy reproduces **exactly** — 7809/9842 = 0.7934362934362934, matching the stored
`val_acc` to 1e-9, on a from-scratch evaluation.

**FINDING — `models/README.md` was wrong.** It stated the checkpoint came from an
all-defaults run. It did not: `--max_data` defaults to 100000, which yields a 16,669-type
vocabulary, and the checkpoint stores 33,671. Only `--max_data 0` (549,367 pairs) reproduces
that. The README has been corrected to record `--max_data 0` as required and not default.
The seed remains unverified and is now labelled as such rather than presented as recovered.

**OOV sensitivity confirms the failure mode is silent.** With the checkpoint's `stoi`, 8.8%
of tokens are OOV as claimed. Rebuilding `stoi` from the annotation corpus gives **0% OOV** —
every token maps to *some* embedding row, just the wrong one. There is no error, no warning,
and no OOV signal. This is why the check matters.

## 8. Binarisation — PASS

```sh
python verify/check_binarise.py
```

Sensitivity first: two units with deliberately different scales, where per-unit and global
thresholds must diverge.

```
SENSITIVITY (unit0 ~N(0,1), unit1 ~N(50,1), alpha=0.1):
  per-unit densities : [0.1 0.1]   <- both ~0.10 if PER UNIT
  global-threshold   : [0.  0.2]   <- 0.0 / 0.2 if GLOBAL
  verdict: PER UNIT
```

Then the real states, with the threshold recomputed via `np.percentile` (a different call
than `binarize`'s `np.quantile`):

```
  alpha  stored mean d  recomputed mean d  masks identical    min d    max d
    0.5        0.49770            0.49770             True  0.42394  0.49998
    0.2        0.19649            0.19649             True  0.12211  0.20001
    0.1        0.09380            0.09380             True  0.02161  0.10000
   0.05        0.04668            0.04668             True  0.00273  0.05000
  0.005        0.00488            0.00488             True  0.00169  0.00500
RESULT: PASS
```

Masks are bit-identical. Realised density never exceeds alpha and drifts below it — the
documented tanh saturation effect. The drift is severe in the tail: at alpha=0.1 the minimum
realised density is 0.0216, a 4.6x shortfall, and at alpha=0.05 it is 0.0027, an 18x
shortfall. This is why per-run density is load-bearing.

---

## Additions after the fixes

### n=23 is MACHINE-DEPENDENT

Four of the 27 length-4 Phase B runs hit the time budget **on this hardware** (see
`results/ENVIRONMENT.md`) and are excluded, giving n=23. On faster hardware fewer runs
time out, n is larger, and **the stratified rank correlations, the ratio-of-averages, and
the gap distribution all change** — not because anything is wrong, but because the sample
is defined by a wall-clock cutoff. Two consequences:

- Any re-run should report its own n and its own timeout list, not inherit n=23.
- The time budget is additionally **soft**: it is checked once per 256 heap pushes, and one
  run overshot to 2085s against a 1500s cap. A `halted=time` row's `time_s` is not a precise
  measurement.

The length-3 grid has **no timeouts at all**, so its n=27 is hardware-independent. Prefer it
for anything that has to be stable across machines.

### The OOV inversion — a clean number is the alarm, not the all-clear

With the checkpoint's own `stoi` (33,671 types), 8.8% of probe-corpus tokens are OOV.
Rebuilding `stoi` from the probe corpus instead gives **0.0% OOV** — and that is the broken
configuration.

This is backwards from intuition. The instinct is that a wrong vocabulary shows up as OOV
noise, so a low OOV rate reads as reassurance. The opposite holds: a vocabulary built *from*
the probe corpus covers it perfectly by construction, so every token maps to some embedding
row, none are flagged, and every row is the wrong one. There is no error, no warning, and no
signal of any kind.

**A suspiciously clean OOV rate against a pretrained checkpoint means the vocabulary is
wrong.** A nonzero OOV rate is the healthy case. This is the failure that hides, and it is
why `real_activations.py` takes `stoi` from the checkpoint and prints the rate.

## Surprises, including things that passed

1. **`--max_data` was not the default.** The one substantive error the audit found. Caught
   by a fingerprint (vocabulary size) rather than by the accuracy check, which passed
   regardless — an exact-reproducing model told us nothing about *how* it was produced.

2. **The stub `__getattr__` returns the builtin `object`, not a poison value.** So a stubbed
   `SomeClass()` would silently return a plain object instead of raising. Nothing touches it
   (check 6), so this is latent, not active — but it is a wrong-behaviour risk rather than a
   fail-fast one, and would be better as an object that raises on any use.

3. **Rebuilding `stoi` from the corpus gives 0% OOV, not a high OOV rate.** The intuitive
   expectation is that a wrong vocabulary shows up as OOV noise. It does the opposite: the
   corpus vocabulary covers the corpus perfectly by construction, so the failure is
   completely invisible.

4. **Density drift is much larger than "a few percent" in the tail** (18x at alpha=0.05).
   Already documented as a phenomenon, but the magnitude at the extreme was not.

5. **`verify_alignment` still only checks row counts.** It passed everything here because
   nothing reorders, but the function itself remains incapable of detecting a reordering.
   The guarantee comes from `enforce_sorted=False` in upstream's `get_states`, not from our
   assertion. If upstream ever changed that, `verify_alignment` would not notice.

## 10. Brute-force oracle — PASS, and it found something

The oracle asserts the search returns the true optimum **of the space `expand_node` can
construct**. It also measures, separately, how much IoU that space leaves on the table.

Closes the gap flagged under check 4. Rather than reimplement the quantity helpers, the
oracle enumerates the **entire formula space** and scores every formula with plain numpy —
no heuristic, no frontier, no shared helpers — then asserts the max equals what the search
returns.

Two spaces are enumerated.

**In-grammar** — exactly what `expand_node` (`optimal.py:554-582`) can build. Candidate
labels are plain leaves and there are only three moves:

```python
next_op == "OR"  -> F.Or(label, leaf)
next_op == "AND" -> F.And(label, leaf)
next_op == "NOT" -> F.And(label, F.Not(leaf))
```

So formulas are **left-deep** (the right child is always a bare literal), negation appears
**only as AND-NOT**, there is **no OR-NOT**, and the leftmost term is never negated. This is
the assertion target: the search must equal this max.

**Unrestricted** — all trees over literals with AND/OR and NOT on leaves, enumerated over
**masks** rather than trees so associativity and commutativity collapse for free. A strict
superset. The search is *not* expected to reach it; the difference is reported as the
**expressiveness gap**.

### The length-4 finding

At length 3 the two spaces coincide on every case tested (gap +0.0000%) and everything
matches. **At length 4 they diverge**, which the first version of this oracle reported as a
FAIL before the cause was diagnosed:

| case | in-grammar max = search | unrestricted max | gap |
|---|---|---|---|
| trained unit88 a=0.1 | `0.19102002503128912` | `0.19102002503128912` | +0.0000% |
| untrained unit92 a=0.1 | `0.19492814877430262` | `0.19523729099814222` | **+0.1586%** |
| proxy | `0.8527409974013117` | `0.8527409974013117` | +0.0000% |

The in-grammar enumeration reproduces the search's value to full precision **and recovers
the identical formula string**, on both real units:

```
untrained unit92: (((tag=NN AND NOT const=VP) OR dep=punct) AND NOT tag=IN)
trained  unit88: (((const=NP AND NOT const=PP) AND NOT tag=DT) AND NOT dep=punct)
```

So **the search is correct**. What it cannot reach is
`(tag=NN AND (dep=ROOT OR NOT const=VP)) OR dep=punct`, which needs `OR NOT`. Isolated by
running a left-deep enumeration that *permits* OR-NOT — that one reaches the unrestricted
max exactly, while the enumeration that forbids it lands exactly on the search's value.
**`OR-NOT` is the distinguishing move, not tree shape.** The 2+2 balanced shape, which only
becomes reachable at length 4, is not involved.

This is a property of the upstream method's formula grammar, not of this pipeline, and it is
invisible at length 3 — which is the length the paper reports.

```sh
python tests/test_bruteforce_oracle.py           # length 3
ORACLE_LENGTH=4 python tests/test_bruteforce_oracle.py
```

Length 3, K=15, M=24,199 — 7,360 distinct masks, three cases:

| case | brute-force max IoU | search best_iou | difference |
|---|---|---|---|
| trained unit88 a=0.1 | `0.18162962962962964` | `0.18162962962962964` | 0.000e+00 |
| untrained unit92 a=0.1 | `0.1947306198277318` | `0.1947306198277318` | 0.000e+00 |
| proxy neuron | `0.8527409974013117` | `0.8527409974013117` | 0.000e+00 |

Exact to full precision. This validates the concept masks, `compute_quantities`,
`compute_disjoint_info`, the admissible heuristic, the frontier, and the returned optimum
together.

The top level is streamed rather than materialised: at length 4 the full level is ~1.26M
masks x 24,199 elements ≈ 30 GB as bool. Streaming keeps peak memory in the hundreds of MB
and cannot change the result, since only the max is needed.

## What this audit does NOT cover
- The training seed (check 7) — unverifiable from surviving artifacts.
- Beam-search correctness at `MAX_FRONTIER_SIZE = 200`. Check 3 verifies only the `None`
  path. The beam path changes results by design, so "identical output" is not the test, and
  no independent reference beam implementation was available to compare against.
- Numerical reproducibility across machines/BLAS versions.
