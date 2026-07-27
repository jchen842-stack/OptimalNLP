# OptimalCE — NLP Extension

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
patches/0001-frontier-beam-fallback.patch   the fix, vs upstream 70805299
results/*.csv                    recorded experiment outputs
diary/summer_d5.md               research diary (synthetic reproduction + fix)
diary/summer_d5.1.md             research diary (real SNLI token masks; revises D5)
diary/summer_d5.2.md             research diary (real neuron activations; revises D5.1)
diary/summer_d5.3.md             research diary (trained neurons; revises D5.2)
infra/{pvc,pod,pod-cpu}.yaml      Nautilus manifests
scripts/setup_pod.sh             clone upstream + apply patch + sync harness into a pod
UPSTREAM                         pinned upstream commit
```

## The result in one paragraph

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
   ⚠️ **Partly superseded by D5.2.** This point originally continued: "and at K=15/length=4 the
   real search terminates in 6.5 s where synthetic predicted a timeout." That held only for the
   *proxy* neuron. With real unit activations the same case runs 14 s → non-termination
   (1 of 5 units), so the tractability gain was substantially the easy target, not the block
   structure. The block structure is real; its share of the effect was overstated. See
   [`diary/summer_d5.2.md`](diary/summer_d5.2.md).
4. **The wall is nonetheless real**, and it arrives on the K axis (`results/real_K{30,50}.csv`):

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
   than on synthetic data — quality plateaus early instead of wobbling:

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

  | neuron | frontier (med) | time (med) | best IoU (med) | timeouts |
  |---|---|---|---|---|
  | proxy | 3,085 | 6.5 s | 0.905 | 0/1 |
  | untrained | 18,760 | 145.6 s | 0.611 | 1/5 |
  | trained (density-matched) | 15,491 | 86.5 s | 0.530 | 0/5 |

  **Trained neurons land next to untrained, not next to the proxy** — the proxy is a 5×
  frontier / 13× time outlier against a properly trained model. That gap is an order of
  magnitude larger than the noise and is the robust finding here.
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
- 200 sentences (~2,547 tokens) were used to keep M near the synthetic M=2048.

```bash
python src/real_token_masks.py                      # overlap statistics, all arms
python src/real_token_search.py --arms lemma all \
    --lengths 4 --K 50 --beam_list none 200         # the wall and the fix
```

## Reproduce (current or a new namespace)

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
meta-path finder, so it runs on a bare CPU container without those installed.

## Cleanup / moving off the PVC

`kubectl delete pod optimalce-cpu` frees compute (no GPU is held by the CPU pod anyway).
Keep the PVC only for the Cityscapes data + activation cache; none of *this* repo depends on
it. To migrate: push this repo to your own GitHub, then `setup_pod.sh` against a pod in the
new namespace.
