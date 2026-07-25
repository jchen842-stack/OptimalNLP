# Research Diary — Summer D5: OptimalCE — Reproducing the NLP Failure & the Beam-Fallback Fix

*Continues D3 (reproduction + code mapping) and D4. D3 read the method top-to-bottom and
predicted, from code alone, where the vision→NLP transfer breaks: four "seams," rooted in
`mask_utils.py:160` (`common = sum_elements > 1`). This session **reproduces that predicted
failure empirically**, maps the failure curve, implements the paper's §3.3 escape hatch, and
validates it — all without touching real token data yet, and all GPU-free.*

## 0. Headline

- The optimal search is **100% CPU** work (Python `heapq` + estimation over tiny boolean
  tensors). The GPU only ever computed ResNet activations/masks, which we replace with
  synthetic masks — so every experiment here runs on a bare CPU pod, no GPU held.
- **Mechanism confirmed:** concept *overlap* (not the pixels) is what makes vision tractable.
  As overlap rises the disjoint fast-path vanishes, ceilings loosen, the frontier balloons.
- **The wall:** formula **length ≥ 4 × high overlap** → the frontier hits a 200k cap / times
  out (a real OOM-equivalent). The disjoint control at the same size finishes in ~1 s.
- **Fix (§3.3):** a size-bounded beam fallback. Refined finding — a size cap bounds *memory*,
  but the beam must be *tight* to also bound *time*: wide cap (2000) still times out; beam
  100–500 turns a non-terminating search into 1.4–30 s with a usable explanation.

## 1. Infra decision: CPU-only, and why

The frontier explosion is a property of the best-first *search* (`compositional/optimal.py`),
not of activation computation. Masks are just per-concept sparse boolean arrays over a flat
set of "elements," composed with `|`/`&`/`~` on the CPU. So we stood up a **CPU-only pod**
(`infra/pod-cpu.yaml`) mounting the same PVC and stubbed the vision-only imports
(`detectron2`, `cityscapesscripts`) with a meta-path finder. GPU would only add
host↔device overhead on tiny tensors — it would be *slower*, not faster. The scale-up wall we
chase is RAM (frontier size) and CPU time.

## 2. Method: isolate overlap as the single variable

To avoid confounding "does it break?" with "did I build the token adapter right?", we test
the *mechanism* on synthetic masks before any real NLP data. Construction
(`src/synthetic_overlap_sweep.py`):

- Build a **disjoint base** (vision-like): each active element belongs to exactly one concept.
- Introduce **overlap** by *adding* concept memberships to a fraction `p_add` of elements
  (a token carrying multiple concepts). Base memberships are preserved, so any signal defined
  on the base persists across overlap levels.
- The **neuron** is fixed as a 3-concept OR target over the base + 5% label noise, so
  `num_hits` is constant and — crucially — the disjoint level has an (almost) perfect short
  explanation → strong floor → fast, exactly like Cityscapes.
- Metrics: peak frontier size (via a `heapq` probe), visited/expanded/estimated nodes, best
  IoU, wall time. A frontier cap + per-level time budget prevent a real OOM while still
  showing the balloon.

## 3. Experiment 1 — overlap sweep (the mechanism)

`M=2048` elements, `K=15` concepts, `length=3`; only overlap moves.

| overlap `p_add` | mean overlap | common % | disjoint pairs | peak frontier | expanded | estimated | best IoU | time |
|---|---|---|---|---|---|---|---|---|
| 0.00 (vision) | 1.00 | 0% | **210** | 65 | 9 | 118 | **0.770** | 0.02 s |
| 0.25 | 1.46 | 25% | **0** | 201 | 108 | 1093 | 0.627 | 0.16 s |
| 0.50 | 1.94 | 50% | 0 | 498 | 402 | 4789 | 0.529 | 0.67 s |
| 0.75 | 2.39 | 75% | 0 | 1075 | 468 | 5009 | 0.456 | 1.57 s |
| 1.00 (token) | 2.86 | 100% | 0 | 724 | 504 | 5477 | 0.419 | 1.71 s |

Every D3 prediction holds: disjoint = cheap & exact (2 nodes visited, 0.02 s); the fast-path
precondition (`disjoint_pairs`) collapses **210 → 0** the instant overlap appears — it is
removed, not merely slowed; and both compounding penalties are visible — estimated nodes
**46×**, time **85×**, *and* IoU degrades 0.77 → 0.42 (loose ceilings stop separating).

## 4. Experiment 2 — scale curve (the wall)

`M=2048`, frontier cap 200k, 30 s/level budget; disjoint control vs token regime at each size.

| K | length | regime | peak frontier | estimated | best IoU | time | outcome |
|---|---|---|---|---|---|---|---|
| 15 | 3 | disjoint | 65 | 118 | 0.770 | 0.02 s | ✓ |
| 15 | 3 | token | 724 | 5,477 | 0.419 | 1.7 s | ✓ |
| 15 | **4** | disjoint | 103 | 244 | 0.770 | 0.04 s | ✓ |
| 15 | **4** | token | 22,458 | — | — | 33 s | ⏱ timeout |
| 30 | 3 | token | 4,158 | 42,676 | 0.358 | 6.9 s | ✓ |
| 30 | **4** | disjoint | 227 | 4,513 | 0.639 | 0.56 s | ✓ |
| 30 | **4** | token | 167,497 | — | — | 30 s | ⏱ timeout |
| 50 | 3 | token | 10,014 | 193,094 | 0.332 | 27 s | ✓ (barely) |
| 50 | **4** | disjoint | 278 | 9,383 | 0.524 | 1.03 s | ✓ |
| 50 | **4** | token | **200,001** | — | — | 27 s | 💥 hit frontier CAP |

**The disjoint control never breaks — at any size** (always 2 nodes visited, ≤1 s). The wall
is `length ≥ 4 × overlap`: length is the accelerant (branching), overlap is the root cause
(no pruning). At K=50/length=4 the token frontier is **~720×** the disjoint control and does
not finish — the OOM the diary predicted, on demand.

## 5. The fix — size-bounded beam fallback (§3.3)

`patches/0001-frontier-beam-fallback.patch` (vs upstream `70805299`). Adds a module global
`MAX_FRONTIER_SIZE` and `_apply_beam_cap()` to `compositional/optimal.py`, enforced at both
frontier chokepoints: `reduce_frontier` (the diary's named intervention, L412) and
`update_frontier` (where the frontier actually grows each expansion). When the frontier
exceeds the cap, keep only the top-N nodes by ceiling and re-heapify.

Default `MAX_FRONTIER_SIZE = None` is a **no-op** → exact optimal search is unchanged, so the
patch is non-invasive for existing vision runs.

## 6. Experiment 3 — beam-width sweep (validation)

Hardest terminating-boundary case: `K=15, length=4, full token overlap`, 120 s budget.

| beam width | frontier | terminates? | visited | best IoU | time |
|---|---|---|---|---|---|
| none (exact) | 22,458 | ✗ | — | — | 131 s timeout |
| 100 | 155 | ✓ | 96 | 0.377 | **1.36 s** |
| 200 | 254 | ✓ | 156 | 0.368 | 3.2 s |
| **500** | 554 | ✓ | 2035 | **0.424** | 30 s |
| 1000 | 1054 | ✓ | 3797 | 0.420 | 99 s |
| 2000 | 2054 | ✗ | — | — | 122 s timeout |

**Refined §3.3 finding.** Applying a *wide* cap (2000) across the whole scale grid bounded the
frontier everywhere (`results/scale_beam.csv`, peak ≈ 2000 vs 200k) — i.e. it fixes **memory**
— yet the length-4 token cases still timed out. The beam-width sweep shows why: the per-node
sample estimation is expensive, so a wide beam drains too slowly. A **tight** beam (100–500)
bounds both memory and time, converting a 131 s non-terminating search into **1.36 s** with a
real explanation. IoU rises with width then plateaus (~0.42 by 500) — the classic beam
width ↔ quality ↔ time tradeoff. Sweet spot here ≈ 500. So the intervention is not "add a size
cap" but "add a beam whose width is tuned to the compute budget."

## 7. Vision → NLP namespace map (for the transition)

The generic term **"element"** is the namespace-neutral pivot already used in the harness.
When we move to real tokens, the rename is mechanical:

| Vision (today) | Code locus | NLP / token analogue |
|---|---|---|
| pixel / spatial location | element of a mask over `H*W` | token position in the corpus |
| image | bitmap row (`N`) | sentence / sequence |
| `mask_shape = (H, W)` | `config.get_mask_shape()` | `(n_sequences, max_len)` |
| concept segmentation `.npz` | `masks[c]`, sparse `(N, H*W)` | token-level concept annotation (POS, word-sense, feature) |
| neuron activation bitmap | `bitmaps` | unit activation over token positions |
| ~disjoint concepts (Cityscapes 19) | `disjoint_info` | rarely disjoint — tokens carry many concepts |
| `sum_elements == 1` dominates | `mask_utils.py:160` | `sum_elements > 1` (common) dominates |

Identifiers to rename at transition time: `pixel→token`, `image→sequence`,
`mask_shape→sequence_shape`, `segmentation→annotation`. The core patch is domain-neutral and
needs no change.

## 8. Next steps

- **(A) Real NLP token data** as the actual deliverable: pick a token-level concept-annotated
  dataset + a model whose units we explain, encode token masks in the `(N, elements)` format,
  and confirm the wall + fix on genuine data.
- **(B) Sharpen the fix:** auto-select beam width from a time budget, or add a total-expansion
  budget so termination is guaranteed regardless of width. Optionally compare the beam
  fallback's explanations to the exact optimum on cases where exact still finishes, to quantify
  the optimality gap.

## 9. Artifacts

- Harness: `src/synthetic_overlap_sweep.py` (modes `overlap` / `scale` / `beam`)
- Fix: `patches/0001-frontier-beam-fallback.patch`
- Results: `results/{overlap_sweep,scale_curve,scale_beam,beam_sweep}.csv`
- Infra: `infra/{pvc,pod,pod-cpu}.yaml`; setup: `scripts/setup_pod.sh`
- Upstream pin: `70805299` (`UPSTREAM`)
