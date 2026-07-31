# Research Diary — Summer D5.3: OptimalCE — Trained Neurons Close the Bracket; the Mechanism Stays Open

*Revises D5.2. D5.2 replaced the proxy neuron with real **untrained** unit activations and
argued this was the pessimistic bound: random features have no reason to align with linguistic
concepts, so a trained neuron should be more explainable, prune harder, and search faster. A
Bowman SNLI classifier trained to **79.3 %** dev accuracy answers the first half: trained units
land next to the untrained ones, not next to the proxy, so the bracket closes at the pessimistic
end and D5.2's correction to D5.1 strengthens. The second half stays open. I drafted this entry
claiming the mechanism was refuted; testing that claim properly reversed its sign, and the
honest verdict at n=5 is that the direction is not established. §4 records how that nearly went
out.*

> ### ⚠️ Corrections from later entries (added D5.5, in place — nothing deleted)
>
> - **The IoU comparisons in §3–§5 are WITHDRAWN (D5.4).** Both arms were measured at
>   alpha=0 (density ~0.50), where IoU ≈ density by construction. The trained-vs-untrained IoU
>   gap tracks the density gap. §4 already flagged the difference as unestablished at n=5; the
>   reason is worse than sample size.
> - **§5's conclusion that the concept VOCABULARY is the binding constraint is unsupported.**
>   The binding constraint was the activation range. At a corrected range the trained arm
>   reaches lift 3.41 and **separates from untrained** (3.41 vs 2.07 at alpha=0.05) where at
>   alpha=0.5 the two are indistinguishable — so the D5.3 question is reopened, with an
>   instrument that works (D5.5 §2, §5).
> - **The model itself is verified.** `bowman_snli_best.pth` re-evaluates to
>   0.7934362934362934 dev accuracy exactly. But the recorded invocation was wrong: it needs
>   `--max_data 0`, not the defaults (`VERIFICATION.md` check 7, `models/README.md`).
> - **The density-band lesson in §4 generalised** into a recurring error shape, documented in
>   `results/METHOD_NOTES.md`: an effect attributed to the swept variable while the unit set
>   moves underneath it.

## 0. Headline

- **The bracket collapses toward the pessimistic end.** D5.2 left the answer somewhere between
  proxy (6.5 s) and untrained (145.6 s median, 1/5 timeouts). Trained units, density-matched:
  **86.5 s median, frontier 15 491** — near untrained, nowhere near the proxy.
- **My predicted mechanism is unresolved, and my first reading of it was wrong.** I argued
  trained units would be *more* explainable → stronger floor → faster. The n=5 search sample
  appeared to show the opposite (IoU 0.530 vs 0.611) — but that comparison is censored and
  underpowered, and a **density-matched test over all 600 eligible units reverses the sign**
  (trained 0.431 vs untrained 0.420, p=0.008, d=0.20). The direction of the explainability
  difference is **not established**; the effect, if real, is small either way. See §4.
- **A confound I introduced, then removed.** The first trained run sampled units across a much
  wider density range than the untrained run had, and reported a 45 % frontier reduction.
  Density-matched, that shrinks to **17 %**. The unmatched number overstated the effect.
- **D5.2's core correction stands and strengthens.** The proxy remains a 5× frontier and 13×
  time outlier against *properly trained* neurons.
- **Substantive interpretability result (the robust one):** neurons in a 79.3 %-accuracy SNLI
  model reach only IoU ≈ 0.5 under length-4 formulas over POS/dep/lemma/synset/constituent
  concepts — and untrained units score no better. Real neurons of either kind are *not* well
  described by this vocabulary. This rests on the absolute IoU level, not on any
  trained-vs-untrained difference.

## 1. The model

`src/train_snli_encoder.py` — a driver over their `SNLI`/`models`/`pad_collate`, written
because `snli_train.py` offers only `--debug` (1 000 pairs) or the full corpus for 50 epochs
with no subset knob. Full SNLI, 549 367 train pairs, vocab 33 671, 3 epochs, batch 100, Adam.

| epoch | train loss | train acc | val acc |
|---|---|---|---|
| 0 | 0.699 | 0.701 | 0.772 |
| 1 | 0.537 | 0.783 | 0.792 |
| 2 | 0.461 | 0.817 | **0.793** |

~13 min/epoch on CPU. **0.7934** dev accuracy is above the task7 Bowman reference (0.775) and
in the expected band for this architecture, so this is a legitimately trained model, not a
token gesture.

A vocabulary trap worth recording: the embedding table is indexed by the *training* vocab, so
`real_activations.py --ckpt` must take `stoi` from the checkpoint. Rebuilding it from the
annotation corpus would map every token to the wrong embedding row and still emit
plausible-looking activations — a wrong answer that looks right. OOV against the real vocab is
**8.8 %** (vs 16.5 % against the 2 000-pair smoke-test vocab), which is the canary that the
right vocabulary is in use.

## 2. What training does to the units

Training makes units **selective**, which is visible before any search runs:

| | density mean | density **std** | in band [0.15, 0.85] | near-dead (<0.02) | near-saturated (>0.98) |
|---|---|---|---|---|---|
| untrained | 0.504 | 0.099 | 512/512 (100 %) | 0 | 0 |
| trained | 0.496 | **0.229** | 446/512 (87 %) | **4** | **8** |

Mean density barely moves; the spread more than doubles. Untrained units all sit in a narrow
mid-density band — random projections of embeddings fire on roughly half of everything.
Training pushes units apart: some die, some saturate, some specialise.

Qualitatively, the specialised ones are recognisably linguistic. Trained `unit395`
(density 0.171) aligns best with **`tag=DT`** — determiner-selective. By contrast almost every
*untrained* unit's nearest concept is `const=NP`, which is mostly a density artifact: NP is the
most frequent concept, so it scores well against any dense mask.

## 3. Experiment — proxy vs untrained vs trained (`results/trained_units_K15*.csv`)

`K=15, length=4, all categories`, 300 s budget. **Only the neuron changes.**

| arm | n | density range | frontier (med) | time (med) | best IoU (med) | timeouts |
|---|---|---|---|---|---|---|
| proxy (D5.1) | 1 | 0.43 | **3 085** | **6.5 s** | **0.905** | 0/1 |
| untrained (D5.2) | 5 | 0.40–0.64 | 18 760 | 145.6 s | 0.611 | **1/5** |
| trained, unmatched | 5 | 0.17–0.75 | 10 290 | 89.4 s | 0.445 | 0/5 |
| **trained, density-matched** | 5 | 0.45–0.56 | **15 491** | **86.5 s** | **0.530** | 0/5 |

Per-unit, density-matched:

| unit | density | frontier | IoU | time |
|---|---|---|---|---|
| 109 | 0.536 | 17 724 | 0.530 | 86.6 s |
| 115 | 0.529 | 14 881 | 0.533 | 50.0 s |
| 369 | 0.565 | 15 491 | 0.591 | 56.3 s |
| 389 | 0.494 | 19 547 | 0.500 | 159.8 s |
| 506 | 0.453 | 11 614 | 0.464 | 89.8 s |

### 3.1 The confound I introduced, and removing it

The first trained run sampled from the full `[0.15, 0.85]` band. Because training widens the
density distribution, that pulled in units at 0.17 and 0.75 — a *different* density
distribution from the untrained run's 0.40–0.64. It reported frontier 10 290, a 45 % reduction.
Re-running restricted to 0.39–0.65 gives **15 491, a 17 % reduction**. Most of the apparent
advantage was density, not training.

This is precisely the error D5.2 §4 criticised D5.1 for — changing two things and reading the
result as though one had changed. It took one run to reproduce it. The density band is now a
CLI argument (`--dmin/--dmax`) so the matching is explicit rather than accidental.

## 4. The mechanism: unresolved, and a sign error I nearly published

D5.2 §5 argued: trained units are more explainable → higher best-so-far IoU → stronger pruning
threshold → smaller frontier. The prediction was **more explainable and faster**.

My first reading of this experiment was that the prediction had been *refuted* — that trained
units were **less** explainable yet still faster, which would have made the proposed mechanism
untenable. Two measurements looked like they agreed:

| measurement | untrained | trained | verdict as first read |
|---|---|---|---|
| best IoU under search (n=4 vs 5) | 0.608 | 0.524 | trained worse |
| best single-concept IoU, 40-unit sample | 0.418 | 0.400 | trained worse |

**Both were flawed, and testing them properly reverses the conclusion.**

*The search comparison is censored.* The untrained arm has one unit that timed out, so its IoU
is unknown and was dropped. That unit is by construction the **hardest** one, so excluding it
biases the untrained mean *upward* — exactly in the direction that produced the apparent gap.
With n=4 vs 5 and that bias, `p = 0.032` is not worth much.

*The 40-unit check was not density-matched.* It sampled the full `[0.15, 0.85]` band. Because
training widens the density distribution (§2), that band pulls extreme-density trained units
into the trained arm but has no equivalent effect on the untrained arm — the same confound as
§3.1, reintroduced in the very check meant to corroborate the result.

Done properly — density-matched to `[0.39, 0.65]`, all eligible units, no sampling, no censoring:

| | n | mean best single-concept IoU | sd | median |
|---|---|---|---|---|
| untrained | 406 | 0.4199 | 0.062 | 0.414 |
| **trained** | 194 | **0.4309** | 0.049 | 0.429 |

Mann–Whitney `p = 0.0085`, Welch `p = 0.018`, **Cohen's d = 0.20**. The sign is **opposite** to
my first reading: trained units are, if anything, *marginally more* alignable — by 0.011 IoU,
a small effect on a large sample.

**So the honest state is:**

- The **direction** of the explainability difference between trained and untrained units is
  **not established**. The best-powered, unbiased measurement favours trained being slightly
  more alignable; the underpowered censored one says the reverse. The effect is small either way.
- The **speed difference** (86.5 s vs 145.6 s median) therefore has no established explanation.
  It is not licensed to attribute it to explainability in either direction at this n.
- D5.2's mechanism is neither confirmed nor refuted by this experiment. Settling it needs (A4).

*What nearly went wrong:* the headline "my prediction is refuted" is a more interesting claim
than "the effect is small and I can't sign it," and both flawed measurements pointed at it. The
corroborating check was chosen after the fact and inherited the same confound as the primary
result. Neither was tested until the write-up was already drafted.

A separate reading survives and is worth stating: both arms sit near IoU 0.4–0.6, so **real
neurons of either kind are poorly described by this concept vocabulary**. If that is the binding
constraint, the fix is a richer vocabulary rather than a better search — a result about the
*method*, not our implementation of it. That claim rests on the absolute IoU level, which is
robust, not on the trained-vs-untrained difference, which is not.

## 5. What this means for the project's claim

The three entries now read:

| entry | masks | neuron | K=15 / length 4 |
|---|---|---|---|
| D5 | synthetic | synthetic OR-target | timeout (131 s) |
| D5.1 | real | proxy OR-target | 6.5 s ✓ |
| D5.2 | real | real **untrained** unit | 145.6 s med, 1/5 timeout |
| D5.3 | real | real **trained** unit (79.3 % acc) | 86.5 s med, 0/5 timeout |

D5.1's "real data is more tractable than D5 predicted" was substantially an artifact of the
proxy target. With genuine trained neurons the case is **13× slower and 5× larger** than the
proxy suggested. The block-structure finding (D5.1 §5) remains true and remains the reason
real data is not *as* catastrophic as the uniform-overlap synthetic model — but it was never
the whole story.

## 6. Scope

- **n = 5 units per arm.** Enough for order of magnitude, not for direction. The per-unit
  spread (50 s–160 s within the matched arm) is large relative to the between-arm difference,
  so **neither** the 17 % frontier gap **nor** the search-IoU gap is established; §4 shows the
  latter reverses sign under a properly matched, uncensored test. What *is* robust is the gap to
  the proxy (5× frontier, 13× time), which is an order of magnitude larger than the noise.
- **One censored observation.** The untrained arm's timeout has unknown IoU and is excluded from
  IoU means, biasing them upward. Times are reported as medians, which tolerate the censoring;
  IoU means do not.
- 3 epochs, not to convergence — 0.793 is a good model but not a maximally trained one.
- 200 sentences / 2 547 tokens; K=15; single mask seed.
- The concept vocabulary is fixed at the 15 most frequent concepts, which is dominated by
  frequent structural labels (`NP`, `VP`, `PP`, `NN`, `DT`). A different vocabulary could
  change the explainability results substantially — see §4.

## 7. Next steps

- **(A4) More units, properly.** ~50 units per arm with matched densities, reporting
  distributions rather than medians. This is what would turn §3's 17 % into a real claim or
  retire it. Cheap to run, just slow.
- **(C) Richer concept vocabulary.** §4 suggests trained neurons encode distinctions absent
  from POS/dep/synset annotation. Larger K, or concepts beyond this annotation set, would test
  whether low IoU is a property of the neurons or of the vocabulary.
- **(B) Beam sweep with trained units.** The untrained beam sweep validated width 200
  (`results/real_beam_units_K50.csv`); the trained equivalent is the last piece of the
  practical recommendation.
- **(D) Explain the speed difference** in §4, which is currently an observation without a cause
  — and which (A4) may yet show is not a real effect.
- **(E) Pre-register the comparison.** §4's near-miss came from choosing the corroborating
  measurement after seeing the primary result. Fix the units, densities, and tests before the
  next run rather than after.

## 8. Artifacts

- Trainer: `src/train_snli_encoder.py` → `models/bowman_snli_best.pth` (0.7934 dev; gitignored)
- Trained activations: `src/real_activations.py --ckpt` → `results/real_activations_trained.npz`
- Search: `results/trained_units_K15.csv` (unmatched), `results/trained_units_K15_matched.csv`
- Density band now explicit: `real_token_search.py --dmin/--dmax`
- Unchanged: masks, statistics, the beam patch
