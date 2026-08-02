# Method notes for the alpha sweep (M = 24,199 tokens, K = 15, length 4)

## The recurring error shape: comparing across a condition while the unit set moves

2026-07-30. The exact-search probes at M=24,199 both timed out (alpha=0.05 at 301s with
peak frontier 26,442; alpha=0.005 at 319s with peak frontier 14,720), and were read as
"exact search is intractable at this corpus size". Phase B's pilots then terminated
comfortably on the same corpus and the same alpha — 133.5s for trained unit 86 at
alpha=0.05, 23.5s for trained unit 413 at alpha=0.2.

The probes and the pilots ran **different units**. The probes predate the `min_fire`
floor and drew unit 416 by seeded random selection; Phase B targets the units Phase A
qualified. Termination is unit-dependent, so the probe measured one unit's difficulty and
it was read as a property of the corpus.

This is the fifth instance of the same error shape in this project, and the first caught
within hours rather than across diary entries. The shape: **an effect is attributed to the
swept variable while the unit set silently changes underneath it.** Previous instances
include the D5.3 density-band mismatch, where the trained arm sampled the full [0.15, 0.85]
band and the untrained arm did not, and the trained/untrained IoU gap that produced was
read as a training effect.

Two structural causes, both live in this codebase:

- `load_real_neurons` selects by seeded `rng.choice` over an *eligible* set. Any filter
  that changes the size of that set — a density band, the `min_fire` floor — changes which
  units are drawn, silently. At alpha=0.05 the `min_fire` floor excludes 7 of 512 trained
  units, and that alone makes the trained alpha=0.05 unit set {86, 87, 91, 395, 412}
  disjoint from the {88, 92, 396, 413, 510} drawn at every other alpha.
- Single-unit probes are cheap and therefore tempting, and a single unit has no error bar.

Mitigations now in place: `--unit_ids` pins the unit set explicitly for cross-condition
comparisons, and Q3 is scored only on within-unit paired series.

## Two more measurement traps found in the same pass

**Density must be per run, never per arm.** tanh saturation puts ties at the quantile
threshold, so a unit's realised density drifts below its nominal alpha — unit 86 sits at
0.0365 when alpha=0.05, and two trained units drift further (unit 413 at 0.026 when
alpha=0.1, unit 88 at 0.129 when alpha=0.2). Lift divides by density, so an arm mean
produces lifts that cannot be reconciled against their own IoU and coverage.

**Peak frontier is not a search-cost measure.** It counts nodes held at once; `visited`
counts nodes actually explored. In the Phase B pilots they point opposite ways — peak
frontier 8,239 (alpha=0.2) vs 6,847 (alpha=0.05) suggests low alpha prunes better, while
visited 2,063 vs 7,805 says the opposite — and wall time tracks visited. Peak frontier has
now misled twice: once here, once in the timeout probes above, where it is additionally
not a converged statistic because the run was cut off mid-exploration.

## Diagnostics that inherit the confound they were built to detect

The consistency check on Phase B's pilot flagged unit 86 as impossible: exact IoU 0.220 at
coverage 0.060 appeared to give lift 6.84, below beam's 7.93, which a lower bound cannot
be. The check itself had substituted the arm-mean density 0.047. With the unit's realised
density 0.0365 the exact lift is 7.946, above beam and consistent. **The diagnostic built
to catch arm-mean substitution performed an arm-mean substitution.**

Third instance of this shape in the project:

1. D5.3's 40-unit density check — run to validate the trained/untrained comparison, and
   itself not density-matched, so it reproduced the very mismatch it was checking.
2. D5.4's IoU-without-null — IoU used to assess explanation quality, with no null model, so
   it could not detect that IoU was tracking density rather than fit.
3. This one.

The general lesson: a diagnostic written in the same session, by the same hand, against the
same mental model as the thing it checks will tend to share its assumptions. A check is only
independent if it is derived from a different quantity — here, per-run `n_inter` and
`n_fires` rather than any aggregate.

## Pre-registered prediction: optimality gap vs lift

Registered 2026-07-30, BEFORE the 27-pair Phase B batch completed, so that it is scored
rather than fitted.

> **The optimality gap is inversely related to lift.** High-lift units give a strong
> incumbent early, so beam and exact converge; low-lift units leave the pruning threshold
> weak and beam drifts further from the optimum.

Evidence at registration time, n=2: unit 86, lift 7.93, gap +0.2%; unit 413, lift 3.14,
gap +5.1%.

To be scored as a rank correlation (Spearman) of `rel_gain_pct` against `exact_lift` over
all 27 pairs, reported with a scatter. If it holds it is a quantitative form of the paper's
§4.3 advice to run beam first and refine only the units deemed interpretable: the units
that most need exact search are the ones beam is least able to serve. If it fails, that is
reported as a failure.

### Amendment before the data existed: the pooled statistic was confounded

Both predictions were first registered to be scored by a **pooled** Spearman over all 27
pairs. That is invalid here. Alpha drives lift and density together — from Phase A,
`rho(density, lift)` is **-1.00** (trained) and **-0.80** (untrained) — so a pooled
`rho(time, lift)` is `rho(time, density)` with the sign flipped and cannot distinguish the
two. Prediction #1 had the identical problem.

The registered mechanism is that a strong incumbent prunes harder. That is a claim about
differences **between units at fixed density**, not about the alpha axis, so the alpha axis
has to be removed by construction rather than adjusted for afterwards.

**Amended scoring, replacing pooled Spearman:**

1. Rank `rel_gain_pct` (and `time_s`) against `exact_lift` **within** each (arm, alpha)
   stratum — 6 strata.
2. Pool the within-stratum ranks and compute Spearman over all 27. Removes the density
   effect by construction, and retains the power that six separate n=5 tests would lose.
   Ranks are centred on (n+1)/2 and scaled by n before pooling, because the strata have
   unequal size (3, 4, 5) and a raw rank of 3 means "middle" in one and "top" in another.
3. Per-stratum rho is reported too, so a single stratum driving the result is visible.
4. SUPPORTED / NOT SUPPORTED is scored against the stratified statistic at p < 0.05.

The pooled figure is printed only when explicitly labelled CONFOUNDED, never as headline.

**p comes from a stratified permutation test, not Spearman's asymptotic p.** The centring
leaves unequal spread across unequal strata — `(rank-(n+1)/2)/n` spans ±0.400 at n=5,
±0.375 at n=4, ±0.333 at n=3, so the n=3 stratum is down-weighted about 17%. Permuting
`exact_lift` within each stratum 10,000 times and recomputing the pooled figure reproduces
that weighting exactly under the null rather than assuming it away, and makes no tie or
normality assumptions. Spearman's asymptotic p is not meaningful at 3–5 per cell in any
case. Validated both ways on synthetic data: a true inverse relation gives rho −0.751 at
permutation p = 0.0002 (SUPPORTED), a pure null gives rho −0.022 at p = 0.464
(NOT SUPPORTED). The null arm matters — a test that only fires on the signal case has not
been validated.

Only the permutation p is printed. An earlier draft of this note claimed the permutation
and asymptotic p "disagree at small n", citing 0.464 against 0.914 on the null arm. **That
was false.** The permutation p is one-sided and Spearman's asymptotic p is two-sided; at
n=27 with rho=−0.022, t = −0.022·√(25/(1−0.022²)) = −0.110 on df=25, giving one-sided
0.4566 and two-sided 0.9133 — which matches the 0.9136 that was printed. Compared
like-for-like they agree to three decimals and there is no small-n discrepancy. The
asymptotic p has been dropped from the output rather than made one-sided, because a
p-value that reads 2x high is exactly the kind of number that gets quoted loose from a log,
and the permutation p is the one being scored on.

Recorded because the false version was plausible-sounding, arrived with a supporting table,
and would have been repeated unchallenged. The reason to permute here is the unequal
stratum weighting, not small-sample failure of the asymptotic p.

The pilots score nothing either way: unit 86 (lift 7.93, 133.5s) and unit 413 (lift 3.14,
23.5s) sit at different alphas, so they are confounded identically.

**This is the same shape one level up.** The "knob vs quantity" note was written, and then
a prediction was registered that conflates the knob's *two downstream quantities* with each
other. Recognising that alpha is not the right axis did not prevent registering a statistic
that put alpha back on the axis implicitly, through a variable it controls.

It is also the strongest argument for pre-registration in this project so far: the flaw
surfaced **before the data existed**, so it was fixed rather than rationalised. Had the 27
pairs landed first, a pooled rho of the predicted sign would have been available to report,
and the confound would have had to be argued away against a result that already looked
right.

**Companion prediction, registered at the same time:** `rho(exact time_s, exact_lift) < 0`.
Same scatter, same script verdict. The two predictions share one mechanism — a strong
incumbent found early prunes harder — so if BOTH the optimality gap and the search time
fall as lift rises, the mechanism is triangulated rather than assumed from the gap alone.

It also makes an earlier miss testable. The Phase B pilots were selected for high lift
(unit 86 at 7.93, unit 413 at 3.14) and both terminated fast, which was read as evidence
that exact search is tractable at M=24,199 in general. Under this prediction they were the
fastest cases *by construction*, and the first batch running long is what that predicts.
Scoring the correlation scores that reading too.

### Second instance, in prose, hours after the code fix

Status note, same day, on Phase B's first batch running long: *"consistent with prediction
#2 given unit 88's realised density of 0.129."*

That inference runs through **density**, which is the confounded route removed from the
scoring a few hours earlier. It is also backwards on its own terms: under #2 a lower
density implies higher lift, and higher lift predicts **faster**, so a long runtime is weak
evidence *against* #2 if it is evidence of anything. And a single unit cannot score a
within-stratum rank correlation in either direction.

So the failure documented and fixed in code — "recognising alpha was the wrong axis did not
prevent putting alpha back on the axis implicitly" — recurred in narration within hours of
the fix, with the sign wrong as well.

This is the argument for the verdict being **emitted by the script rather than narrated**.
The stratification, the permutation p and the SUPPORTED / NOT SUPPORTED string are all
computed from the data by `score_prediction`; prose alongside them is where the confound
re-entered, because prose has no stratification and no sign check. Interim commentary on a
running batch is the highest-risk surface for this and should say what is running and
nothing about what it implies.

## Pre-committed interpretation of the four outcomes

Registered 2026-07-30, before the 27-pair batch completed, so the reading is fixed in
advance rather than chosen once the signs are known.

| | #2 supported (time falls with lift) | #2 not supported |
|---|---|---|
| **#1 supported** (gap falls with lift) | Incumbent strength explains both. Mechanism triangulated. Quantitative version of the paper's §4.3 "beam first, refine the interpretable units". **See the warning below.** | Incumbent affects WHAT you find, not HOW LONG it takes. Runtime is driven by something else — per-unit mask structure, coverage granularity. Mechanism splits. |
| **#1 not supported** | Beam quality is independent of incumbent strength. The recommendation collapses to "just use beam". | Incumbent strength is not the mechanism. Pushes toward bound-tightening as the only remaining lever. |

### Fifth outcome, registered separately: INVERTED #1 (gap RISES with lift)

The four-outcome table above registered what SUPPORTED and NOT SUPPORTED mean. It did not
register **inverted** — a significantly *positive* rho — and batch 1 points that way
(rho = +1.00 on the IoU gap, though at n=3 in one stratum the permutation p floor is 1/6,
so it is weak and scores nothing). Registered here before the remaining batches land.

**If the gap rises with lift, the registered mechanism is wrong.** The alternative: a
high-lift unit is one where a genuinely good formula *exists*. Exact search reaches it;
beam, being width-limited, misses it; and missing a good optimum costs more in relative
terms than missing a mediocre one. Under that reading the gap measures **available
headroom**, not pruning quality.

This coexists cleanly with #2 supported, because the two would then track different things:
time tracks pruning, gap tracks headroom. So "inverted #1 + supported #2" is a coherent
joint outcome and not a contradiction.

**Consequence, recorded before the result because it is favourable:** inverted #1 *inverts*
the awkward conclusion registered below. Under it, exact search is most valuable where
explanations are **good**, not where they are marginal — which argues *for* the project
rather than against it. A result favourable to the work is exactly the one most at risk of
being adopted without the same scrutiny applied to an unfavourable one, so it goes on record
in advance on the same terms.

### Recorded in advance because it is the outcome most likely to be talked around

**If BOTH predictions are supported, that argues against this project.** Both supported
means beam converges to the optimum exactly where explanations are good, and diverges only
where they are marginal. Exact search would then be most valuable precisely where the
answer matters least — which is the regime the paper itself says to skip. It is written
down here, before the data exists, so that if it lands it gets reported as-is and not
reframed as a validation of the beam-then-refine workflow.

## Optimality gap MAGNITUDE — a separate headline

Predictions #1 and #2 are both about the **order** of gaps across units. Neither says
anything about their **size**, and the size is a result in its own right: it is the first
NLP measurement of a quantity the paper reports only for vision.

Reported as mean / median / range of the gap against the paper's Table 4 vision band of
**+5.1% to +6.5%**. Registration-time evidence, n=2, spans +0.2% (unit 86) to +5.1%
(unit 413).

**If the mean comes out under ~1%, that is a bigger finding than either prediction:** exact
search buys almost nothing in NLP. The script prints an explicit verdict against the 1%
line and against the vision band, so this is emitted rather than narrated.

**Report median and the per-unit distribution, not just the mean.** Batch 1's three
IoU-gains are +2.56, +2.84, +17.25: mean 7.55%, median 2.84%. The mean sits *above* the
vision band and the median *below* it, and the entire difference is one unit (92). At n=3
the mean is not a summary of anything. The script prints mean, median, range and every
per-unit value, and flags when mean and median diverge by more than a point.

## Write-up flag: does exact systematically widen coverage?

Exact widening coverage relative to beam is the alpha=0.5 blanket problem in miniature,
appearing at alpha=0.2 — and exact is **more** exposed to it than beam, precisely because
it reaches the IoU optimum rather than stopping short of it. If `cov_ratio_exact_over_beam`
is systematically > 1 across all 27 pairs, that is a finding about **the objective**, not
about our units: IoU rewards coverage in a way that works against the interpretability the
search is meant to deliver.

Not to be inferred from unit 88 alone. Batch 1 already shows it is not uniform — unit 88 is
1.65, unit 413 is 1.21, unit 92 is 0.71, so exact *narrowed* coverage in one of three. The
ratio is tracked as a column and summarised over all 27 with a count above 1.0.

### Correction: "beam lift lower-bounds the optimum's lift" is false

The guarantee the search provides is **`exact_IoU >= beam_IoU`, and nothing more.** The
search maximises IoU; lift is a derived quantity that divides by coverage. Exact can and
does buy IoU by widening coverage, which trades precision for recall and makes lift FALL
while IoU rises.

Observed in Phase B batch 1, trained unit 88 at alpha=0.2: IoU 0.254 -> 0.261 (+2.84%,
as guaranteed) while lift 3.067 -> 2.535 (-17.33%). Coverage ratio exact/beam = 1.65,
precision 0.394 -> 0.326, recall 0.416 -> 0.567. The precision/recall trade is the finding;
lift compresses it into one number and hides it, which is why precision and recall are now
reported for both searches alongside lift.

This invalidated an earlier claim made in reporting Phase A ("beam IoU is a valid lower
bound on the optimum's lift"). It also means the unit-86 reconciliation that appeared to
confirm the bound (exact 7.946 vs beam 7.93) was the ordering happening to survive, not the
guarantee holding.

### Prediction #1 is scored twice

- **primary:** `rel_gain_iou_pct` — non-negative by construction, is the actual optimality
  gap, and is what the paper reports.
- **secondary:** `rel_gain_lift_pct` — the literal registered quantity.

IoU-gain is a **correction to how the registered concept was operationalised, not a change
of hypothesis.** The registered concept was "the optimality gap"; lift-gain turned out not
to be an optimality gap, since it can be negative. But that intent **cannot be claimed
retrospectively** — the prediction was written loosely, and the choice of IoU-gain was made
after seeing three data points showing lift-gain go negative. That is precisely the move
pre-registration exists to prevent. Both are therefore reported, and neither is substituted
for the other.

If the two verdicts disagree, the disagreement is a result: it means beam and exact differ
in the **kind** of formula found, not merely its quality.

**Two gap definitions, and they are not interchangeable.** The paper's Table 4 band is an
**IoU** improvement, whereas `rel_gain_pct` — the quantity predictions #1 and #2 are scored
on — is a **lift** improvement. Lift divides by density, so the two diverge whenever beam
and exact select formulas of different coverage. `rel_gain_iou_pct` is therefore reported
alongside as the like-for-like number, and it is the one compared to +5.1%..+6.5%. Scoring
the predictions against the lift-based gain while comparing magnitude against the IoU-based
gain is deliberate, not an inconsistency.

### Third prose-layer instance: per-unit values invented from summary statistics

Reporting Phase B batch 1, a per-unit IoU-gain column was presented as data. The script had
printed no such column at that point — only the lift-based gain per unit, and a summary
line reading `mean +7.55% median +2.84% range +2.56% .. +17.25%`. The per-unit values were
produced by distributing the range endpoints and the median across the three rows, and two
of the three were wrong (unit 413 was reported as +2.56% when it is +17.25%).

Caught because the correct column, added an hour later, disagreed with the earlier prose.
The join was never wrong; nothing was joined. The numbers were invented from summaries.

The shape: **summary statistics were on screen, per-unit values were not, and per-unit
values were written as if they were.** A range and a median over n=3 look like they contain
the per-unit data, and with three rows and three numbers the reconstruction feels like
arithmetic rather than guessing.

This is the third failure at the prose layer while the code was correct, after the
density-routed inference about batch 1 runtime and the false permutation/asymptotic p
claim. All three share a structure: the script emitted a correct, narrow result, and the
narration around it added an unearned quantity. It reinforces the rule already recorded —
verdicts and per-unit values are emitted by the script or not stated at all.

### The pattern is shared, and it concentrates in corrections

Not one participant's failure mode. The fabricated per-unit table was propagated onward as
"the mean is driven entirely by unit 92" and repeated as fact by the reader as well as the
writer. The guard was available and unused on both sides: the table carried `beam_IoU` and
`exact_IoU` next to the gain column, so recomputing `exact/beam - 1` would have caught it on
sight. That check was run one message later instead.

**The errors concentrate in corrections specifically.** Reviewing the instances above:

- The unit-86 consistency check — written to catch arm-mean substitution — performed an
  arm-mean substitution.
- Prediction #2 was registered in the very message that proposed the knob-vs-quantity rule,
  and was itself confounded by the knob.
- The density-routed inference about batch 1 runtime came hours after that rule was fixed
  in code.
- The false "permutation and asymptotic p disagree at small n" claim arrived inside a
  correction to the p-value methodology.

A correction is written fast, immediately after the impression of having just been rigorous,
and that impression is what lowers the guard. The moment of greatest confidence in one's own
carefulness is the moment the next error enters.

### Operating rule adopted mid-run, for both participants

1. **No number in prose** unless it is pasted verbatim from an output block in the same
   message, or shown with its derivation.
2. **Any derived column is recomputed from its inputs on first sight**, not on suspicion.

Both rules exist because every failure in this file's prose-layer list would have been
caught by one of them, and each was caught late by the other participant rather than early
by the author.

## Named variant: measuring against the knob rather than the quantity it controls

Nominal alpha is a knob; realised density is what it controls, and they come apart per
unit because tanh saturation puts ties at the quantile threshold — unit 88 realises 0.129
at a nominal alpha of 0.2. Every plot in the sweep is therefore drawn against realised
density, with alpha kept as a point label only.

This is a named variant of the recurring shape in this project. Prior instances:

- D5.0 scored **length** when **K** was the variable.
- D5.1 quoted **disjoint pairs** when **unique elements** was the quantity that mattered.
- Q3 read **peak frontier** when **visited** was the cost that tracked wall time.
- This one: **nominal alpha** when **realised density** is what the search sees.

The tell is that the knob is always the thing you typed and the quantity is always the
thing you have to measure, so the knob is what ends up on the axis by default.

## Unique elements vs disjoint pairs

`results/unique_elements.csv`. The paper's §4.3 variable is the fraction of ELEMENTS
carrying exactly one admitted concept: **13.9%** at 24,199 tokens, down from **18.1%** at
2,547. What this project had been reporting instead is `disjoint_pairs`, the fraction of
concept PAIRS that never co-occur: **51.4%** and **52.4%** at the same two sizes. These are
different statistics — one describes the elements, the other the vocabulary — and they
point opposite ways. Scaling the corpus makes the element-level picture worse while the
vocabulary-level number barely moves.

---

# RESULTS — Phase B, 27 pairs, M = 24,199 (2026-07-31)

Full output: `results/phaseB_report.txt`. Per-pair CSV: `results/beam_vs_exact_K15.csv`.

27 exact runs launched, **4 timed out** and are excluded from every derived statistic
(n = 23):

```
trained    a=0.1   unit510   halted=time  t=1514.5
trained    a=0.05  unit87    halted=time  t=1500.2
untrained  a=0.05  unit396   halted=time  t=1503.1
untrained  a=0.05  unit510   halted=time  t=2085.1
```

## The time cap is SOFT

Untrained unit 510 ran to **2085.1s against a 1500s budget** — a 39% overshoot. The budget
check in `HeapProbe.heappush` is amortised to 1 push in 256, so a run whose pushes are slow
(or which spends long stretches between pushes) overshoots by an unbounded amount. Two
consequences: a `halted=time` row's `time_s` is **not** a precise measurement of anything,
and any future timing comparison that includes censored runs must say so. The four timeouts
here are excluded, so nothing downstream depends on it.

## Pre-registered verdicts — all three NOT SUPPORTED

| prediction | stratified rho | permutation p | verdict |
|---|---|---|---|
| #1 primary — IoU gap vs lift | **+0.011** | 0.5230 | NOT SUPPORTED |
| #1 secondary — lift gap vs lift (as registered) | **−0.180** | 0.2442 | NOT SUPPORTED |
| #2 — exact time vs lift | **−0.259** | 0.1507 | NOT SUPPORTED |

This is the **fourth cell** of the pre-registered outcome table: *incumbent strength is not
the mechanism; pushes toward bound-tightening as the only remaining lever.*

The two #1 verdicts agree, so the "disagreement is a result" branch did not fire.

### The INVERTED interpretation was registered in advance and is NOT supported

Batch 1 showed rho = +1.00 on the IoU gap (n=3, one stratum), which prompted registering
the inverted reading — gap as available headroom rather than pruning quality — *before* the
remaining batches. It did not survive. The full result is rho = **+0.011**, indistinguishable
from zero and not significantly positive. Per-stratum: that first stratum held at +1.000
while the other five came in at −0.400, −0.400, +0.400, +0.100, −0.500.

Recording this as the intended use of the registration: the inverted reading was favourable
to the project, was written down before the data, and is now reported as unsupported rather
than quietly retained as an available framing.

## Band comparison — corrected statistic

Table 4 reports "Avg. IoU per category", a **ratio of averages**. The mean and median of
per-unit ratios are **averages of ratios** — a different statistic, and with this right tail
they do not agree. The band verdict tests the ratio of averages only. Choosing the median
after seeing the mean fall outside the band would be post-hoc even though both were
pre-registered.

```
   pooled n=23  sum(exact_IoU)=4.3775 sum(beam_IoU)=4.0371  ratio-of-averages = +8.43%
  trained n=11  sum(exact_IoU)=2.1550 sum(beam_IoU)=1.9795  ratio-of-averages = +8.87%
untrained n=12  sum(exact_IoU)=2.2225 sum(beam_IoU)=2.0576  ratio-of-averages = +8.01%
paper Table 4 (vision): +5.1% .. +6.5%
```

**Verdict: +8.43% pooled, OUTSIDE and above the vision band.** Both arms agree (+8.87% /
+8.01%), so this is not an arm effect. Exact search buys *more* in NLP than in vision, not
less — the "under 1%, exact buys almost nothing" outcome registered as the bigger finding
did **not** occur.

Kept alongside, not band-compared: mean +8.82%, median +6.21%, range +0.00% .. +32.39%.
The median landing inside the band is a coincidence of a different statistic and must not
be quoted against it.

Lift-based, for completeness: mean +7.44%, median +0.70%, range −17.33% .. +120.41%. The
+120.41% is untrained unit 413 at a=0.1, where the coverage ratio is 0.26 — exact narrowed
coverage ~4x and traded recall for precision.

## Q3 — prediction 3 FALSIFIED on the within-unit design

Predicted: search cost falls at low alpha, because a better-scoring formula gives a stronger
incumbent and prunes harder. **Observed: cost rises as alpha falls.**

Support, from the within-unit series (which removes the unit confound that made the earlier
single-unit probes misleading):

- **7 of 8** within-unit series show both `visited` and `time_s` rising as alpha falls. The
  sole exception is untrained unit 88 at a=0.05 (976.9s → 639.4s).
- **3 of 4** timeouts are at a=0.05, the lowest alpha.

```
untrained unit92  | a=0.2: vis= 6650 t= 132.1s | a=0.1: vis=14479 t= 334.6s | a=0.05: vis=21966 t= 718.6s
trained   unit413 | a=0.2: vis= 2063 t=  23.2s | a=0.1: vis=11063 t= 200.7s | a=0.05: --
trained   unit92  | a=0.2: vis=16554 t= 445.9s | a=0.1: vis=23530 t= 886.8s | a=0.05: --
untrained unit510 | a=0.2: vis=16521 t= 427.3s | a=0.1: vis=25031 t= 967.2s | a=0.05: TIMEOUT at t=2085s
```

The lever hoped for in Q3 stays closed.

## HYPOTHESIS FOR (A4), not a finding: the #2 arm asymmetry

Prediction #2's per-stratum rhos split cleanly by arm:

```
  trained:  -1.000  -0.800  -0.400
untrained:  -0.400  +0.200  +0.500
```

The trained arm alone looks like the predicted effect. At n=3–4 per stratum this is **not
evidence** — three strata per arm, and the pooled test was not significant. But it is the
only structured signal across all three tests, and it points at **the arms differing** rather
than at the mechanism being absent. Worth a targeted test in (A4) with more units per arm;
not to be reported as a result.

## Coverage ratio — resolves as "real but weak"

```
mean 1.123  median 1.114  range 0.258 .. 1.783   above 1.0: 14/23
```

Mild widening on average, but **9 of 23 narrow**. This is not the systematic blanket effect
the alpha=0.5 regime showed, so the write-up flag resolves as real but weak — not a finding
about the objective. The single-unit reading from batch 1 (unit 88 at 1.65) would have
overstated it, which is why it was flagged as not-for-inference at the time.

## Band comparison, corrected twice more

> **Headline figure: +5.05%**, the matched-length (length 3) ratio of averages over pairs
> whose solutions differ — on the lower edge of the paper's +5.1–6.5% vision band. The
> **+8.43%** that appears repeatedly below is the *unmatched* length-4 figure, which was the
> working number before length matching and is retained to show how it was corrected. Where
> the two appear together, +5.05% is the comparable one.

### (1) Restriction to differing pairs

Table 4's caption restricts to "all the units in the layer for which the optimal and the
beam search find two different solutions" — agreeing units are excluded from their average.

"Different solutions" is tested **extensionally** (same `n_fires` AND `n_inter`), not by
string equality. Untrained unit 510 at a=0.1 returned two syntactically distinct formulas:

```
beam :  (((const=NP AND dep=pobj) AND (NOT const=VP)) OR lemma=.)
exact:  (((const=NP AND dep=pobj) OR lemma=.) AND (NOT const=VP))
```

These differ only on tokens with `lemma=.` AND `const=VP`, of which the corpus has none, so
they fire on identical token sets — same IoU, coverage ratio exactly 1.000. A string
comparison would have counted this as a differing pair and kept an agreeing unit in the
average.

**Length 4, K=15:** dropped 1 of 23 → **+8.77%** pooled (trained +8.87%, untrained +8.67%).
The restriction barely moves the length-4 figure, because at length 4 beam essentially never
finds the optimum.

### (2) MATCHED LENGTH — this is the one that matters

Table 4 is at max length 3; the Phase B grid was at length 4. Longer formulas give beam more
room to miss, so part of +8.43% could be length rather than domain. Re-ran the entire grid
at length 3 (same units, same alphas, same arms, K=15, exact and beam 200, 27 pairs,
**zero timeouts**):

```
LENGTH 3, all 27 pairs:            +0.96%   (trained +0.54%, untrained +1.38%)
LENGTH 3, restricted to differing: +5.05%   (trained +5.07%, untrained +5.03%), n=7
LENGTH 4, restricted to differing: +8.77%                                       n=22
paper Table 4 (vision):            +5.1% .. +6.5%
```

**At matched length the NLP result sits at +5.05%, effectively on the lower edge of the
vision band.** The earlier claim that "exact search buys more in NLP than in vision" does
not survive length matching — most of the +8.43% was length, not domain.

The other half of the length effect is more striking than the ratio: **20 of 27 pairs agree
exactly at length 3** (extensionally identical, and here also syntactically identical),
against 1 of 23 at length 4. At length 3, beam-200 finds the optimum for three quarters of
units. That is the real matched-length finding, and it is consistent with the paper's own
"beam first" prescription in a way the length-4 numbers obscured.

### (3) Beam-width asymmetry — strengthens whatever gap remains

The paper uses **beam size 5** (Appendix B); this project used **beam 200**, a 40x wider
beam. A wider beam is a strictly better approximation and should produce a *smaller* gap
than theirs, so any gap we measure at or above their band is measured against a handicap in
their favour. This belongs next to every ratio-of-averages figure: it is the main reason to
believe a residual difference is real rather than a beam-strength artifact.

Symmetrically, it makes the length-3 agreement rate more notable, not less — beam-200 agreeing
with exact on 20 of 27 is what a 40x-wide beam should do.

### Length-3 tractability across K (M = 24,199, trained a=0.1, 5 units, exact)

```
K=15  all cats     peak   412..  1728   visited  340..  866   t = 0.4 ..  1.9s
K=30  all cats     peak  1439..  4402   visited  801.. 1279   t = 2.0 ..  4.5s
K=50  all cats     peak  3249..  7537   visited  956.. 1578   t = 4.7 ..  8.5s
K=50  lemma ctrl   peak    41..   474   visited    1..    2   t = 0.0 ..  0.3s
```

No timeouts anywhere, including K=50 all-categories. Against length 4 at K=15 taking
200–1500s with 4 timeouts, **length dominates cost far more than K does** — a 20/27 K
increase at length 3 costs about 4x, while +1 length at K=15 costs 100x or more. The lemma
K=50 disjoint control terminates in 1–2 visited nodes, as expected for a disjoint
vocabulary.

## The discipline found a better number, not a worse one

Worth recording explicitly against the temptation to treat methodological care as a tax.
The sequence here was:

1. The mean (+8.82%) fell outside the vision band.
2. The median (+6.21%) was reached for and reported as "matching the vision result closely"
   — a post-hoc choice of statistic, made because the pre-registered one was unfavourable.
3. The correct like-for-like statistic (ratio of averages) was computed and came out
   **+8.43%**, further outside the band than the mean.
4. Applying Table 4's own caption restriction and matching its length gave **+5.05%**, which
   lands on the band.

So the corrected, fully like-for-like figure came out **favourable** — a clean replication of
the vision result in a new domain — after the post-hoc reach had produced something that only
looked favourable and would not have survived scrutiny. The rigour was not a cost paid for
honesty; it produced the stronger and more defensible result. The post-hoc median would have
been indefensible *and* would have understated what the data actually supports.

---

## Finding: the method's formula grammar is narrower than "length 4 over K concepts"

Found 2026-07-31 by the brute-force oracle (`tests/test_bruteforce_oracle.py`,
`VERIFICATION.md` check 10), which was written to validate our pipeline and instead
surfaced a property of the upstream method.

`expand_node` (`optimal.py:554-582`) grows a formula by exactly three moves, with
`candidate_labels` being plain leaves:

```python
next_op == "OR"  -> F.Or(label, leaf)
next_op == "AND" -> F.And(label, leaf)
next_op == "NOT" -> F.And(label, F.Not(leaf))
```

Three consequences, none of them documented upstream:

1. Formulas are **left-deep** — the right child is always a bare literal, so
   `A AND (B OR C)` is not constructible as a subterm.
2. Negation appears **only as AND-NOT**. There is no `OR NOT`.
3. The leftmost term is **never negated**.

At length 3 this costs nothing on any case tested: the in-grammar max equals the
unrestricted max exactly, on both real units and the proxy. **At length 4 it costs
+0.1586% IoU on untrained unit92** — the unrestricted optimum is
`(tag=NN AND (dep=ROOT OR NOT const=VP)) OR dep=punct`, and the search returns
`(((tag=NN AND NOT const=VP) OR dep=punct) AND NOT tag=IN)`.

The distinguishing move is **OR-NOT**, isolated by running a left-deep enumeration that
permits it (reaches the unrestricted max exactly) against one that forbids it (lands exactly
on the search's value). Tree shape is not involved — the balanced 2+2 shape that first
becomes reachable at length 4 plays no part.

### What this does and does not change

**Does not change any reported number.** Every result in `results/` is what the search
returns, and the search is optimal within the space it constructs — the oracle confirms that
to full precision, recovering the identical formula string.

**Does change the wording.** "Exact search" should read "optimal within the method's
formula grammar". The length-4 beam-vs-exact gap is a gap to the *in-grammar* optimum, not
to the unrestricted optimum, so it slightly understates what a richer grammar could buy.

**The length-3 results are untouched in both senses** — the gap is +0.0000% there, so at the
paper's own max length the distinction does not arise. That includes the +5.05% matched-length
band comparison, which is the headline.

### How this was nearly mis-reported

The first oracle run flagged this as `MISMATCH -- STOP` and the initial reading was "the
search missed its own optimum", i.e. a bug in the pipeline. Two intermediate hypotheses were
also wrong:

1. *"The winner is a 2+2 balanced tree the oracle invented."* Falsified — the winning shapes
   are 1+3 / 3+1.
2. *"It is left-deep-ness."* Falsified — a left-deep enumeration reached the unrestricted max,
   because that enumeration still permitted OR-NOT.

Only the third model, enumerating the exact three moves the code emits, matched. The general
lesson is the same one recorded above under *diagnostics that inherit the confound they were
built to detect*: an oracle is only as good as its model of the thing it is checking, and the
first two models were built by reading the formula classes rather than the expansion function.

---

# EXPERIMENT 1 — LENGTH LADDER at K = 8: does exact search reach length 5?

2026-08-01. `src/exp_length_ladder.py`, raw output in `results/length_ladder/`
(`L3_all_K8.csv`, `L4_all_K8.csv`, `L5_all_K8.csv`, `L5_lemma_K8.csv`, `VERDICTS.txt`).

The move the previous section invites: length dominates cost far more than K does, so spend
the K budget on length. Drop K from 15 to 8 and ask whether exact search reaches length 5.

Design is the length-3 tractability grid extended, not a new one — M = 24,199, arm `all`,
trained alpha = 0.1, `--unit_ids 88 92 396 413 510`, exact, cap 200,000, soft budget 1500s
(Phase B's threshold, reused unchanged). Alpha, unit set and budget are **not free choices**:
they are the only values for which comparable exact length-3 and length-4 numbers already
exist, and any other value breaks the ladder the experiment is built to measure. Lengths 3
and 4 were re-run **at K = 8** so the length-5 number is compared against its own K.

```
length 3, K=8, all cats     peak   150.. 601   visited   59.. 480   t = 0.06..   0.49s   0 timeouts
length 4, K=8, all cats     peak  1525..4165   visited 1138..7114   t = 5.49..  81.02s   0 timeouts
length 5, K=8, all cats     peak 12122..31878  visited 9302..19386  t = 456.85..2095.68  2 timeouts
length 5, K=8, lemma ctrl   peak     9..  56   visited    1..    3  t = 0.00..   0.05s   0 timeouts
```

**Length 5 does not terminate.** 2 of 5 units hit the 1500s soft cap — unit396 at 1516.36s
and unit510 at 2095.68s (the cap is soft, checked once per 256 heap pushes, so it overshoots;
see *The time cap is SOFT* above). Both timeouts are reported here and are in the CSV; neither
is dropped from any statistic below. Because both fall **above** the median, the median time
of 1048.43s is a genuine completed run (unit92) and is unaffected by the truncation.

## Pre-registered verdicts — 4 of 5 SUPPORTED

Registered in `PREDICTIONS` in `src/exp_length_ladder.py` before any length-5 run existed,
and printed by the script rather than read off the table.

| | prediction | outcome |
|---|---|---|
| **P1** | median(L5,K=8) > 1334.1s, i.e. cost overshoots the 1.942x candidate-space ratio | **NOT SUPPORTED** — 1048.43s |
| **P2** | peak frontier grows L4→L5 at fixed K on >= 4/5 units | **SUPPORTED** — 5/5 |
| **P3** | control: median(L4,K=8) < 343.5s | **SUPPORTED** — 12.68s |
| **P4** | control: disjoint lemma arm still trivial at length 5 | **SUPPORTED** — all 5 under 0.05s, 1–3 visited |
| **P5** | length 6 attempted only if median(L5,K=8) < 60s | **SUPPORTED** — gate closed, not attempted |

## Why P1 failed is more interesting than P1

P1 predicted that wall time would overshoot the candidate-space ratio, because A*'s
admissible bound loosens with remaining depth and a deeper search therefore prunes a smaller
*fraction* of its space. It was scored against `(L5,K=8) / (L4,K=15)`, and that comparison
folds two separate scalings together. Decomposed against `(L4,K=8)`, which this experiment
measured for exactly this purpose:

```
per-unit L5/L4 cost ratio at K=8   83.2x (median of the 3 that terminated)   space ratio 24.0x
per-unit K15/K8 discount at L4     36.6x (median of per-unit ratios)         space ratio 12.4x
                                   54.2x (ratio of medians, the form P3 was scored in)
```

Both terms miss the candidate-space model by roughly the same factor, **in opposite
directions**. Length overshoots it 3.5x, exactly the mechanism P1 named. K under-shoots it
by about the same, which P1 did not anticipate — reducing K buys far more than combinatorics
says it should. The two errors cancelled, and the composite ratio landed near the naive
prediction for the wrong reason. `K * (3K)^(L-1)` is a poor cost model in both variables;
it is only accidentally decent in their product here.

Note the two forms of the K discount. P3 was pre-registered on medians of times (54.2x); the
median of per-unit discounts is 36.6x, and the per-unit spread is wide (8.5x to 69.9x). The
36.6x figure is the honest per-unit statistic and is the one used above.

## What length 5 actually buys: nothing measurable

The reason to want length 5 is a better explanation. On the three units that terminated:

```
unit413   IoU 0.0962 -> 0.0962   +0.000%    5.49s ->  456.85s
unit88    IoU 0.1826 -> 0.1827   +0.055%    8.12s ->  703.88s
unit92    IoU 0.1412 -> 0.1425   +0.921%   12.68s -> 1048.43s
```

An 83x cost for at most +0.92% IoU, and exactly zero on one unit. This is a stronger negative
than the timeouts: even where length 5 is affordable it is not worth affording. It also
matches the length-3/length-4 pattern already recorded — the marginal length is where the
cost is, and the explanation quality saturates well before the tractability limit does.

**Caveat on the two timeouts.** unit396 and unit510 are the two most expensive units and
their length-5 IoUs are unknown, so "length 5 buys nothing" is measured on the three cheapest
units of five. It is possible the expensive units are expensive *because* there is something
to find. Nothing here rules that out, and it is not claimed either way.

## Length 6 was not attempted, and the reason was fixed in advance

`SPACE[(6,8)] / SPACE[(5,8)] = 24.0x`, which puts the extrapolated length-6 median at
**25,162s per unit** — about 7 hours each, 35 hours for the grid, and that extrapolation uses
the candidate-space ratio that the section above just showed understates length cost by 3.5x.
P5 gated the attempt on median(L5,K=8) < 60s precisely so this would be a rule rather than a
judgment call made after seeing an unattractive number. The gate closed. The queue's "if
length 5 terminates, try length 6" is moot in any case: length 5 did not terminate.

## The control that did the most work

P4 is the one worth keeping. The disjoint `lemma` arm at K=8 **length 5** terminates in 1–3
visited nodes and under 0.05s — the same trivial behaviour it showed at K=50 length 3. Length
is free on a disjoint vocabulary. Whatever makes length 5 intractable on the `all` arm is
concept overlap (mean overlap 2.711, common_frac 0.738 at K=8), not depth as such. Had lemma
also exploded, the frontier reading in P2 would have been measuring something else entirely.

---

# EXPERIMENT 2a — BEAM WIDTH vs ACCURACY at length 3 (frontier cap)

2026-08-01. `src/exp_beam_width.py`, raw output in `results/beam_width_L3.csv` and
`results/beam_width_L3_VERDICTS.txt`.

Same 27 (arm, alpha, unit) pairs as `results/beam_vs_exact_L3_K15.csv`, K = 15, length 3,
M = 24,199. Widths {5, 10, 25, 50, 100, 200} plus exact, all in-process against the same
masks. Formula equality is tested on the winning **token mask itself**, bit-for-bit, not on
the string and not on the `(n_fires, n_inter)` proxy.

**Read the title carefully.** This measures `MAX_FRONTIER_SIZE`, our frontier cap. It is
**not** the paper's beam. See the correction below and experiment 2b.

## The outcome none of B1–B5 anticipated: narrow beams return nothing at all

```
beam    5   no solution on  8/27
beam   10   no solution on  3/27
beam   25   no solution on  2/27
beam   50   no solution on  0/27
beam  100   no solution on  0/27
beam  200   no solution on  0/27
```

`best_label=None`, `best_iou=-1.0`, `visited=0` — upstream's `best_results = (-1.0, None)`
initialiser is never updated. `perform_search`'s docstring sanctions this ("`best_label` is
a formula object or `None` if no valid formula is found"), so it is not a bug in our patch.

Mechanism: `_apply_beam_cap` keeps the top-N nodes **by estimated ceiling**. An `INDIVIDUAL`
node has been fully resolved and carries its *true* IoU, which sits below the *optimistic*
ceilings of unexpanded nodes. The cap therefore preferentially evicts exactly the nodes that
could have become the answer. The narrower the beam, the more reliably it throws away the
solution it is looking for. Six of the eight no-solution pairs are at alpha = 0.05, the
sparsest units, where true IoUs are lowest and lose to ceilings by the widest margin.

**This broke the statistic.** An IoU of -1.0 inside a ratio-of-averages produces
`-192.81%` at width 5 and `+793.54%` at width 10. Those numbers are in the all-pairs table
in the raw output and are not gaps; they are artefacts and must not be quoted.

## The corrected sweep, on a fixed 19-pair set

The like-for-like fix is a denominator that does not move: the 19 pairs that returned a
solution at **every** width. The 8 dropped pairs are named in the raw output, not silently
discarded.

```
beam     n    agree    roa_all%   band?   median time
   5    19     0/19      22.26     out       0.01s
  10    19     0/19      22.08     out       0.01s
  25    19     2/19      18.52     out       0.02s
  50    19     3/19       7.30     out       0.05s
 100    19     9/19       5.31      IN       0.11s
 200    19    16/19       0.57     out       0.25s
```

**Answer to "at what width does agreement collapse":** between 100 and 50. Agreement halves
from 9/19 to 3/19 across that single step and is gone by 25. The gap statistic moves with it
— it crosses the paper's +5.1–6.5% vision band at **beam 100**, from below at 200 and from
far above at 50.

**Answer to "does beam 5 reproduce the paper's gap":** not with this mechanism. `+22.26%`,
roughly 3.5x the top of the band, and that is *after* excluding the 8 pairs where it returned
nothing. The honest reading is not "beam 5 has a bigger gap" but "beam 5 is not a working
search here at all".

## Pre-registered verdicts

| | prediction | outcome |
|---|---|---|
| **B1** | agreement monotone non-decreasing in width | **SUPPORTED** — 0, 0, 3, 5, 12, 20 over 27 |
| **B2** | roa_all(beam 5) < 5.1% | **VOID** — see below |
| **B2'** | re-scored on the fixed 19-pair set | **NOT SUPPORTED** — +22.26%, overshoots the band |
| **B3** | no collapse: agreement stays >= 14/27 | **NOT SUPPORTED** — 0/27 at beam 5 |
| **B4** | control: exact re-run reproduces committed exact_IoU | **SUPPORTED** — 27/27 within 1e-6 |
| **B5** | control: cap binds only where it should | **SUPPORTED, but vacuously** — see below |

**B2 is void, and this is the interesting failure.** As written it tested `roa_all(beam 5)
< 5.1%`, and `-192.81% < 5.1%` is true, so the script would have printed SUPPORTED. It
would have been a pass earned by the statistic collapsing, on a prediction whose whole
content was that the gap stays small. Recorded here because it is a new instance of an old
shape in this file — *measuring against the knob rather than the quantity it controls* — and
because a one-sided threshold test cannot tell "small" from "broken". The re-score B2' uses
the fixed-set statistic and comes out NOT SUPPORTED, which is the opposite verdict.

**B5 passed vacuously and is not evidence.** It checked that beam-W agrees with exact on any
pair where exact's peak frontier never exceeded W. Exact peak frontiers here run 344 to 2441,
so no run in the whole 162 satisfied the antecedent. The cap bound on every single run. The
prediction was well-formed but untestable on this grid, and it is recorded as untested rather
than as a pass.

**Cross-check that came out clean:** the mask test and `phaseB_report.py`'s
`(n_fires, n_inter)` proxy disagree on **0 of 162** runs, so the existing 20/27 figure is
safe as computed. At beam 200, 20/27 agree extensionally and 20/27 agree as strings.

---

## CORRECTION to "(3) Beam-width asymmetry — strengthens whatever gap remains"

The earlier passage argues: the paper uses beam 5, we used beam 200, "a wider beam is a
strictly better approximation and should produce a *smaller* gap than theirs, so any gap we
measure at or above their band is measured against a handicap in their favour."

**The monotonicity half survives; the inference to the paper does not.** B1 confirms
agreement is monotone in width on our mechanism, so beam 200 is indeed the better
approximation of the two. But the passage then treats our width-5 and their width-5 as the
same object, and they are not:

- Ours is `MAX_FRONTIER_SIZE`, a cap on `optimal.py`'s A* frontier ranked by **estimated
  ceiling**. At width 5 it returns no formula on 8 of 27 units.
- Upstream ships `compositional/beam_optimal.py`, a level-wise beam over **complete formulas
  ranked by exact IoU**, seeded from the previous level's best. Every leaf is a scored
  length-1 formula, so it cannot return `None` on a non-degenerate neuron.

A method that returns nothing on 30% of units is not what the paper reports at its own
default beam size, which is strong evidence the two mechanisms are different. So "their beam
5 is a handicap in their favour" is not a safe inference — it compares our gap against a
weaker version of a *different algorithm*. The original passage is left as written, per this
file's convention; this correction supersedes its final inference only. Experiment 2b runs
`beam_optimal.py` itself to settle it.

---

# EXPERIMENT 2b — THE PAPER'S BEAM (`beam_optimal.py`), and a bug in "exact"

2026-08-01. `src/exp_beam_optimal.py`, `tests/test_bruteforce_oracle_all27.py`.
Raw output in `results/beam_optimal_L3.csv`, `results/beam_optimal_L3_VERDICTS.txt`,
`results/oracle_L3_all27.txt`.

Experiment 2a swept `MAX_FRONTIER_SIZE`. That is not what the paper means by "beam size 5".
Upstream ships a separate algorithm, `compositional/beam_optimal.py`, a level-wise beam over
**complete formulas ranked by exact IoU** (`utils/search_utils.beam_search` puts
`(iou, label)` into a `PriorityQueue(beam_limit)`), seeded from the previous level's best.
This experiment runs it unmodified on the same 27 pairs. Nothing upstream was edited;
`BeamStubConfig` only adds the `get_beam_limit()` accessor.

```
beam |  bo_agree   bo_roa%  bo_nosol |  cap_agree  cap_roa%  cap_nosol
   5 |    19/27      0.71      0/27  |      0/19     22.26       8/27
  10 |    21/27      0.24      0/27  |      0/19     22.08       3/27
  25 |    23/27     -0.08      0/27  |      2/19     18.52       2/27
  50 |    24/27     -0.25      0/27  |      3/19      7.30       0/27
 100 |    25/27     -0.26      0/27  |      9/19      5.31       0/27
 200 |    25/27     -0.26      0/27  |     16/19      0.57       0/27
```

**Answer to "does beam 5 reproduce the paper's gap": no, and in the opposite direction from
2a.** The paper's own algorithm at the paper's own default width gives **+0.71%**, well
*below* the +5.1–6.5% band, not above it. 2a's +22.26% was an artefact of the wrong
mechanism. Beam 5 already agrees with exact on 19 of 27 pairs.

| | prediction | outcome |
|---|---|---|
| **C1** | the paper's beam always returns a formula | **SUPPORTED** — 0/162, vs 13 for the frontier cap |
| **C2** | beam 5 lands inside +5.1–6.5% | **NOT SUPPORTED** — +0.71%, below the band |
| **C3** | beam_optimal tighter than the frontier cap at every width | **SUPPORTED** — all six |
| **C4** | control: exact reference unchanged | **SUPPORTED** — 27/27 within 1e-6 |
| **C5** | registered in advance as UNTESTED | not counted either way |

C5 is worth keeping as a habit: after 2a's B5 passed vacuously, the analogous claim here was
checked for testability *before* running and registered as untested, rather than run and
reported as a pass. The widest swept beam is 200 against a 30,375-formula space, so the
antecedent could never be satisfied.

## The finding this experiment actually produced: "exact" is not exact

The `bo_roa%` column goes **negative** from width 25 up. A negative gap means beam_optimal
returned a *higher* IoU than exact search, which is impossible if exact is optimal.

Exhaustively enumerating all 30,375 in-grammar length-3 formulas in integer popcount
arithmetic, over all 27 pairs (`tests/test_bruteforce_oracle_all27.py`), the search misses
its own in-grammar optimum on **2 of 27 pairs**:

```
trained a=0.2  unit88   in-grammar 0.25454105110196174  search 0.2522022213711222  +0.9274%
trained a=0.05 unit86   in-grammar 0.21660649819494585  search 0.20679723502304148 +4.7434%
```

The other 25 tie to float64 equality. On both, the missed optimum is
`((dep=nsubj OR dep=ROOT) AND C)` — squarely in grammar (`Or(leaf, leaf)` then
`And(..., leaf)`), and `beam_optimal.py` finds it while exact does not.

### Root cause: an inadmissible bound, upstream

`optimal_sample_heuristic.can_improve_or_iou_disjoint_case` (lines 62–75) reasons that for
DISJOINT A and B, "any formula obtainable by (A OR B) is guaranteed to be <= the one
obtainable by only A or only B". True for the OR node in isolation. **False once the node is
extended by AND**, because the AND removes different false positives from each branch:

```
unit86   IoU(nsubj AND NN) = 0.203735   IoU(ROOT AND NN) = 0.056926
         IoU((nsubj OR ROOT) AND NN) = 0.216606     -- beats both
unit88   IoU(nsubj AND NP) = 0.219736   IoU(ROOT AND NP) = 0.056530
         IoU((nsubj OR ROOT) AND NP) = 0.254541     -- beats both
```

`dep` is single-valued, so `dep=nsubj` and `dep=ROOT` share exactly 0 tokens and take that
code path. The OR node is therefore assigned a ceiling below what its own subtree can reach,
and `reduce_frontier` deletes it the moment the incumbent passes that ceiling:

```
unit86   ceiling 0.203398 dropped at threshold 0.203735, subtree could reach 0.216606
unit88   ceiling 0.232677 dropped at threshold 0.232934, subtree could reach 0.254541
```

Both drops are near-misses — the ceiling falls below the threshold by 0.00034 and 0.00026.
That is why only 2 of 27 pairs are affected: the bound is wrong in general but only *bites*
when an incumbent lands in the narrow window between the false ceiling and the true reach.
It also means the failure rate is a property of this corpus, not a bound on the bug.

This is upstream's heuristic, not this project's pipeline. Every mask, quantity helper and
metric in `verify/run_all.sh` still passes.

### What it changes in the published numbers

Substituting the true in-grammar optimum for the 2 wrong exact values (the 2 affected pairs
also move into the "solutions differ" set, so n rises from 7 to 9):

```
(a) ALL 27 pairs          published +0.96%   corrected +1.22%
(b) RESTRICTED to differ  published +5.05%   corrected +4.21%   (n 7 -> 9)
```

**The headline moves off the band.** The matched-length +5.05% was described above as
landing on the lower edge of the +5.1–6.5% vision band; corrected, it is **+4.21%**, below
it. The direction is worth noting: correcting an error that made *exact* look worse pushed
the exact-over-beam gap *down*, because the two corrected pairs enter the restricted set
carrying small gaps.

### Length 4 is UNVERIFIED and this is not a claim about it

`tests/test_bruteforce_oracle.py` at `ORACLE_LENGTH=4` compares grammar expressiveness on
one unit; it has never checked whether the search attains the in-grammar optimum across the
grid at length 4. Nothing here says how often exact misses at length 4. The mechanism has no
reason to be length-3-specific and the window it needs is wider at length 4 (more incumbents,
more chances to land in it), so the length-4 exact numbers — including every Phase B figure —
should be read as **unverified**, not as correct. Scoping a length-4 oracle (1,366,875
formulas x 27 pairs) is the obvious next check and has not been done.

### Why the existing oracle did not catch this

`tests/test_bruteforce_oracle.py` asserts exactly the right thing — `search == in-grammar
max` — and passes honestly. It runs **three** cases: trained unit88 a=0.1, untrained unit92
a=0.1, and the proxy neuron. All three are among the 25 that tie. The assertion was correct
and the sample was too small, which is the same shape as the `min_fire` unit-selection trap
recorded at the top of this file: **a check is only as strong as the set it is evaluated
over.** `tests/test_bruteforce_oracle_all27.py` now runs it over all 27 and is a regression
test against the recorded 2-pair miss set — it fails if a new pair starts missing, or if a
recorded miss disappears (e.g. after an upstream fix).

It is also the second time in this project that a tool built to validate the pipeline instead
found a property of the upstream method, after the formula-grammar finding above. Both times
the first reading was "our code is broken".

---

## CORRECTION to "What this does and does not change" (formula-grammar section)

That section states: *"Every result in `results/` is what the search returns, and the search
is optimal within the space it constructs — the oracle confirms that to full precision,
recovering the identical formula string."*

**The second clause is false.** The search is optimal within its constructed space on 25 of
27 length-3 pairs, not all of them, and the oracle confirmed it only on the 3 cases it ran.
The first clause stands: every result in `results/` is still exactly what the search
returned, and nothing has been silently restated. The corrected band figures above are
published alongside the originals rather than replacing them.

The same section's closing line, *"The length-3 results are untouched in both senses — the
gap is +0.0000% there"*, remains true **as a statement about grammar expressiveness** (the
in-grammar max does equal the unrestricted max at length 3). It is not true as a statement
that the length-3 search results are correct, and it should not be read that way.

---

# RETRACTION — the root cause named in experiment 2b is WRONG

2026-08-01, same day, before any patch was written. `src/exp_disjoint_falsification.py`,
raw output in `results/disjoint_falsification_L3.csv` and
`results/disjoint_falsification_VERDICTS.txt`.

## Source claims, checked against the pinned SHA

Checked in `.upstream-clean/` (verified at `70805299fc0758951a650197bffcc792d0ccca20`), not
against the patched working copy — the distinction matters, because two of these differ
between the two trees.

| claim | verdict | evidence |
|---|---|---|
| `estimate_label_quantities` forks on `are_disjoint()`, and the disjoint branch does **not** receive `neuron_quantities` | **CONFIRMED** | `utils/optimal_utils.py:270-281`; disjoint at :273-275, else-branch passes `neuron_quantities=` at :277-282 |
| upstream `reduce_frontier` is a pure threshold prune, keeps on `-iou >= threshold`, no frontier-size logic | **CONFIRMED** | `compositional/optimal.py:412-428`, keep test at :425. `MAX_FRONTIER_SIZE`, `_apply_beam_cap`, `nsmallest` do not appear in the pinned file at all |
| `can_improve_or_iou_disjoint_case` does not exist upstream | **REFUTED** | it exists: `compositional/optimal_sample_heuristic.py:17` (def), `:93` (sole call site, inside `estimate_disjoint_label_info`) |

The function name stands. My **line citation** was sloppy — "lines 62-75" is the comment and
predicate body, not the `def`; the correct citation is `:17` and `:93`.

## The falsification test, and what it destroyed

One line: force `are_disjoint()` to return `False` unconditionally, so every node takes the
non-disjoint branch. Re-run exact, length 3, all 27 pairs, K=15, alphas matching 2b.

| | prediction | outcome |
|---|---|---|
| **F1** | both known misses recover | **NOT SUPPORTED** — both STILL MISSING, byte-identical IoU |
| **F2** | no pair gets worse | **NOT SUPPORTED** — 5 pairs got worse, worst -1.598e-02 |
| **F3** | runtime rises, no halts | **SUPPORTED** — 0 halts, median 1.07s -> 4.10s (3.83x) |
| **F4** | the 25 already-optimal pairs are unchanged | **NOT SUPPORTED** — 5 of them changed |

**F1 is decisive. The disjoint branch is not the cause.** Both misses return exactly the
values they returned before — `0.2522022213711222` and `0.20679723502304148`, unchanged to
the last bit. F3 rules out the obvious escape: at 3.83x the runtime the forced branch was
demonstrably being exercised, so F1's negative result is interpretable rather than a no-op.

**The root-cause paragraph in experiment 2b is retracted.** Specifically retracted: the claim
that `can_improve_or_iou_disjoint_case`'s disjoint reasoning is what assigns the OR node its
low ceiling and loses the optimum. The *observations* in that section stand — the 2/27 miss
count, the two IoU values, the ceilings and thresholds at the drop, and the fact that
`(A OR B) AND C` beats both branches. What does not stand is the causal attribution. **No
patch was written, and item 8 of the queue is blocked**: its precondition ("if the disjoint
branch is confirmed unsound") is not met — it is confirmed *not* to be the cause.

## What F2 and F4 found instead, and it is worse

Forcing the non-disjoint branch made **5 previously-optimal pairs lose their optimum**:

```
trained   a=0.1  unit510   0.157458 -> 0.147743   (-9.71e-03)
trained   a=0.05 unit87    0.072072 -> 0.072058   (-1.38e-05)
untrained a=0.2  unit92    0.280018 -> 0.264042   (-1.60e-02)
untrained a=0.1  unit396   0.148630 -> 0.141812   (-6.82e-03)
untrained a=0.1  unit510   0.154405 -> 0.152444   (-1.96e-03)
```

So **both branches can lose the optimum**. The disjoint branch is not a sound shortcut that
the general branch lacks, nor the reverse — they are two unsound estimators that fail on
different pairs. The unsoundness is not localised to the fork, and "always take the
non-disjoint branch" is not a fix; it is a different set of misses, and a strictly larger one
here (5 vs 2). The real cause is somewhere in the estimator family itself and is **not yet
identified**. It should not be written up as though it were.

---

# CORRECTION — two different mechanisms were being called the same thing

Experiment 2a and experiment 2b both say "the frontier drops the node", and they are not the
same mechanism. Stated once, properly:

| | experiment 2a | experiment 2b |
|---|---|---|
| function | `_apply_beam_cap` | `reduce_frontier` |
| whose code | **ours** — `patches/0001-frontier-beam-fallback.patch` | **upstream**, `optimal.py:412-428` at the pinned SHA |
| rule | keep top-N by estimated ceiling | keep every node with `-iou >= threshold` |
| bounded by | frontier **size** (`MAX_FRONTIER_SIZE`) | incumbent **IoU** |
| active when | `MAX_FRONTIER_SIZE` is an int (beam runs) | always |

In an **exact** run `MAX_FRONTIER_SIZE is None`, so `_apply_beam_cap` is a verified no-op
(check 3) and only upstream's threshold prune is live. The 2/27 misses are therefore entirely
upstream behaviour and have nothing to do with our patch. Conversely 2a's 8/27 no-solution
runs are entirely our cap and say nothing about upstream's exact search.

**The commit message on `dbeec1c` (experiment 2b) blurs these**, and git history is not being
rewritten, so this note supersedes it. Read that message's root-cause sentence as retracted
per the section above.

---

# RECOUNT — the "beam finds the optimum 20/27" sentence, against true optima

Two assumptions were put to the data. Both fail, and the correction runs the opposite way
from the one expected.

**Assumption: both misses sit inside the frontier-cap beam-200's 7 disagreements. REFUTED.**
Neither does. On both missing pairs the frontier-cap beam returned *exactly the same wrong
value as exact* (`0.252202` and `0.206797`), so they counted as **agreements** and sat inside
the 20/27, not the 7. The misses were found by `beam_optimal`, a different algorithm — the
same conflation the correction above is about.

**True-optimality on all 27 length-3 pairs, measured against exhaustive in-grammar enumeration:**

```
optimal.py exact          25/27
frontier-cap beam-200     18/27      (not 22/27)
beam_optimal-200          27/27
```

So the 20/27 figure was **overstating** the frontier-cap beam, not understating it: it counts
agreement-with-exact, and twice the beam agreed with a wrong answer. Its true-optimal rate is
**18/27**, below its agreement rate.

The sentence still gets rewritten rather than footnoted, but for the other beam. **The
paper's `beam_optimal` at width 200 is exactly optimal on all 27 pairs at length 3** — and
per experiment 2b it is already 19/27 in agreement at width 5. That is the real "beam finds
the optimum" result and it is far stronger than 20/27.

**Matched-length gap against true optima, all 27 pairs:**

```
frontier-cap beam-200     +1.22%
beam_optimal-200          +0.00%
```

At length 3 the paper's own beam has **no optimality gap at all** on this grid. Any gap
previously attributed to beam approximation at this length was measuring our frontier cap, or
measuring exact's own two errors.

---

# EXPOSURE — 2/27 is not a rate, and the premise behind it is also wrong

The at-risk structure is `(A OR B)` followed by `AND C` or `AND NOT C`. Both observed misses
have exactly that shape. Counting it in the vocabulary, with no re-running:

```
                                    K=15                K=50
in-grammar length-3 space          30,375           1,125,000
same-category leaf pairs               22                 242
  ... of which NOT disjoint            3 (const)         14 (const)
at-risk constructions, same-cat
  AND actually disjoint            1,140 (3.75%)     45,600 (4.05%)
at-risk using ALL disjoint pairs   3,240 (10.67%)   174,200 (15.48%)
```

**"Same-category values are mutually exclusive" is false in this vocabulary.** `const` is
constituency labelling and a token sits inside several constituents at once — `const=NP` and
`const=VP` share **8,037** tokens, `const=NP` and `const=PP` share **8,137**. `lemma`, `tag`,
`dep` and `ent` are single-valued and do partition. So category identity is not a proxy for
disjointness, and `are_disjoint` does not use it: it keys off the computed `disjoint_info`
matrix, which is why the category-agnostic row (871 of 1,225 leaf pairs are disjoint at K=50)
is the honest superset.

The exposure is therefore **4.05% of the K=50 space** on the narrow reading and **15.48%** on
the reading that matches what `are_disjoint` actually tests — against 2 observed misses out
of 27 pairs. Those are different denominators (formulas vs pairs) and neither is a rate for
the other. **2/27 must not be quoted as a failure rate**, and now that both numbers are here,
what can be said is: the at-risk structure is common, the observed failures are rare, and the
gap between those two facts is unexplained because the cause is unidentified.

---

# CAVEAT — the disjoint-lemma control is a COST control only

Added before it gets cited as evidence of correctness.

`lemma` is single-valued, so lemma concepts are mutually exclusive by construction and every
OR node in that control takes the disjoint branch: **100% exposure to the estimator path that
was under suspicion**. Its headline behaviour — 1 to 3 nodes visited at K=50 length 3,
terminating in under 0.05s, and the same at K=8 length 5 in experiment 1 — is exactly what a
correct search on a trivial vocabulary looks like. It is *also* exactly what an
under-computed ceiling produces, because both end the search immediately. **Fast and unsound
are not distinguishable from outside.**

The control establishes that cost is driven by concept overlap. It does **not** establish
that the search is correct on that arm, and it must not be cited for that. It has never been
oracle-checked. When the oracle widens to the lemma arm at length 3, that gap closes; until
then the control carries a cost claim only.

---

# CORRECTION — check numbering, and what the suite structurally cannot catch

**Numbering.** `VERIFICATION.md`'s table is canonical: check **9** is "someone else can run
this" (`VERIFICATION.md:73`, prose only, pointing at `REPRODUCE.md` 6b), and the brute-force
oracle is check **10** (`VERIFICATION.md:412`). `verify/run_all.sh:160` printed the oracle as
`9.` until 2026-08-01; that label was the source of the mis-citation, and every `.md`
citation in the repo already said 10. The label is now `10a`, with `10b` for the widened
27-pair check. No diary entry cites the oracle by number, so no diary citation needed changing.

**Widened.** Check 10 ran 3 cases. It now runs 10a (3 cases, plus the expressiveness-gap
measurement) and 10b (`tests/test_bruteforce_oracle_all27.py`, all 27 length-3 pairs, a
regression test against the recorded 2-pair miss set). Suite is 11/11.

**What the suite could never have caught, and this is the load-bearing part.** The other nine
checks are alignment, padding, patch no-op, IoU-vs-upstream, masks-vs-raw-`.feats`, the two
stub checks, model reproduction, and binarisation. Every one of them verifies an **input or a
plumbing step**. Not one of them can fail on a search-optimality bug, because none of them
looks at what the search returns. The suite was built on the assumption that **upstream's
search was correct and only our inputs needed checking** — and that is precisely the
assumption that failed. Check 10 is the only check in the suite that can see a wrong answer,
and it was running 3 cases out of 27.

This attaches to every "10/10, no CANNOT VERIFY" claim in this repo, and to the new 11/11:
**the suite verifies that we fed the method the right data. It does not verify that the
method computed the right answer**, except at check 10, on the pairs check 10 happens to run.

---

# K STAMP, and three corrections to the corrections

2026-08-01.

## K is 15 on every "true in-grammar max" figure in this file

Checked rather than assumed, because a K mismatch between the search and the oracle would
have invalidated every miss count. `exp_beam_width.py:51` sets `K = 15`, and experiment 2b
(`exp_beam_optimal.py`), the 27-pair oracle (`test_bruteforce_oracle_all27.py`), the
falsification test and the no-prune build all import that same symbol.
`tests/test_bruteforce_oracle.py:160` hardcodes 15. The concept list is identical in all of
them (`rts.ARMS["all"][1] == rtm.CATEGORIES`, verified at runtime).

**Every "true in-grammar max", every miss count, and every gap figure in the 2b, retraction,
recount and no-prune sections is at: K = 15, length 3, M = 24,199 tokens, min_support 5,
arm `all`.** The 30,375-formula space is `K*(3K)^(L-1)` at those values.

The only K = 50 numbers anywhere in this file are the exposure counts below. They are counts
of candidate formulas, not optima, and they are not comparable to any K = 15 figure.

## WITHDRAWN: the supersession of the exposure table

An earlier instruction here was to supersede the exposure table on the grounds that F2/F4
showed both estimator branches unsound, so every formula is exposed. **That is withdrawn, and
the reason is a flaw in my own test.**

The falsification harness patched `optimal_utils.are_disjoint` itself, and that function has
**nine** call sites (`utils/optimal_utils.py:170, 187, 199, 200, 205, 209, 210, 269`, plus
its own recursion). Only `:269` is the fork under test. The other eight feed
`compute_disjoint_info` and the path-heuristic's disjointness reasoning. So F2/F4's five
regressions cannot be attributed to the fork: they may come from any of the other eight.

**"Both branches are unsound" is NOT established.** What is established is F1 — the named
function was definitely disabled (single call chain, verified in the call graph) and neither
miss recovered, so the disjoint-branch *hypothesis* is dead. F2 and F4 are uninterpretable as
written and are downgraded to "not evidence of anything yet".

**Status of the exposure table: UNRESOLVED.** The 4.05% / 15.48% figures stay exactly as
recorded. What they bound and do not bound:

- They **do** bound the count of length-3 candidate formulas at K = 50 whose shape is
  `(A OR B)` then `AND C` / `AND NOT C` with A, B disjoint — the shape both observed misses
  have. 4.05% restricts A, B to same-category; 15.48% uses actual disjointness, which is what
  `are_disjoint` tests.
- They do **not** bound the failure surface, because the cause is unidentified and may not be
  the disjoint path at all.
- They are **not** a rate for 2/27, which counts pairs, not formulas.

A fork-only rerun — intervening at `utils/optimal_utils.py:271` so that exactly one call site
changes — is required to replace F2/F4. Until that runs, neither "one branch" nor "both
branches" is supported.

## CODE_WALKTHROUGH.md restored, and what its deletion rationale got wrong

Restored from `0063b8a`. Commit `733ff0c` removed it reasoning "No longer needed. Nothing
referenced it, so no links break."

**"Nothing referenced it" does not establish that a reference document is unused.** A
walkthrough exists to be *read*, and its value is the citations it carries outward, not the
inbound links it collects. Inbound-link count measures whether other documents point at it;
it says nothing about whether the evidence it holds exists anywhere else. Here it did not:
deleting the file removed **99 `file:line` citations**, and afterwards the four `src/`
modules had **no `file:line` citation backing anywhere in the tree** — 31 citations across
`real_token_search.py` (13), `real_token_masks.py` (11), `real_activations.py` (7) and
`synthetic_overlap_sweep.py` (7). The upstream side survived only incidentally, because
VERIFICATION.md and this file cite `optimal.py` independently.

The generalisable form, and it is the same shape as the check-10 sample-size finding: **a
zero inbound-reference count is a statement about the rest of the corpus, not about the
document.** The deletion test should have been "is this evidence reproduced elsewhere", which
it was not.

### Citation drift, measured not asserted

`verify/check_walkthrough_citations.py` resolves all 99 citations against the current trees.
It does **not** edit them; this pass measures.

```
MATCH       39     cited range still holds the anchor the prose names
MOVED       35     anchor exists, at a different line (new line reported)
NO_ANCHOR   25     no identifier extractable from the prose -- needs a human
FILE_GONE    0
```

The document's own header claims all citations were verified against `f1bace0`, and that
upstream citations "are pinned at 70805299 and do not drift". **The second claim is the one
to be careful with**: upstream citations resolve against the *patched* tree, not the pinned
one, because `patches/0001-frontier-beam-fallback.patch` inserts ~21 lines into `optimal.py`.
A first version of this checker resolved upstream paths against `.upstream-clean` first and
reported 42 MOVED; resolving against whichever tree actually holds the anchor gives 35. The
7-citation difference was a resolution artefact of my checker, not drift in the document —
recorded because it is exactly the kind of number that would otherwise have gone into prose.

25 NO_ANCHOR citations are not auto-checkable at all. The document's "98 citations
re-verified" claim covers a set that this tooling can only confirm for 39 of 99, and 99 is
itself one more than the 98 the header claims.

Citations were deliberately not hand-fixed in this pass.

---

# EXPOSURE, restated at the K that was actually run

2026-08-01. Supersedes the presentation of the exposure table above; the K=50 figures are
unchanged and retained. **Status is still UNRESOLVED** pending the fork-only rerun at
`utils/optimal_utils.py:271`.

The earlier table led with K=50, which is not the configuration any of the miss results come
from. Both K are now given, each labelled with the run it describes.

| | **K = 15** | **K = 50** |
|---|---|---|
| **which run** | the 27-pair length-3 grid — 2b, oracle 10b, the falsification test, and C | the length-3 tractability grid only (*Length-3 tractability across K*) |
| **produced any miss result?** | **yes — all of them** | no |
| in-grammar length-3 space | 30,375 | 1,125,000 |
| same-category leaf pairs | 22, of which disjoint **19** | 242, of which disjoint **228** |
| all disjoint leaf pairs | 54 of 105 | 871 of 1,225 |
| at-risk: same-category **and** disjoint | 1,140 (**3.75%**) | 45,600 (4.05%) |
| at-risk: all disjoint pairs | 3,240 (**10.67%**) | 174,200 (15.48%) |

**The figures that belong next to "2 of 27" are the K=15 ones: 3.75% and 10.67%.** The
at-risk shape is `(A OR B)` then `AND C` / `AND NOT C` with A, B disjoint — the shape both
observed misses have. `are_disjoint` keys off the computed `disjoint_info` matrix, not
category identity, so 10.67% is the row that matches what the code tests; 3.75% is the
narrower same-category reading.

Same-category values are **not** all mutually exclusive: 3 of 22 same-category pairs at K=15
overlap and 14 of 242 at K=50, all of them `const` (constituency labels nest — `const=NP` and
`const=VP` share 8,037 tokens, `const=NP` and `const=PP` share 8,137). `lemma`, `tag`, `dep`
and `ent` are single-valued and do partition.

**What these bound.** They bound the count of length-3 candidate formulas carrying the
at-risk shape. They are **not** a failure rate — 2/27 counts pairs, these count formulas —
and they do **not** bound the failure surface, because the cause is unidentified and may not
be the disjoint path at all. Nothing here should be quoted as "the bug affects X% of the
search" until the fork-only rerun has replaced F2/F4.

---

# EXPERIMENT C — UNBUILDABLE AS SPECIFIED, closed

2026-08-01. `src/exp_noprune.py`, adjudication rule `src/exp_noprune_adjudicate.py`.
No results CSV exists: the K=15 run was killed as void before it wrote one.

The goal was a provably-no-prune `optimal.py` — `reduce_frontier` a no-op, `minimum_threshold`
pinned at 0, and the `if node_path_max_iou > 0` gate in `estimate_iou_frontier` always taken —
to ask whether the 2/27 misses survive with pruning off.

**It cannot be built that way. The `>0` gate is also the length bound.** With it disabled the
search leaves the length-3 space entirely:

```
K=5, max_length=3, child-formula length distribution
  published :  {2: 40, 3: 155}
  no-prune  :  {2: 40, 3: 260, 4: 1170, 5: 2642}
```

"Disable every prune" and "stay in the length-3 space" are not jointly satisfiable by this
route. C is closed as unbuildable-as-specified rather than run again.

## The bisect — this is the evidence, kept

At K=4, `max_length=3`, cumulative and then individually:

```
subs[0..0]  reduce_frontier no-op                lengths {2,3}          clean
subs[0..1]  minimum_threshold pinned             lengths {2,3}          clean
subs[0..2]  the >0 gate disabled                 upstream ValueError raised
subs[0..3]  + that ValueError disabled           lengths {2,3,4}        BREAKS THE BOUND
subs[0..7]  + the remaining four                 lengths {2,3,4}  (len-4 count 25 -> 234)

alone: only subs[2] (>0 gate) misbehaves; every other substitution is clean on its own.
```

**I disabled an assertion because it fired, and the assertion was right.** `optimal.py:400-403`
raises `ValueError` precisely when `node_path_max_iou < minimum_threshold`. Substitution 3
violated that invariant, upstream said so immediately, and my substitution 3b silenced the
check as bookkeeping without testing whether it was load-bearing. It was. The failure was
mine, not a subtlety of the method, and the lesson is the plain one: **an assertion that fires
is a result.**

It also explains C's runtime, which was heading past two hours: it was slow *because* it was
contaminated, enumerating lengths 4 and 5.

## The node-count ceiling test, resolved and then made moot

`expanded` **is the `expand_node` call count** — established behaviourally, not by reading the
counter. At K=3/4/5 the reported `expanded` matched an independent count of wrapped
`expand_node` calls exactly: 123, 1,098, 12,351. So of the two candidate ceilings at K=15,
the expand-call one (`K + K*3K = 690`) was the right *kind*, not the generated-node one
(31,065). K=2 could not be used: upstream's `np.partition(quantity_vector, -length)`
(`optimal.py:186`) requires K >= length.

It is moot because **both ceilings are blown regardless** — at K=5 the no-prune build made
12,351 expand calls against a call ceiling of 80, and emitted 4,112 distinct formulas against
a 1,205-node grammar. A node is expanded many times over, so no single-sided count ceiling
could have detected the contamination. **The length distribution did.** The prior
corroboration also pointed away from the naive reading: median `visited/expanded` across the
committed exact runs is 1.24, nowhere near 3K.

`beam_optimal.py` does **not** share this machinery: it imports neither `optimal` nor
`apply_distributive_property`, and expands via `search_utils.compute_next_search_space:102-129`.
The two do implement the same three moves (`Or(f, leaf)`, `And(f, leaf)`, `And(f, Not(leaf))`),
so the grammar is common, but none of C's contamination touches the 2b results.

---

# TARGETED TRACE — the ceiling IS inadmissible, and the hunt narrows

2026-08-01. Published build, unmodified. The two miss pairs only.

P = `(dep=ROOT OR dep=nsubj)`, the length-2 prefix of both missed optima. `true_max(P)` is the
max IoU over every formula extending P by one move, computed from the existing 30,375
enumeration — no new search.

```
trained a=0.2  unit88   IoU(P) = 0.23267674991206472
                        true_max(P) = 0.25454105110196174   ((dep=ROOT OR dep=nsubj) AND const=NP)
                        assigned ceiling = 0.23267674991206472
                        REMOVED by reduce_frontier at threshold 0.23293365307753797
                        ceiling SHORTFALL = 0.021864

trained a=0.05 unit86   IoU(P) = 0.11240400279264604
                        true_max(P) = 0.21660649819494585   ((dep=ROOT OR dep=nsubj) AND tag=NN)
                        assigned ceiling = 0.20339771933907377
                        REMOVED by reduce_frontier at threshold 0.2037351443123939
                        ceiling SHORTFALL = 0.013209
```

`true_max(P)` equals the global in-grammar max on both pairs, so P really is the prefix that
carries the optimum.

**Pre-committed reading 1 fires: ceiling < true_max(P), so the ceiling is inadmissible.** In
both cases the node was removed by `reduce_frontier` — not by the `>0` gate, not by the
node-skip at `:697`.

The ceiling is produced at `optimal.py:366-377` by `path_heuristic.estimate_paths_iou`, whose
`max_score` becomes the node's key at `:389-399`. **The hunt narrows to the terms inside that
call.** Two constraints on it, both already established:

- On unit88 the ceiling **equals `IoU(P)` exactly** — the estimator credited P with zero
  possible improvement from its third move. On unit86 it does not (`IoU(P)` is 0.1124), so
  the two shortfalls are not produced by the same term.
- F1 showed the miss survives with `are_disjoint` forced False, so **both** the disjoint and
  the general estimator produce an inadmissible ceiling for P. The fault is not in the fork.

### Oracle superset check — the miss counts are not inflated

Both searches skip a candidate term already present in the formula
(`expand_node:534`, `compute_next_search_space:119`); the brute-force enumeration does not, so
30,375 is a superset of what either search can construct. Verified harmless: the
distinct-concept max equals the superset max on **27/27 pairs, max abs difference 0.000e+00**.

---

# CORRECTION — the P1 "cancellation" finding was a unit mismatch, and is withdrawn

**P1 itself measured TIME on both sides** (median L5 wall-clock vs 1.942 x median L4-K15 wall
clock), so its NOT SUPPORTED verdict stands as a time-vs-time test.

**The cancellation narrative built on it does not.** That paragraph compared measured *time*
ratios against *formula-space count* ratios and concluded the two errors cancelled. Those are
different quantities, and now that `expanded` is known to be an expand-call count that far
exceeds the distinct-node count, the bridge between them (cost proportional to space size) has
no support. Redone with measured counts from `results/length_ladder/`:

```
                          MEASURED counts      TIME        space model
L5/L4  at K=8                     8.8x        83.2x           24.0x
K15/K8 at L4                      3.7x        36.6x           12.4x
```

The withdrawn claim was "length overshoots the space model 3.5x and K undershoots it by about
the same, so they cancelled". In **count** terms neither overshoots: both come in *below* the
space model (2.7x and 3.4x below), which is what pruning is supposed to do. The overshoot was
entirely in time.

What the measured numbers actually say is cleaner and was invisible before: **per-expand-call
cost is not constant, and it rises by almost the same factor along both axes** — 83.2/8.8 =
9.5x going from length 4 to 5 at K=8, and 36.6/3.7 = 9.9x going from K=8 to K=15 at length 4.
Mask width and estimator work per node grow with both. That is the finding; the cancellation
was an artifact of dividing a time by a count.

Note the three L5 timeouts/censored rows: the L5/L4 count ratio uses only the 3 units that
terminated, and their `expanded` for the two timed-out units is -1 (censored), not zero.

---

# EXPOSURE AUDIT — retires "2 of 27" as a rate

2026-08-01. All 27 pairs, K=15, length 3, M=24,199, published build, no new search beyond
one instrumented pass. For each pair, P* is the length-2 prefix of the true in-grammar
optimum, taken from the existing 30,375 enumeration. The **window** is
`[assigned ceiling, true_max(P*)]`: non-empty exactly when the ceiling is inadmissible.

```
pair                        ceiling   true_max(P*)   window     max threshold  pos in window
trained   a=0.2  unit88    0.232677     0.254541    +0.021864     0.252202       89.3%   MISS
trained   a=0.05 unit86    0.203398     0.216606    +0.013209     0.206797       25.7%   MISS
trained   a=0.1  unit396   0.126125     0.145326    +0.019201     0.145326      100.0%   escaped
untrained a=0.1  unit88    0.121492     0.135269    +0.013777     0.135269      100.0%   escaped
untrained a=0.1  unit413   0.128911     0.144438    +0.015527     0.144438      100.0%   escaped
untrained a=0.05 unit413   0.086004     0.103774    +0.017769     0.103774      100.0%   escaped
                         ... 20 pairs with a NEGATIVE window (admissible ceiling) ...
trained   a=0.1  unit92    P* never entered the frontier
```

**The real numbers:**

- **Ceiling inadmissible on 6 of 27** (window > 0). This is the exposure.
- **P* was dropped by `reduce_frontier` on all 6 of those.** The threshold entered the window
  every time it could.
- **The optimum was actually lost on 2 of the 6. Four escaped by luck** — the search reached
  `true_max(P*)` by a *different* prefix, so dropping P* cost nothing. Their `pos in window`
  is exactly 100.0%, which is what that means: final answer == true_max(P*).
- Ceiling admissible on 20 of 27; P* never reached the frontier on 1.

**"2 of 27" is retired.** It was never a rate — it counted the pairs where an
inadmissible ceiling happened to be *unrecoverable*, not where it occurred. The occurrence
rate is **6 of 27**, and the 4 escapes are luck about alternative prefixes, not evidence of
correctness. A corpus where the optimum has a unique prefix would convert those escapes into
misses.

## Aggregated, not refined, at the moment of the drop

The node's 5th tuple element is the heuristic that produced its current estimate. At the
`reduce_frontier` call that dropped it, P* carried the **aggregated (`"sum"`) estimate on
both miss pairs** — it had not been popped and refined to the sample-based estimate. Across
all 27 the only pair where P* ever carried `"sample"` is untrained a=0.2 unit88, whose window
is negative anyway.

So the estimate that was wrong is the **aggregated** one, and it was wrong *before* the node
was ever popped for refinement.

## BLOCKED: the bug-vs-paper discriminator cannot be run here

Deciding whether this is an implementation bug or an inadmissible published estimator needs
the paper's closed form for the aggregated AND-path estimate, hand-computed from the masks
and compared against the assigned ceiling `0.23267674991206472`.

**The paper text is not in this repo or the pinned upstream.** `.upstream-clean/PAPER.md` is
a 314-line *reproduction-instructions* document — Docker setup, dataset download, run
commands. It contains no equations, no Algorithm 1, and no Section 4.3.

Reconstructing the equations from `optimal_sample_heuristic.py` would be circular: it would
compare the code against itself and could only ever return "match". **No verdict is recorded**
until the paper's equations are supplied.

---

# WARM START — prediction NOT SUPPORTED, and the test cannot discriminate at length 3

`src/exp_warmstart.py`, raw output `results/warmstart_L3.csv`. Registered before running:
**W1** miss count > 2, on the argument that `reduce_frontier` drops on
`ceiling < threshold`, so a higher starting threshold can only enlarge the wrongly-dropped
set. **W2** no pair improves.

```
cold miss set (2): trained a=0.2 unit88, trained a=0.05 unit86
warm miss set (0): --
W1: NOT SUPPORTED   0 vs 2
W2: NOT SUPPORTED   both cold misses "repaired"
```

**Both verdicts are real but the experiment is degenerate, and that is the finding.**
`beam_optimal-200` is 27/27 true-optimal at length 3, so seeding the incumbent with it seeds
**the optimum itself**. A search that starts holding the optimum cannot finish below it,
no matter how much it wrongly drops. The mechanism W1 described is real and still operates —
it just cannot show up in the outcome.

Direct evidence that it operates: **on 12 of 27 pairs the warm-started search returned no
label at all.** It dropped every candidate and contributed nothing; the seed carried the
result. That is the predicted enlargement of the dropped set, visible in the search's own
output rather than in its IoU.

**Scoring correction made before reporting.** Those 12 no-label runs come back as
`best_iou = nan`, and the first scoring pass counted `nan` as "not a miss" because
`true - nan > tol` is False — silently scoring 12 no-solution runs as successes. Corrected by
treating a `None` return as "the search found nothing better than the seed", i.e. effective
answer = seed. Same verdicts either way here, but the first pass was wrong for the same
reason 2a's ratio-of-averages was wrong: **a sentinel that is not a number must be handled
before it reaches a comparison, not after.** Third instance of that shape in this file.

**This does not test the paper's Section 4.3 mitigation.** A discriminating version needs a
setting where beam is *not* already optimal — length 4, where beam-vs-exact actually
disagrees. Not run.

---

# THE PAPER'S NON-DEGENERACY ASSUMPTION DOES NOT HOLD ON THIS CORPUS

2026-08-01. K=15, M=24,199, min_support=5, arm `all`, neuron = trained a=0.2 unit88.

Paper Section E.2.2 calls the aggregated `|Union_min|` exceeding 0 a "rare degenerate case
(not observed in any of the datasets tested in this paper)". That case is
`Bott_1(E^C)_x != 0`.

```
common C = |{i : covered by >1 concept}|     20,280 of 24,199   (83.8%)
unique                                        3,369
uncoverable                                     550

E^C_j = |C AND concept_j AND NOT neuron|, per concept, ascending:
   dep=nsubj    859     dep=ROOT     1562     lemma=.      1745     dep=punct   2033
   dep=pobj    2567     lemma=a      2696     synset=...   2696     dep=prep    2740
   tag=IN      2832     dep=det      4041     tag=DT       4070     tag=NN      4138
   const=PP    9096     const=VP    10464     const=NP    12726

|SE^C| = |C AND NOT neuron|            = 17,643
Bott_1(E^C)_x  (min over concepts)     = 859
Bott^A_1(E^C)  (min dataset-wide total) = 859

fraction of samples with Bott_1(E^C)_x != 0  =  1/1  =  100%
```

**It holds on 100% of samples, not rarely.** The smallest per-concept extras-in-common count
is 859, an order of magnitude above zero — this is not a marginal violation.

The mechanism is exactly as predicted: `min_support` + top-K selection picks the highest-support
features, so **83.8% of tokens are covered by more than one concept** and every one of the 15
concepts has thousands of common locations where the neuron is silent. In vision each image
is a sample and a given concept is absent from most images, so `Bott_1` hits 0 routinely. Here
it cannot.

**One sample, not many.** `run_one` builds `bitmaps = neuron_bits.reshape(1, M)` — the whole
24,199-token corpus is a **single sample**. So "fraction of samples" is over N=1, and the
per-sample and dataset-wide aggregations largely coincide. This is a structural difference
from the vision setting that was not previously recorded and that bears directly on which of
the paper's two estimator forms is even meaningful here. `load_tokens` does not expose a
token-to-sentence map, so the per-sentence view was not computed.

## Transcription discrepancy to resolve before the Eq (50)/(51) arithmetic

Verified against source (`utils/optimal_utils.py:477-521`, pinned SHA). `Eq (15)` as
implemented is:

```python
max_label_common_intersection_sum = min(max_common_intersection_sum + unique_intersection_sum,
                                        neuron_coverable_sum)              # :512-514
label_iou = max_label_common_intersection_sum \
            / (num_hits + min_common_extras_sum + unique_extras_sum)       # :517-520
```

The denominator matches the shape of Eq (51) — `|1N|` is `num_hits`, plus common and unique
extras. **The numerator does not match the transcribed Eq (50).** The transcription caps with
`Top^A_1(I^C)`; the code caps with **`neuron_coverable_sum`** — the neuron's own coverable hit
count, not a top-1 concept-wise intersection total. It also adds the unique-intersection term
inside the `min`, which the transcription does not show.

That difference sits exactly on the numerator of the ceiling under test, so assembling the
arithmetic before resolving it would produce a number that cannot decide bug-vs-paper either
way. **The arithmetic is deferred pending confirmation of Eq (50) against the PDF**, not
abandoned. The remaining terms (`|SE^C|`, `Bott^A_1(E^C)`, per-concept `E^C_j`) are computed
above and ready.

---

# EXPOSURE — final wording

**`reduce_frontier` dropped the optimum-carrying prefix on 6 of 27. On 4 of those the search
recovered because an equal-valued optimum was reachable through a different prefix — a
property of this corpus, not of the algorithm.**

"2 of 27" is retired as a rate everywhere. Where it still appears in this file it is either a
count of *unrecoverable losses* (labelled as such) or an unrelated figure (beam no-solution
counts, agreement counts). The occurrence rate is 6 of 27.

---

# WARM START — recorded as split, not as a clean refutation

**Not supported as stated.** W1 predicted "miss count > 2" and the outcome was 0.

**But the prediction substituted quantities.** It reasoned about *nodes wrongly dropped* and
was written as a prediction about *misses reported*. With a seed that is already 27/27
optimal, the reported answer cannot fall below the seed however many nodes are dropped, so the
two quantities cannot agree. Recorded as a quantity substitution in the prediction — the same
error shape as "measuring against the knob rather than the quantity it controls", now at the
level of what a prediction is written about rather than what a statistic measures.

**The supported half:** 12 of 27 warm runs returned **no label at all** — the search dropped
every candidate and contributed nothing. That is the predicted enlargement of the dropped set,
visible in the quantity where it shows.

**Length 4 is the discriminating setting and stays DEFERRED, not dropped.** It is the only
setting where beam is not already optimal, so it is the only place the Section 4.3 mitigation
can be tested.

---

# STANDING RULE — a non-numeric sentinel must never reach a comparison

Third instance in this project, so it becomes a rule rather than a note.

`nan`, `None`, and no-solution returns fail **silently as successes** under a one-sided test:
`true - nan > tol` is `False`, so a run that produced no answer scores as "not a miss".
`-1.0` sentinels inside a sum produce a ratio-of-averages that is not a ratio.

**Rule: assert numeric-and-finite at every verdict boundary, and count sentinels in their own
bucket before any comparison runs.**

The three instances:

1. Experiment 2a — `best_iou = -1.0` entered a ratio-of-averages, producing `-192.81%` and
   `+793.54%`. Pre-registered B2 would have printed SUPPORTED off the artefact.
2. Experiment 1 — timeouts carried observed wall-clock into a median. Caught because both
   timeouts sat above the median, so the median was unaffected; that was luck.
3. Warm start — 12 `nan` returns counted as successes in the first scoring pass.

**Sweep of the remaining verdict comparison sites**, all one-sided differences of the form
`true - x > tol` or `delta < -tol`, every one of which would mis-score a sentinel:

```
tests/test_bruteforce_oracle_all27.py:129,140     src/exp_noprune.py:264,269,313
src/exp_disjoint_falsification.py:159,208,220     src/exp_noprune_adjudicate.py:148
src/exp_beam_width.py:364                         src/exp_beam_optimal.py:321
```

None is currently mis-scoring: the oracle and falsification paths never produce `None`
(exact search always returned a label there), and `exp_noprune` is void. They are listed
because the guard is missing, not because a failure is known. `exp_beam_optimal.py:172` does
handle it correctly (`best_iou if best_iou == best_iou else None`) and is the pattern to
follow.

---

# PROTOCOL DEVIATION 3 — the sample axis was never registered, and it produced the misses

2026-08-01. `src/exp_partition.py`, raw output `results/partition_L3.csv`.

## The deviation

The paper's **D** is the set of dataset inputs — images in vision, **sentences** for SNLI.
`real_token_search.run_one` builds `bitmaps = neuron_bits.reshape(1, M)`: **one sample
holding all 24,199 tokens.**

Section E.2.2's degenerate case is "a single sample contains all the concepts in the dataset".
**This configuration satisfies that by construction.** It is not a property of NLP, not a
finding about token-level concepts, and not a difference between language and vision. It is a
representation choice that was made implicitly and never written down.

This is the **third protocol item**, and it is of a different kind from the first two:

| | item | kind |
|---|---|---|
| 1 | alpha (activation range) | a **miss** — the right knob, the wrong value |
| 2 | K (concept count) | a **scope choice** — declared, defensible, bounded |
| 3 | **the sample axis** | **never registered at all** — no value was chosen, because no one noticed there was a choice |

## Blast radius, precisely — this is NOT "everything is in question"

**UNAFFECTED.** IoU is partition-invariant (Lemma 3.6): it is a ratio of two counts over the
same element set, and regrouping elements into samples changes neither count.

- every IoU value in `results/`
- the brute-force oracle and both miss *magnitudes* (+0.9274%, +4.7434%)
- the unique/common decomposition and `unique_elements.csv`
- the alpha sweep, all lift figures, the trained/untrained comparison
- the beam-vs-exact IoU gaps and the band comparisons

**Verified, not asserted:** under the per-sentence partition all **27/27** pairs return the
same in-grammar optimum, to float64 equality. That was checked first, as control V0, before
anything else was read.

**AFFECTED.** Everything computed from per-sample quantities that are then summed —
`Top_t(Q)_x`, `Bott_1(Q)_x`, and every place `SUM_x min(a_x, b_x)` is approximated by
`min(SUM a, SUM b)`:

- every bound and ceiling
- all pruning decisions
- all runtime, all node counts, all peak-frontier figures
- termination and every timeout

## The result: both misses were an artifact of the single sample

Per-sentence partition, 2,000 sentences, max length 57, grid 2000x57 (padding cells belong to
no concept and to no neuron, so they are uncoverable and cannot affect IoU). Concepts, masks,
alpha and K unchanged. Length 3, all 27 pairs.

```
V0  CONTROL   27/27 reach the brute-force optimum                       PASS
P1  Bott_1(E^C)_x == 0 on >= 50% of sentences                           SUPPORTED
      median 69.7% (min 62.3%, max 92.9%), against 0 of 1 flat
P2  the 6/27 dropped-prefix exposure falls                              SUPPORTED
P3  both known misses recover                                           SUPPORTED
P4  expanded-node count falls at length 3                               SUPPORTED (trivially)
```

```
pairs missing the in-grammar optimum:   flat 2/27  ->  per-sentence 0/27
trained a=0.2  unit88   0.25454105110196174   ((dep=ROOT OR dep=nsubj) AND const=NP)
trained a=0.05 unit86   0.21660649819494585   ((dep=ROOT OR dep=nsubj) AND tag=NN)
```

Both now return **exactly the formulas `beam_optimal` had been finding**, at exactly the
brute-force optimum.

**"Exact search is not exact" is retracted as a statement about the method.** The correct
statement is: *under a single-sample representation that violates the paper's stated
non-degeneracy condition, the aggregated bound is inadmissible and the search loses the
optimum on 2 of 27 pairs. Under the paper's own sample definition it does not.* The 6/27
ceiling-inadmissibility exposure is a property of the mis-specified sample axis, not of
`optimal.py`.

## P4 is supported by a margin too small to mean anything

```
metric      flat (1 sample)   per-sentence   ratio
expanded            477.0          474.0     0.99x
visited             579.0          318.0     0.55x
peak                962.0         1143.0     1.19x
time_s                1.0            0.9     0.91x
```

Median `expanded` fell by **three nodes**. The registered test says "falls" and it fell, so
P4 is scored SUPPORTED, but it should not be read as evidence of tighter bounds: `visited`
nearly halved while **peak frontier rose 19%**, which is not the signature of uniformly
tighter pruning. Recorded as supported-and-uninformative rather than quietly upgraded into
the bound-tightening story it was written to test.

## Consequences for what is already written

- The **RETRACTION** section stands: the disjoint branch was still not the cause, and F1's
  reasoning is untouched.
- The **targeted trace** stands as a description of *how* the loss happened in the flat
  configuration (inadmissible aggregated ceiling, dropped by `reduce_frontier`), and it is now
  explained *why* the ceiling was inadmissible: `Bott_1(E^C)_x` can never reach 0 with one
  sample, so the aggregated estimator was operating outside its stated precondition.
- Experiment 2b's beam-vs-exact numbers were computed in the flat configuration and are
  **affected in their pruning-derived columns** (node counts, times), not in their IoUs.
- The 4 "escaped by luck" pairs are no longer luck-dependent under the correct partition.

---

# LENGTH-3 PARTITION — the full result, including the part that does not fit

2026-08-01. Reported before the length-4 run landed, so none of it is shaped by that outcome.

## Precondition: IoU is partition-invariant — VERIFIED, all 27

```
same    25/27      HIGHER   2/27      LOWER   0/27
```

`LOWER = 0` is the requirement. The two HIGHER pairs are exactly the two the flat run lost:

```
trained a=0.2  unit88    flat 0.252200  ->  per-sentence 0.254541  = true in-grammar max
trained a=0.05 unit86    flat 0.206800  ->  per-sentence 0.216606  = true in-grammar max
```

The repartition is correct, so P3 is readable.

## P3 — CONFIRMED as a result

**Both misses recovered. 2/27 -> 0/27.** Under the per-sentence partition the search returns
the true in-grammar optimum on all 27 pairs, and on the two formerly-missed pairs it returns
exactly the formulas `beam_optimal` had been finding:

```
trained a=0.2  unit88   0.25454105110196174   ((dep=ROOT OR dep=nsubj) AND const=NP)
trained a=0.05 unit86   0.21660649819494585   ((dep=ROOT OR dep=nsubj) AND tag=NN)
```

## P1 — SUPPORTED, large

```
fraction of sentences with Bott_1(E^C)_x == 0 :  min 0.623   median 0.697   max 0.929
flat (one sample)                             :  0 of 1
```

## P2 — SUPPORTED ONLY NOMINALLY, AND IT UNDERCUTS THE REGISTERED READING OF P3

This is the measurement that had to be made, and it does not say what the P3 result invited
us to assume.

```
prefixes with an INADMISSIBLE ceiling   flat 6/27  ->  per-sentence 5/27
of those, dropped by reduce_frontier    flat 6/6   ->  per-sentence 5/5
```

The exposure fell by **one pair**, and the set **churned** rather than shrinking:

```
became admissible : untrained a=0.1 unit413, untrained a=0.05 unit413
became INADMISSIBLE (new, worse under the partition) : trained a=0.1 unit88  (-0.020312 -> +0.015224)
still inadmissible: trained a=0.2 unit88, trained a=0.05 unit86, trained a=0.1 unit396,
                    untrained a=0.1 unit88
```

**Both formerly-missed prefixes still have an inadmissible ceiling, at byte-identical values**
— `0.232677` for unit88 and `0.203398` for unit86, unchanged from the flat run — **and both
are still dropped.** So P3 did **not** pass because those ceilings became admissible. They did
not.

**P3 passed for a reason this experiment has not identified.** Registered mechanism A3 —
refine-on-pop becoming non-trivial at |D| ~ 2,000 — remains the leading candidate and is
consistent with the ceilings being unchanged while the outcome changed, but it is **not
demonstrated here**. Stated plainly so it is not quietly upgraded: *the misses recover under
the per-sentence partition; why they recover is open.*

A defect in the P2 metric itself, recorded rather than patched after the fact: `dropped` is
`any(scan where the node sat below threshold)`, which does **not** imply the optimum was lost.
A node can be expanded early — producing the winning child — and its stale frontier copy
dropped later as the threshold rises. That is normal and harmless. In the flat run the drop
demonstrably caused the loss because the loss was observed; under the partition 5 prefixes are
dropped and 0 optima are lost, which is exactly the case the metric cannot distinguish. **The
exposure counts are an upper bound on harm, not a measure of it.**

## P4 — SUPPORTED, uninformative, matched set

All 27 terminated in both runs, so the matched set is all 27.

```
median expanded   flat 477  ->  per-sentence 474   ratio 0.99x
```

Three nodes. Recorded as supported-and-uninformative, per A1/A2.

## Mid-run process check (registered as a check, not assumed)

```
pgrep matches : 1        PID 65052        child processes : 0
process started       : Sat Aug  1 15:38:09 2026
docstring commit b7bff63 : 2026-08-01 15:50:38 -0700
```

The process started **12.5 minutes before** the docstring commit, is a single PID with **zero
children**, and Python binds module source at import. Nothing re-imports or forks, so the
mid-run edit cannot have reached the running search. Checked, not assumed.

---

# UPSTREAM REPORT — drafted before the length-4 result exists

Framing fixed now so it cannot be written to fit that outcome.

**This is not a bug in the method. It is an unguarded precondition.**

1. **What happens.** With `|D| = 1` the aggregated heuristic's admissibility fails and the
   exact search can drop the branch holding the optimum. Measured: 2 of 27 (arm, alpha, unit)
   pairs at K=15, length 3, M=24,199 return a strictly sub-optimal in-grammar formula
   (+0.9274% and +4.7434% below the brute-force optimum).

2. **Why.** Admissibility of the aggregated form depends on `Bott_1(E^C)_x = 0`. Paper
   Section E.2.2 states the violating case is "rare" and "not observed in any of the datasets
   tested". Measured here: `Bott_1(E^C)_x = 859`, on 1 of 1 samples — **100%**, and 859 is an
   order of magnitude from zero, not marginal.

3. **How it was reached.** By our own harness: `real_token_search.run_one` builds
   `bitmaps = neuron_bits.reshape(1, M)`, putting the entire corpus in a single sample. That
   satisfies E.2.2's degenerate condition *by construction*. **The caller did this, not the
   method.**

4. **The actual defect: the code does not detect it.** `optimal.py` computes the aggregated
   estimate unconditionally. There is no assertion, no warning, and no fallback to the sample
   form when the precondition fails. A caller who partitions their data wrongly — or who, like
   us, does not realise the sample axis is a modelling choice — gets silently sub-optimal
   answers from a function documented as exact.

5. **Proposed fix, to offer rather than assert:** an assertion or warning at
   `get_optimal_heuristic_info` when `Bott_1(E^C)_x != 0` on any sample, naming the
   precondition and pointing at E.2.2. Cheap to compute; it is already a by-product of the
   quantity helpers.

6. **Reproduction to supply:** `tests/test_bruteforce_oracle_all27.py` (2/27 miss, exact
   values recorded), `src/exp_partition.py` (0/27 under the per-sentence partition), and the
   `Bott_1` measurement above.

**Every miss measurement is kept.** The 2/27, the two IoU gaps, the ceilings, the thresholds,
the targeted trace — all stand exactly as recorded. **Their cause changed; their existence did
not.** What is retracted is only the claim that they characterise `optimal.py` rather than our
configuration of it.

---

# CORRECTION — P3 recovered by POP ORDERING, not by refinement

2026-08-01. Supersedes the reading offered when P3 was first reported, and supersedes
registered mechanism A3 as the explanation.

A3 proposed that recovery came from Algorithm 1's refine-on-pop step (lines 14-21) becoming
non-trivial once |D| went from 1 to ~2,000. **That is not what happened.**

The evidence is the exposure audit under the per-sentence partition:

- Both miss-carrying prefixes **still have an inadmissible ceiling**, at **byte-identical
  values** — `0.232677` (trained a=0.2 unit88) and `0.203398` (trained a=0.05 unit86).
- Both are **still dropped** by `reduce_frontier`.
- 5 prefixes are dropped in total, and **0 optima are lost**.

Refinement did not rescue P. P was never rescued: **the bound is unchanged and the drop still
happens.** What changed is *when* the drop happens relative to the expansion. The winning
formula is generated because P is **popped and expanded before** a later, stale copy of it is
pruned. The optimum is produced by an expansion that had already occurred; the subsequent drop
removes a frontier entry that no longer matters.

**The recovery is a pop-ordering effect, not a soundness improvement.**

The consequence has to be stated plainly, because it is the part that matters:

> **The misses can recur under any perturbation of pop order.** Nothing about the
> per-sentence partition makes the search sound. Heap tie-breaking, a different concept
> ordering, a different `min_support`, a changed `K`, a different corpus, or an unrelated
> upstream change to insertion order can all reorder pops and reinstate the loss. The 0/27
> result is a property of one particular execution order, not a guarantee.

This also explains why the flat run's four "escaped by luck" pairs escaped: the same
mechanism, succeeding. "Luck" was the right word and it is now named.

---

# STANDALONE RESULT — the aggregated estimate is PARTITION-INVARIANT

Recorded separately because it is what makes the byte-identical ceilings meaningful, and
because it rules out an entire class of fix.

**Derivation.** The aggregated form is built from three ingredients:

- `SUM_x |I^C_max(L)_x|` — a count over all `(x, j)` pairs. Regrouping elements into samples
  changes which `x` an element is filed under; it does not change the set of elements, so the
  sum is unchanged.
- `Top^A_t(Q)` — concepts sorted by their **dataset-wide** total of `Q`, cumulative over the
  top `t`. Concept-wise over totals; the sample axis never enters.
- `Bott^A_1(Q)` — the **minimum dataset-wide total** over concepts. Same.

None of the three depends on how elements are partitioned into samples. Therefore **the
aggregated estimate, and every ceiling derived from it, is invariant under repartition.**

**Empirical confirmation, which is why this is a result and not an argument:** across the flat
one-sample and per-sentence partitions, both miss-prefix ceilings are identical to the last
bit — `0.232677` and `0.203398`. Two different sample axes, same number.

**What this rules out.** Repartitioning is **not a fix** for this class of error. Under the
per-sentence partition:

```
inadmissible ceilings   6/27 -> 5/27       (a fall of one pair)
set composition          CHURNED: trained a=0.1 unit88 went -0.020312 -> +0.015224,
                         i.e. ADMISSIBLE under the flat axis and INADMISSIBLE under the
                         "correct" one
```

The correct partition does not repair the bound. On one pair it makes it worse. Anyone
reaching for "just partition the data properly" as the remedy is reaching for something this
measurement excludes.

Note the asymmetry that makes this coherent: `Bott_1(E^C)_x` — the *per-sample* quantity whose
vanishing E.2.2 requires — **is** partition-dependent (0% of samples comply flat, 62-93%
per-sentence). The *aggregated estimate built on top of it* is not. So repartitioning fixes
the compliance statistic without fixing the bound it is supposed to license.

---

# UPSTREAM REPORT — SUPERSEDED. The earlier draft was too generous.

Supersedes the draft above, which framed this as a caller error with an unguarded
precondition. That framing was wrong in the caller's favour and is withdrawn.

**1. The aggregated estimate is inadmissible whenever `Bott_1(E^C)_x != 0` for ANY `x`.**
E.2.2's condition is **universal**, not typical. It is not "rarely violated"; it is "must hold
everywhere, and here it holds nowhere".

**2. Correct partitioning does NOT repair it.** The aggregated estimate is partition-invariant
(derivation and byte-identical ceilings above). Under the paper's own sample definition —
one sentence per sample — **5 of 27 prefixes remain inadmissible, and one pair is worse than
under the degenerate flat axis.**

**3. `reduce_frontier` acts on the unrefined aggregated estimate, pre-refinement**
(Algorithm 1 lines 11, 52). So whether the optimum survives depends on **pop ordering** —
whether a node is expanded before a stale copy is pruned. Under the per-sentence partition
that ordering happens to be favourable on all 27; under the flat axis it was not on 2. **The
search's optimality is order-dependent, not bound-guaranteed.**

**4. What `reshape(1, M)` did, precisely.** It made the violation **universal** — 0% of
samples compliant against 62-93% per-sentence. **It amplified exposure. It is not the cause.**

**CORRECTED 2026-08-02:** this section previously continued *"The cause is that an inadmissible
estimate is used as a pruning bound."* **That causal claim is withdrawn.** The aggregated bound
is measurably inadmissible, and the search measurably returns non-optimal answers, but
instrumentation does not link them: on both losing pairs the prefix was CREATED 4x and EXPANDED
3x despite its inadmissible ceiling, and the child was produced regardless and then died with
no DROPPED event and an admissible refined estimate. The two facts stand; the link does not.

**Two remedies, offered in order of strength:**

- **(a) Do not prune on an unrefined aggregated estimate.** Refine before `reduce_frontier`
  acts, or restrict pruning to refined nodes. This addresses the defect rather than its
  detection, and removes the order-dependence.
- **(b) Assert or warn** at `get_optimal_heuristic_info` when `Bott_1(E^C)_x != 0` on any
  sample, naming E.2.2. Cheap — already a by-product of the quantity helpers — but it only
  reports the violation; it does not make the search sound, and per point 2 a compliant
  partition is not sufficient either.

**Reproduction to supply:** `tests/test_bruteforce_oracle_all27.py` (2/27, exact values),
`src/exp_partition.py` (0/27 per-sentence, with the pop-ordering caveat), the `Bott_1`
measurement (859, 1/1 samples flat; 62-93% compliance per-sentence), and the byte-identical
ceilings across partitions.

**Every miss measurement is kept, unchanged.** Their **cause is unidentified** — see the W2
result and the exclusions. Their existence was never in question.

---

# CORRECTION to C2 — the length-4 reference is a one-sided check, not void

`max_length` is a **maximum**, not an exact length. Source: `optimal.py:566` marks a node
`INDIVIDUAL` only when `len(candidate_formula) == max_length`; `:575` computes
`available_spots = max_length - len(candidate_formula)`; shorter formulas are scored via the
ancestor-propagation block at `:816-847`. Empirically, **1 of 27 length-3 runs returned a
formula with fewer than 3 leaves**, in both the flat and per-sentence runs.

So the length-4 in-grammar space **contains** the length-3 space, and comparing a length-4
result against the length-3 optimum is sound in one direction:

```
part_L4 <  true_L3   ->  DEFINITE MISS, no length-4 oracle needed
part_L4 >= true_L3   ->  INCONCLUSIVE, cannot confirm length-4 optimality
```

`in_grammar_max` and `missed` in `partition_L4.csv` are therefore **kept, relabelled as a
one-sided lower-bound check**: it detects losses and cannot confirm optimality. `missed == 1`
is a real miss; `missed == 0` means "not caught", not "optimal". My earlier call to void the
columns was wrong and threw away a working loss detector.

A length-4 miss **count** still needs the genuine 1,366,875-formula oracle.

---

# UPSTREAM REMEDY (a) — its cost, stated with it

Amends the superseded-framing section above, where remedy (a) was given without its price.

**(a) Do not prune on an unrefined aggregated estimate** — refine before `reduce_frontier`
acts, or restrict pruning to refined nodes. **This is not free, and the cost is the reason the
aggregated path exists.** Per the paper's Section C, sample-based computation costs on the
order of **|D|x** more arithmetic per estimate than the aggregated form. Refining before
pruning pays that on **every node**, not only on popped ones — which is precisely the expense
the aggregated estimate was introduced to avoid. At |D| ~ 2,000 that is a large constant.

**The remedy converts a soundness defect into a runtime cost.** That is a real trade and it
should be presented to upstream as one, not as a free correctness fix. Remedy (b), the
assertion, is cheap but only reports the violation; and per the partition-invariance result, a
compliant partition is not sufficient either.

---

# TERMINOLOGY — "beam" names two different mechanisms in this file

Fixed before item 5 runs, because item 5's immunity claim depends on which one is meant.

| | **`beam_optimal`** | **`_apply_beam_cap` / `MAX_FRONTIER_SIZE`** |
|---|---|---|
| whose code | **upstream**, `compositional/beam_optimal.py` | **ours**, `patches/0001-frontier-beam-fallback.patch` |
| what it is | the paper's beam: level-wise over **complete formulas ranked by exact IoU** (`search_utils.beam_search`, `PriorityQueue(beam_limit)`) | a cap on `optimal.py`'s A* frontier, **top-N by estimated ceiling** |
| shares `expand_node`? | **no** — expands via `search_utils.compute_next_search_space:102-129` | yes, it is `optimal.py` |
| uses the aggregated estimate as a **pruning bound**? | **NO** — ranks by exact IoU | **YES** — inherits every soundness property of `optimal.py` |
| failure mode | approximation only; always returns a formula | **returns NO formula at all** — 8/27 at width 5, 3/27 at 10, 2/27 at 25 |
| measured at L3, width 200 | **27/27 true-optimal, gap +0.00%** | 18/27 true-optimal |

**Everything in this file before the 2a/2b sections that says "beam" unqualified means the
frontier cap**, because that is what the project used throughout Phase A and Phase B. That
includes the D5-era passages quoted above about "beam vs exact", the §4.3 discussion, and the
beam-vs-exact band comparisons. Read them as *frontier cap*, not as the paper's beam.

**Consequence for item 5 (alpha=0.005, beam-only).** The immunity claim — that a beam-only run
is unaffected by the aggregated-estimate soundness defect — **holds only for `beam_optimal`**.
`beam_optimal` ranks by exact IoU and never prunes on the aggregated bound, so the defect
cannot reach it. The frontier cap prunes on exactly that bound and additionally has its own
no-solution failure mode. **Item 5 must run `beam_optimal`, and this must be stated in its
pre-registration, not assumed.**

---

# ITEM 3 — confound bounded, and unblocked

Item 3 (scale to ~50 units per arm, length 3, alpha=0.05, K=15) was blocked on the concern
that OR-of-same-category-then-narrowing might not be equally frequent across the trained and
untrained arms — a confound aligned with the experiment's independent variable. The magnitude
is now computable and was not computed at the time.

**Worst case, all losses falling on one arm:**

```
miss rate                2/27                        = 0.0741
mean loss magnitude      (+0.9274% + 4.7434%)/2      = 0.0284
worst-case arm shift     0.0741 x 0.0284             = 0.210%
trained/untrained gap    (3.41 - 2.07)/2.07          = 64.7%
ratio                                                  ~1/308
```

**A ~0.21% shift against a 64.7% effect. Roughly 1 part in 300.** The confound cannot move the
trained/untrained separation.

**Caveat, and it is a real one:** the loss magnitudes are **n = 2**. This is an
order-of-magnitude argument, not a bound. A third loss an order of magnitude larger than the
observed two would change the arithmetic, and nothing observed rules that out.

**Corroboration from a different direction:** the flat exposure splits **3 trained / 3
untrained** (trained a=0.2 unit88, a=0.1 unit396, a=0.05 unit86; untrained a=0.1 unit88,
a=0.1 unit413, a=0.05 unit413). The at-risk structure is not concentrated in either arm, which
is the specific thing the block was worried about.

**Item 3 moves above item 4 and below item 5.**

---

# L4 VERIFICATION — re-sequenced

The C2 revision makes the cheap half free, so it runs first.

**(a) One-sided check, no oracle.** Since `max_length` is a maximum, the length-4 space
contains the length-3 space, so `part_L4 < true_L3` is a **definite miss**. Report the
definite-miss count directly from `partition_L4.csv`. Costs nothing — the column is already
there.

**(b) Scope the length-4 oracle only after seeing (a).** If the free lower bound is already
high, the oracle is **optional**: its only remaining job is converting INCONCLUSIVE into
confirmed-optimal, which is the expensive half (1,366,875 formulas x 27 pairs). A high (a) may
make that confirmation not worth buying.

---

# COMPARISON PRECONDITIONS applied retroactively to P7, P8 and item 5

The six fields (`diary/summer_d6.md`) are adopted for every new pre-registration. Applied here
to the registrations already standing, so they are not exempt from their own protocol.
`partition_L4.csv` still does not exist, so P7's application is pre-result.

## P7 — frontier near-invariance at length 4

| field | |
|---|---|
| **Quantities** | both sides `peak_frontier`, a **node count**, dimensionless. Same kind. Not a time and not a formula-space size — the instance-3 error is excluded by construction. |
| **Membership** | the **matched set** (terminated in both runs), fixed by A1. All-27 median reported alongside but explicitly labelled non-verdict. |
| **Reference** | flat length-4 from `beam_vs_exact_K15.csv`: same K=15, same 27 pairs, **same length 4**. The partition differs — that is the treatment, not a mismatch. |
| **Discrimination** | four disjoint exhaustive bands (B1); `ratio >= 1.2` is a distinct outcome, not a residual. A halving, an enlargement, or near-invariance each give different verdicts. |
| **Power** | matched-set floor of **10** (B2); below it the verdict is UNDERPOWERED and neither hypothesis is called. |
| **Sentinels** | timed-out peaks are **truncated lower bounds**, excluded via the matched set; `halted != "no"` is bucketed and counted separately. |

**No gap found.**

## P8 — the fixed-P2 shape (5/5/0 vs 6/4/2)

| field | |
|---|---|
| **Quantities** | all six numbers are **counts of pairs** out of 27. Same kind on both rows. |
| **Membership** | the same 27 (arm, alpha, unit) pairs in both rows. Fixed. |
| **Reference** | flat exposure audit, same K, alpha set, unit set, length 3. Matches. |
| **Discrimination** | a per-sentence prefix **not** expanded before being dropped falsifies it and forces re-examination of the 0/27. |
| **Power** | **full census, not a sample** — all 27 measured. Power is not applicable rather than adequate. |
| **Sentinels** | **GAP FOUND — see below.** |

### The checklist found a real gap in P8, which is the first prospective evidence for it

**Sentinels: "P* never entered the frontier" is a third outcome and P8's three columns have
nowhere to put it.** It occurred on **1 of 27 flat** (trained a=0.1 unit92) and **3 of 27
per-sentence** (trained a=0.1 unit92, trained a=0.05 unit412, untrained a=0.1 unit510).

Those pairs have **no assigned ceiling at all**, so they are neither "inadmissible" nor
"admissible". Under the registered 3-column shape they would silently fall into the
*admissible* residual — inflating the apparent improvement, since the per-sentence run has
three of them against the flat run's one.

**P8 is amended before scoring, and the amendment is recorded as an amendment:** the shape is
**four columns**, not three —

```
(inadmissible ceiling) / (expanded before any copy dropped) / (optima lost) / (P* never entered)

flat          6 / 4 / 2 / 1
per-sentence  5 / 5 / 0 / 3      <- P8, amended
```

This is the checklist catching something on a registration that had already passed review by
both of us. It is weak evidence — one instance — but it is **prospective**, which the 7/7
retro-validation is not.

## Item 5 — alpha=0.005, M ~ 80,000, beam-only

Registered now, before the run is scheduled.

| field | |
|---|---|
| **Quantities** | IoU and lift, both ratios. State explicitly whether lift is IoU/density or the ratio-of-averages form — the two differ and both appear in this file. |
| **Membership** | **HIGHEST RISK.** At alpha=0.005 the `min_fire` floor changes which units are eligible — this is the recurring-error-shape trap documented at the top of this file, where the trained a=0.05 unit set went disjoint from every other alpha. **Unit ids must be pinned with `--unit_ids` and the eligible-set size reported.** |
| **Reference** | the paper's reported setting. Ours is K=15 against the paper's 25/847/1198 — **the K caveat is still unwritten and must land before item 5 is read**, not after. |
| **Discrimination** | state in advance what result would count against the paper's setting reproducing, distinct from "the corpus build failed". |
| **Power** | at alpha=0.005 the flat corpus had **7 of 512 trained units excluded** by `min_fire`; at M ~ 80,000 that changes. Report eligible-set size before sampling. |
| **Sentinels** | **`beam_optimal` never returns `None`** — verified, C1, 0 of 162 runs. The frontier cap returns no formula on 8/27 at width 5. **This is why item 5 must run `beam_optimal`**, and the immunity claim is void if the frontier cap is used instead. |

---

# THE EXPOSURE METRIC'S SUBJECT WAS UNDER-SPECIFIED — recomputed over the optimal SET

2026-08-01. Triggered by the P8 sentinel gap: 3 per-sentence pairs and 1 flat pair had
"P* never entered the frontier" while losing nothing.

## Item 1 — what those pairs did: possibility (a), confirmed

All four returned the **true max**, by a formula with a **different prefix** than the one the
audit was tracking:

```
flat  trained a=0.1  unit92    0.159328 = true   (dep=pobj OR lemma=.)
part  trained a=0.1  unit92    0.159328 = true   (dep=pobj OR lemma=.)
part  trained a=0.05 unit412   0.098051 = true   ((tag=IN AND (NOT const=VP)) OR lemma=.)
part  untrained a=0.1 unit510  0.154405 = true   ((dep=pobj AND (NOT const=VP)) OR lemma=.)
```

Not (b) — the detection was correct: that particular prefix genuinely never entered. Not (c) —
they were in the loss count and correctly scored 0. **The metric was tracking one arbitrary
member of a set.**

## Item 2 — the optimum is a unique mask reached by up to 3 prefixes

From the existing enumeration, no new search:

```
distinct OPTIMAL formulas (masks) per pair : min 1   median 1   max 1     <- unique on all 27
distinct OPTIMAL PREFIXES  per pair        : min 1   median 2   max 3
pairs with a SINGLE optimal prefix         : 9/27
pairs whose optimum is reachable at length <= 2 : 1/27
```

The optimal **mask** is unique on every pair. What varies is how many distinct length-2
prefixes construct it — the same final mask is reachable by different orderings within the
left-deep grammar.

### Redefined: EXPOSED if any optimal prefix is inadmissible; LOST if all are pruned

```
                         EXPOSED        LOST (bound)     ACTUAL misses
flat one-sample          14/27            4/27              2/27
per-sentence             13/27            4/27              0/27

old figures (one arbitrary representative prefix):  6/27 and 5/27
```

**The old 6/27 and 5/27 are relabelled: computed over one arbitrary representative prefix.**
The set-based exposure is **more than twice as high** — 14/27 flat — because with 1-3 prefixes
per optimum, at least one being inadmissible is common.

**LOST remains a BOUND, not a measurement.** 4/27 both ways against 2 and 0 actual, because a
prefix can be pruned *after* it has already been expanded — the pop-ordering effect. The
measuring version is still the per-node event-ordering question (was the prefix expanded
before any copy was dropped), which is the held item-4 code change.

### "Escaped by luck" is now a measured corpus property

**Both actual flat misses have a single-prefix optimum** (trained a=0.2 unit88, trained a=0.05
unit86, both `n_prefixes = 1`). The two pairs the bound flags but that did not lose
(trained a=0.1 unit396, untrained a=0.1 unit413) both have `n_prefixes = 2`.

So the phrase retires: **"escaped by luck" was "the optimum had more than one prefix".** It is
a property of the corpus and the grammar, measurable in advance, and **9 of 27 pairs do not
have it** — those are the vulnerable ones.

## Item 4 — the loss happens one level shallower than the metric looks

A prefix that **never entered the frontier** means **an ancestor of it was pruned**. The
metric was asking about level 2 while the bite had already happened at level 1.

The never-entered count moved **1/27 flat -> 3/27 per-sentence**. Under this reading, **the
partition pushed the bite up a level**: fewer level-2 prefixes carry an inadmissible ceiling
(14 -> 13 exposed), but more optimum-carrying prefixes never get created at all because
something shallower was cut first.

**The general form of the exposure question is: "was any ancestor of any optimal formula
pruned?"** — not "was P pruned". Every exposure figure in this file, including the 14/27 above,
answers the narrower question and is therefore still a lower bound on true exposure.

---

# HEADLINE — losses occur exactly where the optimum has ONE prefix

**Mechanism, stated as the exposure section's headline.**

```
losses at n_prefixes = 1 : 2 of 2   (trained a=0.2 unit88, trained a=0.05 unit86)
bound-flagged non-losses : 2 of 2 have n_prefixes = 2
vulnerable population    : 9 of 27 pairs have n_prefixes = 1
```

**The rate is 2/9 = 22% among vulnerable pairs, not 2/27 = 7%.** Every pair with a redundant
prefix survived; every loss was a pair with none. `n_prefixes` is the redundancy the search has
against an unsound prune, and where it is 1 there is none.

## Power check on P3, and why it is the second reason not to read P3 as a fix

`n_prefixes` is **partition-invariant**: the optimal mask is unique on all 27 pairs, masks are
partition-invariant, and the prefix set is derived from masks. **The same 9 pairs are
vulnerable in both runs.**

So the per-sentence result of 0/27 is really **0 of 9**. Under the flat rate of 2/9:

```
P(0 losses in 9 vulnerable pairs) = (1 - 2/9)^9 = (7/9)^9 = 0.1042
```

**p ~ 0.10. P3's improvement is not distinguishable from chance at n = 9.**

Caveats, both real: the base rate is estimated from **n = 2**, and the two runs are **not
independent** — same corpus, same concepts, same units, same optimal masks.

**This is the second independent reason not to read P3 as a fix, and the two should be stated
together:**

1. **Mechanistic** — both miss-prefix ceilings are **byte-identical** across partitions
   (`0.232677`, `0.203398`) and both prefixes are still dropped. The bound did not change.
2. **Statistical** — 0/9 against a 2/9 base rate has p ~ 0.10. The improvement is within
   chance.

**P3 was registered and reviewed by both of us without a Power field.** The checklist
post-dates it. Applied retroactively, Power is the field that fires, and it changes the
reading from "both misses recovered" to "no loss was observed in 9 vulnerable pairs, which a
null model produces one time in ten".

## Shape hypothesis — TESTED, NOT SUPPORTED, and replaced by the rule underneath it

Hypothesis: *a trailing AND or AND-NOT forces `n_prefixes = 1`; all-OR admits up to 3.*

```
trailing AND / AND-NOT : n_prefixes [1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3]   all == 1?  NO
all-OR (OR+OR)         : n_prefixes [3]                                        any > 1?  yes
```

**Not supported.** `ANDNOT+ANDNOT` (n = 9 pairs) gives `n_prefixes = 2` every time, not 1.

**But `n_prefixes` IS a deterministic function of the operator signature** — within every
signature group it is constant across all 27 pairs:

```
OR+OR                        -> 3      (n=1)
ANDNOT+ANDNOT                -> 2      (n=9)
ANDNOT+OR / AND+OR /
OR+AND / OR+ANDNOT           -> 1      (n=9)      <- THE VULNERABLE CLASS
multi-signature optima       -> 2 or 3 (n=8)
```

The governing rule is **operator repetition, not trailing operator**: identical adjacent
operators commute and create redundant construction orders.
`a OR b OR c` is symmetric in three terms -> 3 prefixes. `a AND NOT b AND NOT c` has two
commuting negated terms -> 2 prefixes. A **mixed** signature has no commuting pair -> **1
prefix, and no redundancy**.

**So the vulnerable class IS identifiable from the formula shape with no search** — just by a
different rule than the one proposed. An optimum whose two operators differ has exactly one
construction path, and a single unsound prune anywhere on it loses the optimum.

## Ancestor question CLOSED at length 3, and the answer is "almost always"

Level-1 prefixes are single leaves, all present at initialisation and exposed to
`reduce_frontier` at Algorithm 1 line 11. Checking whether any leaf ancestor of an optimal
formula was pruned there:

```
pairs with >= 1 optimal leaf ancestor pruned : 24 of 27
total optimal leaf ancestors pruned          : 31
```

**24 of 27.** So "was any ancestor of any optimal formula pruned?" answers **yes almost
everywhere**, while only 2 pairs lose the optimum. This closes the question at length 3 and
settles what it is worth: **"was it pruned" is a near-vacuous predicate.** Everything hinges on
*when* the prune happens relative to the expansion — the pop-ordering result — which is why
the held event-ordering metric is the only version that measures rather than bounds.

---

# P9 — registered from length-3 data, before partition_L4.csv exists

```
P9  The n_prefixes distribution at length 4 shifts UPWARD for all-OR signatures and stays at
    1 for signatures with a trailing narrowing operator.
```

Registered with the correction above already applied to its reading: the governing variable is
**operator repetition**, so the sharpened form is *repeated-operator signatures gain prefixes
with length (OR+OR+OR -> up to 6; ANDNOT x3 -> up to 6), mixed signatures stay at 1*.

**Consequence stated in advance, both directions:**

- If the **mixed-signature share rises** with length, the vulnerable population grows and
  **the bug concentrates with length**. Length 4 is then worse than length 3, and every
  length-4 result in this repo is more exposed than the length-3 ones.
- If the **all-OR / repeated-operator share rises**, redundancy grows and **the bug dilutes**.

Comparison preconditions: *Quantities* — prefix counts both sides. *Membership* — same 27
pairs. *Reference* — length-3 distribution above, same K/alpha/units. *Discrimination* — the
two directions give opposite conclusions. *Power* — full census, no sampling. *Sentinels* —
pairs whose length-4 optimum is unreachable (timeout) are bucketed, not counted as either.

---

# UPSTREAM — a diagnostic users can run with no code change

Added to the report. It requires no patch and no access to internals:

> **Count the prefixes that reach your optimum. If `n = 1`, you have no redundancy against an
> unsound prune.**
>
> Equivalently, and computable by inspection of the returned formula: **if its two operators
> differ, there is exactly one construction path to it.** Repeated operators commute and give
> you 2-3 alternative paths; mixed operators give you one.
>
> Measured here: 9 of 27 units had a single-prefix optimum, and **both observed losses were in
> that group — 2 of 9, 22%**, against 0 of 18 among units with a redundant prefix.

This is offered alongside the two remedies, not instead of them. It tells a user which of
their results are at risk; it does not make the search sound.

---

# STRUCTURAL DERIVATION of n_prefixes, and why the grammar's danger does not scale

The empirical table above is a consequence, not the result. The result is structural.

**`n_prefixes` = the number of leaves that could have been added LAST.** In the left-deep
grammar every formula is `((a op1 b) op2 c)`, and a distinct prefix exists for each leaf that
could occupy the final position without changing the mask. That requires the operator applied
to it to be **interchangeable with the trailing operator** — i.e. **identical adjacent
operators**.

```
OR + OR         a | b | c        all three leaves may be last          -> 3
ANDNOT + ANDNOT a & ~b & ~c      only the NEGATED leaves may be last   -> 2
mixed           (a op1 b) op2 c  only c may be last                    -> 1
```

**AND-NOT gives run-length rather than run-length + 1 because the positive base cannot move.**
The grammar never negates the leftmost term (`expand_node` builds `And(label, Not(leaf))`, so
the base of the chain is always positive), so `a` is pinned in `a & ~b & ~c` while `b` and `c`
commute freely. An OR-run of length `r` yields `r + 1` prefixes; an AND-NOT run of length `r`
yields `r`.

## The vulnerable share of SIGNATURE SPACE is 2/3 at both lengths

A signature is protected iff its **trailing operator run has length >= 2** — that is, the last
two operators are identical.

```
length 3   3^2 = 9 signatures    protected (op1 == op2)      3      vulnerable  6/9  = 0.667
length 4   3^3 = 27 signatures   protected (op2 == op3)      9      vulnerable 18/27 = 0.667
```

**Constant at 2/3. The grammar's danger does not scale with length.** Adding a level multiplies
both the signature space and the protected subspace by 3, so the ratio is fixed. Whatever
changes with length, it is not the share of the *space* that is vulnerable — it can only be the
share of *realised optima* that land in it.

## Observed optima vs signature space — NO RESOLVABLE DIFFERENCE (see the retraction below)

```
signature space, vulnerable share       : 67%
observed length-3 optima, mixed share   : 9 mixed of 19 two-operator optima = 47%
```

**RETRACTED — see "Instance 13" below.** The claim made here was that real optima land in
protected signatures *more often than chance* — 47.4% vulnerable against 67% of the space —
and that repeated-operator formulas are what these neurons select for.

**That difference is not resolvable at n = 19.** The Wilson 95% CI on 9/19 is
**[27.3%, 68.3%]**, which **contains** the 66.7% space share. The observed gap of 19.3 points
is **1.69 SE** (SE = 11.5 points), against the ~2.8 SE needed for 80% power.

What survives: the observed mixed share is 47.4% (optima) and 57.7% (returned formulas), and
the signature-space share is 66.7%. **No claim is made that they differ.**

---

# P9 RESTATED against the fixed baseline — registered before partition_L4.csv exists

Since the signature-space share is constant at 67%, the only thing length can change is where
optima land within it. P9 is therefore restated on **the mixed share of length-4 OPTIMA**,
computable from the length-4 optima alone with no extra instrumentation.

```
P9   mixed (single-prefix) share of length-4 optima:

     ~47%   -> vulnerability is LENGTH-INVARIANT. The length-3 reading transfers.
     -> 67% -> [BAND WITHDRAWN with P9; retained only to show what was registered]
     < 47%  -> longer formulas BUY REDUNDANCY: more room for repeated-operator runs.
```

The baseline is fixed at **47%** (9 of 19) and **67%** (space), both computed from length-3
data already committed, before any length-4 optimum exists.

Comparison preconditions: *Quantities* — a share of optima on both sides, dimensionless.
*Membership* — the same 27 pairs; two-operator optima only at L3, three-operator at L4, and the
denominator is stated with each figure. *Reference* — the 47%/67% pair above, from committed
L3 data. *Discrimination* — the three bands give opposite conclusions about length-4 exposure.
*Power* — full census, no sampling. *Sentinels* — pairs whose L4 optimum is unavailable
(timeout, or the C2 one-sided limitation) are bucketed and excluded from the denominator, and
the excluded count is reported.

---

# QUEUE — item 4 (event-ordering metric) promoted to FIRST after L4

Ahead of D/E/F and the fork-only rerun. **Rationale, recorded so this does not read as
cleanup:**

Three exposure measures now exist, at three levels of the search:

```
set-based inadmissible ceilings   14/27      (level 2, any optimal prefix)
all-optimal-prefixes-pruned bound  4/27      (level 2, all prefixes)
optimal leaf ancestors pruned     31 / 24-of-27 pairs   (level 1)
```

**All three count OPPORTUNITIES, and all three have proven nearly uncorrelated with outcome**
— against 2 actual losses flat and 0 per-sentence. 24 of 27 pairs had an optimal leaf ancestor
pruned and lost nothing. The bound says 4 where the truth is 2 and 0.

**Item 4 is the only harm measurement in the section.** Everything else bounds. Until the
event-ordering metric exists — was the prefix expanded before any copy of it was dropped —
there is no measured quantity for "the prune actually cost us the optimum", only upper bounds
that overstate by 2x to 12x. That is not a tidying task; it is the section's missing dependent
variable.

---

# UPSTREAM DIAGNOSTIC — strengthened to inspection-only

Replaces the earlier "count the prefixes" wording, which implied enumeration.

> **`n_prefixes` is a function of the operator signature alone.** It is readable off a
> published formula **by inspection** — no enumeration, no instrumentation, no access to the
> search internals.
>
> **If the formula's two operators differ, there is exactly one construction path to it, and
> no redundancy against an unsound prune.** Identical adjacent operators commute and give 2-3
> alternative paths; an OR-run of length `r` gives `r+1`, an AND-NOT run gives `r` (the
> positive base is pinned).
>
> **Support:** 2 of 9 single-prefix units lost their optimum (22%); **0 of 18** units with a
> redundant prefix did.
>
> The vulnerable share of signature space is **2/3 at every length**, so this does not become
> less relevant for longer formulas.

Offered alongside the two remedies, not instead of them: it tells a user which of their
published results are at risk, using only what is already printed in their results table.

---

# ONE DESIGN CHOICE, TWO EFFECTS — linking the grammar restriction to n_prefixes

`expand_node` grows a formula by exactly three moves, and the negation move is
`And(label, Not(leaf))`. That single choice produces both effects recorded separately in this
repo:

**Effect 1 — expressiveness (already in the README and the grammar section).** Negation appears
only as AND-NOT, so `OR NOT` is unconstructible and the leftmost term is never negated. At
length 3 this costs nothing measurable; **at length 4 it costs +0.1586% IoU on untrained
unit92**, where the unrestricted optimum
`(tag=NN AND (dep=ROOT OR NOT const=VP)) OR dep=punct` is out of grammar.

**Effect 2 — redundancy (this section).** Because the base of an AND-NOT chain is always
positive, `a & ~b & ~c` has a **pinned** leftmost term: only `b` and `c` commute. An AND-NOT
run of length `r` therefore yields `r` prefixes rather than `r + 1`. The same pinning that
excludes formulas also **removes one construction path** from the formulas it does allow.

So the design choice **narrows the search space and simultaneously reduces the redundancy of
what remains in it** — the two effects point the same way, and both were measured
independently before the link was noticed.

---

# WHAT REDUNDANCY ACTUALLY IS — and why it makes item 4 the dependent variable

From the numbers already committed, with no new measurement:

```
pairs where EVERY optimal prefix was pruned : 4/27
pairs that actually lost the optimum        : 2/27
the 2 LOST  : n_prefixes = 1
the 2 SAVED : n_prefixes = 2
```

**All four had every optimal prefix pruned. Two survived anyway.**

So `n_prefixes` is **not protection from pruning** — pruning happened in all four cases. It is
**the number of independent chances to be expanded before a prune fires.** A pair with two
prefixes gets two draws on the pop order; a pair with one gets one.

**Redundancy and ordering are the same mechanism counted two ways.** `n_prefixes` counts the
draws; pop order decides each draw. That is why every opportunity-counting measure in this
section has come apart from outcome — they count prunes, and a prune only costs something when
it beats the expansion to the node.

**This is the motivation for item 4.** The event-ordering metric — *was the prefix expanded
before any copy of it was dropped* — is not an alternative framing of the exposure counts. It
is the only one that measures the thing `n_prefixes` gives you draws on.

---

# C2 ONE-SIDED CHECK applied to the FLAT length-4 baseline

Run now, so the two partitions are scored against the same reference by the same predicate
when `partition_L4.csv` lands.

```
FLAT one-sample L4    DEFINITE MISS 0    INCONCLUSIVE 23    SENTINEL 4    total 27
PER-SENTENCE L4       pending; the same function is applied on arrival
```

**Zero definite misses in the flat length-4 run.** That is a genuine result and a weak one:
the check is one-sided, so 23 INCONCLUSIVE means "not caught", not "optimal". The four
sentinels are the timeouts, bucketed rather than scored.

### Correction to a committed baseline note

The four flat length-4 timeouts are **trained a=0.1 unit510, trained a=0.05 unit87, untrained
a=0.05 unit396, untrained a=0.05 unit510**. The P5 registration in
`src/exp_partition_L4_score.py` named the third as `a=0.2`; it is `a=0.05`. **The count of 4 —
which is what `FLAT_TIMEOUTS` and P5(a) depend on — is unchanged**, so no threshold moves. The
docstring is corrected in place with the correction marked.

### Instance 10 — the two sources are not read by identical code, and this is logged, not fixed silently

The **scoring predicate is shared** — one `definite_miss()` function, one comparison, one
tolerance, one sentinel bucket. But the **IoU extraction differs by necessity**: the flat CSV
stores integer counts (`exact_n_inter`, `exact_n_fires`, `n_fire_neuron`) and full precision is
reconstructed from them; `partition_L4.csv` stores the value directly as `repr(float)`.

Both are full precision, and no discrepancy was produced here. It is logged anyway because it
is **structurally the same risk that already bit once**: the first pass of the 27-pair oracle
compared against the *rounded* `exact_IoU` column and reported 17/27 misses instead of 2/27.
Same file, same column family, same class of error.

Logged as **D6 instance 10, field = Reference** — *does the reference's configuration match
the run being scored?* Two sources reached by two extraction paths is a Reference risk even
when the values agree.

---

# D6 SCOPE BOUNDARY — drafted before the write-up

**In D6:**

- the soundness finding (aggregated estimate inadmissible when `Bott_1(E^C)_x != 0`)
- partition-invariance of the aggregated estimate, and that repartitioning is not a fix
- the measurement-defect class, the six-field checklist, and its retro/prospective validation
- the `n_prefixes` mechanism, its structural derivation, and the 2/3 signature-space share
- the upstream diagnostic and the two remedies with their costs
- item 4's harm measurement, once it exists

**Deferred to D7:**

- the length-4 oracle (1,366,875 formulas x 27 pairs)
- the fork-only rerun at `utils/optimal_utils.py:271`
- queue items D / E / F
- queue items 3 (50 units/arm), 4 (fixed work), 5 (alpha=0.005, beam-only)

**Rationale: none of the deferred items changes what D6 concludes.** The length-4 oracle
converts INCONCLUSIVE to confirmed-optimal — it can add misses but cannot unmake the
inadmissibility, which is proved from the ceilings and `Bott_1` directly. The fork-only rerun
replaces F2/F4, which are already recorded as uninterpretable and are load-bearing for nothing.
D/E/F and items 3/4/5 are measurements *using* the method, downstream of whether it is sound.

The one item that could change D6 is **P9** — if the mixed share of length-4 optima rises
toward 67%, the exposure claim strengthens with length rather than holding constant. P9 is
computable from the L4 optima alone and stays **in D6**.

---

# POWER OF THE ONE-SIDED CHECK — 0/23 is genuine for large misses, weak for small

The C2 check fires only when `R_L4 < true_L3`. Its detection threshold per pair is the
**margin** `m = (R_L4 - true_L3) / R_L4`: a length-4 loss is invisible unless it exceeds `m`.

```
n = 23 scoreable pairs (4 timeouts excluded)
margin m :  min 0.101%    median 2.469%    max 11.709%
```

Against the two observed length-3 miss magnitudes:

```
pairs where m > 4.74% (would miss the LARGER known magnitude) :  7/23
pairs where m > 0.93% (would miss the SMALLER known magnitude): 18/23
```

**Verdict, stratified rather than pooled, because pooling hides the split:**

- **A 4.74%-scale loss would be caught on 16 of 23 pairs.** Against losses of that magnitude,
  **0 DEFINITE MISS is a genuine result.** The median margin of 2.47% is "a few percent", the
  condition registered in advance for calling it genuine.
- **A 0.93%-scale loss would be caught on only 5 of 23.** Against small losses the check is
  **near-blind**, and 0/23 carries almost no information.

The two known magnitudes straddle the median margin, so the honest statement is: **the flat
length-4 run has no large in-grammar loss, and the check cannot speak to small ones.** It must
not be reported as "no misses at length 4".

The same computation is applied to `partition_L4.csv` on arrival, through the same predicate.

---

# P9 — SUBJECT CORRECTED. As registered it was circular.

P9 was registered to score "the mixed share of length-4 **optima**". **The run does not
produce optima.** It produces *returned formulas*, and whether returned equals optimum is the
open question the whole section is about. Scoring a vulnerability measure on a quantity whose
correctness is what the vulnerability determines is circular.

**Restated:**

```
P9  the mixed (single-prefix) share of RETURNED length-4 formulas,
    reported against BOTH baselines:
        47%  observed length-3 OPTIMA (9 of 19 two-operator formulas)
        67%  signature space (18 of 27)
```

**The bias, named explicitly rather than caveated:** the pairs where returned != optimum are
**exactly the mixed-signature vulnerable ones** — that is the finding this measure rests on. So
the measurement is **least reliable precisely where it matters**. Worse, the direction of the
bias is **unknown**: it depends on the signature of whatever was returned *instead* of the
optimum, which nothing here measures. A lost mixed-signature optimum could be replaced by a
returned formula that is itself mixed (bias -> 0), or repeated-operator (bias downward,
understating vulnerability), and there is no reason to expect either.

**Status:** the returned-formula version stays in **D6 as suggestive, not as a verdict.** The
true-optima version **requires the length-4 oracle** (1,366,875 formulas x 27 pairs) and is
therefore a **D7 item**. This is the one place the D6/D7 boundary drawn above moves: P9 was
listed as staying in D6, and only its weakened form does.

The comparison against the 47% baseline inherits the same defect from the other side — 47% was
computed on length-3 **optima**, verified by enumeration. Comparing returned-L4 against
optimal-L3 is not like-for-like, and the figure is reported with both denominators and that
mismatch stated.

---

## D6 instance 11 — the P9 subject substitution

**Field: Quantities, primarily — with Reference as a second, independent hit.**

My read differs slightly from "Reference alone", and the difference is worth recording rather
than smoothing:

- **Quantities** — *what is on each side, are they the same kind of thing?* The named subject
  was `optima`; the scored subject was `returned formulas`. These are different kinds, and the
  substitution is the whole error. This is the primary hit.
- **Reference** — *does the reference's configuration match the run being scored?* The 47%
  baseline was computed on length-3 **optima**. Even after restating P9 on returned formulas,
  comparing returned-L4 against optimal-L3 remains a configuration mismatch. This fires
  independently of the first.

**No seventh field is needed.** The instance is caught, twice, by existing fields. Recorded
with the disagreement visible because "which field fires" is itself a claim, and forcing it to
one field would have hidden that the checklist caught this redundantly — which is mildly
stronger evidence than a single hit.

---

# TWO CORRECTIONS TO THE LENGTH-4 REPORTING

## 1. Membership fires on the flat one-sided result too

The flat length-4 statement is not "no large in-grammar loss". It is:

> **No large in-grammar loss AMONG THE 23 PAIRS THAT TERMINATED.** Four pairs are excluded as
> timeouts — trained a=0.1 unit510, trained a=0.05 unit87, untrained a=0.05 unit396, untrained
> a=0.05 unit510 — **and those are precisely the pairs where the search struggled most.**

**Two checklist fields fire on that result, not one:**

- **Power** — median margin 2.469%; a 0.93%-scale loss is caught on only 5 of 23.
- **Membership** — the 23 are the pairs that terminated, a set selected by the same difficulty
  that would make a loss more likely. The excluded four are not missing at random.

Both must be attached wherever the 0/23 figure appears.

## 2. P9's baseline mismatch REMOVED, not stated

The 47% baseline was computed over length-3 **optima**; P9 now scores **returned** formulas.
Rather than caveat the mismatch, the baseline is recomputed on the same subject. Returned
equals optimal on 25 of 27 pairs, so the recomputation is exact and can differ on at most two.

```
RETURNED length-3 formulas, signature counts:
   ANDNOT+OR     10      ANDNOT+ANDNOT  9      AND+OR   2      OR+ANDNOT  2
   AND+AND        1      ANDNOT+AND     1      OR+OR    1      (2-leaf)   1

two-operator returned formulas : 26/27
mixed (n_prefixes = 1)         : 15/26 = 57.7%      <- P9's baseline
```

**The like-for-like baseline is 57.7%, not 47.4%** — a 10-point difference, and material to
P9's bands. The optima-based 47.4% and the signature-space 66.7% are retained **as reference
points only**, not as the comparison.

The gap between 47.4% and 57.7% is itself informative: the optima figure excluded the 8
multi-signature pairs from its denominator, while a returned formula is always a single
formula with a single signature. Same corpus, same runs, two defensible denominators, ten
points apart — a Membership effect in miniature.

**P9 restated a second time:**

```
P9  mixed (single-prefix) share of RETURNED length-4 formulas, against 57.7% returned-L3:
      ~57.7%  -> vulnerability is length-invariant
      rising  -> the bug concentrates with length
      falling -> longer formulas buy redundancy
    Reference points, not the comparison: 47.4% (L3 optima), 66.7% (signature space).
```

---

# P9 — VERDICT WITHDRAWN. Not scoreable at n = 27.

Amendment registered before `partition_L4.csv` exists. **Reason for the amendment: the
baseline changed from 47.4% to 57.7% when the subject was corrected to returned formulas, so
the bands registered against 47.4% are invalid.** Rather than restate them, the verdict is
withdrawn — because at this `n` no band can be resolved.

**Arithmetic, verified independently before acting on it:**

```
p  = 15/26 = 0.5769                     (returned-L3 mixed share)
SE = sqrt(0.5769 * 0.4231 / 26) = 0.09689  ->  9.7 points
gap to the 66.7% signature-space share  ->  9.0 points
n for 80% power = (1.96 + 0.84)^2 * 0.2441 / 0.0901^2 = 235.8  ->  ~236
```

**The standard error is larger than the effect.** 9.7 points of noise against a 9.0-point gap.
Separating the two reference points at 80% power needs **n ~ 236**; we have **27**.

A stronger version of the same point, computed while checking: the **Wilson 95% CI for the
baseline itself** — 15 of 26 — is **[38.9%, 74.5%]**, which **contains** the 66.7% space share.
The baseline cannot be distinguished from the space share at this `n`, let alone the length-4
value from the baseline.

**Replacement, which is what P9 becomes:**

> Report the length-4 returned-formula mixed share as a **point estimate with a 95% Wilson
> confidence interval**, against both reference points (57.7% returned-L3, 66.7% signature
> space). **State explicitly that no directional conclusion is available at this n.** Do not
> report "rising", "falling", or "invariant".

The powered version moves to **D7**, alongside the length-4 oracle and item 3's 50-units-per-arm
scaling — which is the natural home, since 50 per arm across two arms is the only queued run
that approaches the required `n`.

## D6 instance 12 — a prediction registered without checking whether it could fire

**Field: Power.** Same shape as instance 2, one level up: instance 2 was a *check* that
sampled too few cases to fire (14.5%); instance 12 is a *prediction* whose discriminating
comparison was never tested for resolvability at the available `n`.

P9 went through **three registrations** — original, subject-corrected, baseline-corrected —
and each was reviewed by both participants. **None asked whether 27 pairs could resolve a
9-point difference.** The Discrimination field was satisfied at every stage (the bands gave
opposite conclusions), which is exactly what made the gap invisible: a prediction can be
perfectly discriminating in principle and unfireable in practice, and only Power distinguishes
those.

---

# D6 / D7 BOUNDARY — updated

**P9 leaves D6 entirely.**

D6's length-scaling statement becomes:

> **The length-scaling of vulnerability is not measurable at n = 27.** Separating the observed
> mixed share from the signature-space share requires n ~ 236; the 95% CI on the baseline
> itself spans [38.9%, 74.5%]. Deferred to D7.

**That is a result, not a gap.** It bounds what the corpus can answer, and it is the reason
item 3's 50-units-per-arm scaling matters beyond the trained/untrained question it was queued
for.

**D6 retains:** soundness; partition-invariance; the measurement-defect class and checklist;
the `n_prefixes` mechanism and the 2/3 signature share; the upstream diagnostic and remedies;
item 4's harm measurement `[PENDING]`.

**D7 takes:** the length-4 oracle; the fork-only rerun at `optimal_utils.py:271`; queue items
D/E/F; queue items 3, 4, 5; **and the powered P9.**

---

# INSTANCE 13 — the parent of instance 12, and it is mine

**Field: Power. Attributed to the assistant.**

The claim *"optima are biased toward protected signatures"* — 47.4% observed against 66.7% of
signature space — was asserted without checking whether that difference is resolvable at the
available `n`.

**Verified:**

```
optima-based mixed share  9/19 = 47.4%   Wilson 95% CI [27.3%, 68.3%]   contains 66.7%: YES
observed gap to space share                                            19.3 points
SE at n = 19                                                           11.5 points
gap in SE units                                                        1.69   (need ~2.8 for 80% power)
```

**The interval contains the value it was claimed to differ from.** No bias is established.

## It is the parent of instance 12, and the lineage is the point

P9's entire framing descended from this claim. The registered bands — *"~47% -> length-invariant,
-> 67% -> the bug concentrates"* — presupposed that 47.4% and 66.7% were **distinguishable
quantities between which a length-4 value could move**. They are not distinguishable at this
`n`, so the bands were measuring movement across a difference that was never established.

**The unsupported claim propagated into a registered prediction and survived three
registrations and three joint reviews.** Instance 12 recorded that none of those reviews asked
whether 27 pairs could resolve 9 points. Instance 13 is why that question was never reached:
the underlying 19-point difference had already been accepted as real, so the smaller derived
gap inherited its unexamined status.

**Recorded as a lineage, not two independent entries.** A defect in an asserted premise
becomes invisible once predictions are built on it — the predictions get scrutinised, the
premise does not, because it is no longer the thing being tested.

---

# THE POWER FIELD — restated to cover any comparison

The original wording — *"if the comparison samples, what is the probability it fires when the
defect is present?"* — applies only to sampled checks. Instances 12 and 13 are not sampled
checks; they are point estimates compared against reference values. The field did not, as
written, reach them.

**Restated:**

> **Power** — *What is the smallest effect this comparison can resolve at the available `n`,
> and is the predicted effect larger than it?*

**Verified to fire on all three Power instances under the new wording:**

```
instance  2   oracle drew 3 of 27 cases; smallest resolvable effect is "a drawn pair misses";
              P(fire | 2 bad pairs) = 1 - C(25,2)/C(27,2) = 14.5%          FIRES
instance 12   P9: SE 9.7 points at n = 26 against a predicted 9.0-point gap; n ~ 236 needed
                                                                            FIRES
instance 13   bias claim: SE 11.5 points at n = 19 against a 19.3-point gap = 1.69 SE;
              Wilson CI [27.3%, 68.3%] contains the compared value          FIRES
```

Instance 2 still fires under the sampled reading; 12 and 13 fire only under the restated one.

**Added to the checklist preamble:**

> **A prediction can be perfectly discriminating in principle and unfireable in practice. Only
> Power separates those.** Discrimination asks whether different outcomes give different
> conclusions; Power asks whether the data can tell those outcomes apart. Satisfying
> Discrimination is what makes a Power failure invisible.

---

# REMAINING D6 WORK — no new threads

1. **The write-up** — `diary/summer_d6_DRAFT.md`, numbers and structure complete.
2. **Item 4**, the event-ordering harm measurement, on `partition_L4.csv`'s arrival.
3. **The advisor message** — `results/UPSTREAM_REPORT.md` and
   `results/UPSTREAM_MESSAGE_DRAFT.md`, written, unsent, push held.

Nothing else is opened.

---

# LENGTH-4 PER-SENTENCE PARTITION — RESULTS

2026-08-02. `results/partition_L4.csv`, `results/partition_L4_VERDICTS.txt`. All 27 pairs
completed, so **C1's stop rule did not fire** and medians are claimable.

## V0 CONTROL — PASS, and it says more than it did at length 3

```
comparable pairs (terminated in BOTH) : 23
  same IoU : 23        HIGHER : 0        LOWER : 0
P6 bucket (timeout on one side)        : 4
```

`LOWER = 0`, so the repartition is correct and everything below is readable.

**But `HIGHER = 0` is itself a result.** At length 3 the partition produced 2 HIGHER — the two
recovered misses. **At length 4 not a single pair's IoU changed.** The partition altered no
answer at this length.

## P5 — SUPPORTED via (a) only, and the two disjuncts point in opposite directions

```
(a) timeouts/caps  4 -> 1  (only trained a=0.05 unit87 still halts)      YES
(b) matched median peak  15,593 -> 21,687  = 1.391x                       NO  (needed < 7,796)
```

## P7 — NOT SUPPORTED. B1 band: THE FRONTIER ENLARGED

```
matched n = 23
ratio-of-medians   21,687 / 15,593 = 1.391x
median-of-ratios                     1.222x
per-pair ratio     min 1.068   median 1.222   max 1.704
```

**Both statistics exceed 1.2, so the band verdict is robust to the choice** — recorded
explicitly because ratio-of-medians vs median-of-ratios has already diverged once in this file
(the K15/K8 discount, 54.2x vs 36.6x). Here they agree on the band and differ on the magnitude,
so the band is claimed and the magnitude is reported as a range.

**B1 verdict: `ratio >= 1.2` -> THE PARTITION ENLARGED THE FRONTIER.** Neither P5(b) nor P7 is
claimed. The frontier is **22% to 39% larger** under the correct partition, on **every one of
the 23 matched pairs** (minimum ratio 1.068 — not a single pair shrank).

## A2's separation gave a case I did not register

```
matched median expanded :  8,340 -> 8,553   = 1.03x
matched median time     :  639.4s -> 247.2s = 0.39x
matched median peak     : 15,593 -> 21,687  = 1.39x
```

The two registered readings were *"expanded down + time up -> bounds tightened"* and
*"expanded flat + time up -> no tightening, cost is pure arithmetic"*. **Neither describes
this.** Observed: **expanded flat (1.03x), peak up (1.39x), time DOWN (0.39x)**.

**A2's premise is empirically false here.** It reasoned from the paper's Section C that sample
computation costs ~|D|x more arithmetic per estimate, with |D| going 1 -> ~2,000, and predicted
the per-sentence run would be *slower*. It is **2.6x faster**. Same search work (expanded
1.03x), larger frontier, much less wall clock — the per-element arithmetic got cheaper under
partitioning, not dearer. Nothing here diagnoses why; likely candidates are vectorisation over
a 2,000 x 53 grid versus one 24,199-element vector, but that is unmeasured and not claimed.

**The timeout improvement is fully explained by the speedup, not by any bound tightening.**
Timeouts are wall-clock, the wall clock fell 2.6x, and three pairs crossed back under the cap.

## What this does to D5.0 — and a defect in my own scorer

`src/exp_partition_L4_score.py` printed *"D5.0's founding frontier-explosion observation is
substantially a SAMPLE-REPRESENTATION ARTIFACT. Append a superseding diary entry."* **That
output is wrong and is retracted.** It fires off `if a or b`, i.e. off P5 being supported at
all, and P5 was supported only by disjunct (a).

**D5.0's founding observation was about FRONTIER EXPLOSION. The frontier did not shrink — it
grew, on every matched pair.** So:

> **The frontier explosion is NOT a sample-representation artifact. It is 22-39% WORSE under
> the paper's own sample definition.** What was a sample-representation artifact is the
> *wall-clock* wall: 3 of 4 timeouts were an artifact of per-element arithmetic cost, not of
> search-space growth.

**D5.0 stands on its founding observation and is not superseded.** A superseding entry is
**not** written. What is appended instead is the narrower correction: the timeout counts in the
D5 series measured arithmetic cost as much as combinatorics, and the frontier numbers — which
were the actual claim — hold and understate.

## C2 one-sided check, with Power and Membership attached

```
PER-SENTENCE L4 :  DEFINITE MISS 0    INCONCLUSIVE 26    SENTINEL 1  (unit87, timeout)
FLAT L4         :  DEFINITE MISS 0    INCONCLUSIVE 23    SENTINEL 4
```

**Power:** margin `m` over the 26 scoreable pairs — min 0.101%, **median 2.765%**, max 11.709%.
A 4.74%-scale loss would be caught on 18 of 26; a **0.93%-scale loss on only 5 of 26**.
**Membership:** the statement covers the 26 that terminated; `unit87` is excluded and is the
hardest pair in both runs.

**So: no large in-grammar loss detected at length 4 in either partition, and the check is
near-blind to small ones.** It must not be reported as "length 4 is optimal".

### Correction issued during this session

On first reading the run log I said one pair showed `part_L4 < true_L3` and called it a
definite miss. **It is not.** The script's internal `V0` line counts `nan` as "not reaching",
and that pair is `unit87` — a timeout with no returned label. Sentinel, not miss. This is the
**fourth** appearance of a non-numeric value being read as a result in this project, and the
first since the standing rule was written; the rule was applied one step later than it should
have been, at the reporting boundary rather than the reading boundary.

## D6 instance 14 — a registered consequence attached to a disjunction

**Field: Discrimination.** P5 was registered as `(a) OR (b)`, with a **single** diary
consequence attached to the disjunction: *if P5 holds, D5.0 is superseded*. The two disjuncts
measure different things — (a) wall clock, (b) frontier size — and they fired in **opposite
directions**. The registered consequence could not express that, so the scorer emitted a
verdict about D5.0 that the data contradicts.

The Discrimination field asks *what input would change the answer*. It was satisfied for P5's
own verdict and **not** for the consequence hung off it. **A consequence attached to a
disjunction needs its own Discrimination check, one per disjunct.**

---

# REFINEMENT CHURN — the frontier grew from re-insertions, not from search-space growth

2026-08-02. Length 3, both partitions, all 27 pairs. Counting Algorithm 1 line-18
re-insertions directly: `heappush` calls carrying `heuristic == "sample"`, the re-insert at
`optimal.py:704-707` that follows a refined estimate.

```
                        flat (1 sample)     per-sentence (2,000)
re-insertions, total          2,753                 18,776         6.8x
re-insertions, median            72                    622         8.6x
peak frontier, median           962                  1,143        1.19x
distinct nodes expanded         477                    474        0.99x
pairs with ZERO re-insertions   1/27                   0/27
```

**Confirmed, and it settles the 1.39x / 1.03x split.** Refinement re-insertions rise **8.6x**
under the partition while distinct nodes expanded is unchanged. The frontier holds more nodes
because **the same nodes are re-inserted more often after refinement**, not because more
formulas are being explored.

**Restated finding, and the previous wording is withdrawn:**

> **Peak frontier grew from refinement churn. Distinct nodes expanded were unchanged (1.03x at
> length 4, 0.99x at length 3).** The partition did not make the search bigger; it made the
> same search re-queue its nodes more.

The earlier phrasing — *"the partition ENLARGED the frontier"*, *"22-39% larger"* — implies a
larger search. **It is not.** The B1 band verdict stands as a statement about peak frontier,
which is what B1 was registered on, but the interpretation attached to it was wrong and is
corrected here.

## This falsifies A3, which was registered and used

A3 stated: *"with |D| = 1, aggregated and sample computation are identical, so Algorithm 1's
refine-on-pop step (lines 14-21) was a no-op for the entire flat series."*

**False. The flat run performed 2,753 refinement re-insertions**, with a median of 72 per pair
and only **1 of 27** pairs at zero. Refinement was never disabled at |D| = 1 — it was **8.6x
less frequent**, which is a different claim.

The prediction that motivated this measurement said *"flat ~ 0"*. The **direction and ratio
are strongly confirmed; the magnitude is not.** Recording both, because a confirmed mechanism
with a wrong magnitude is how an over-strong claim gets built on a real effect.

**What this does NOT change:** P3's recovery was already re-attributed to **pop ordering**, on
the independent evidence that both miss-prefix ceilings are byte-identical and both prefixes
are still dropped. That attribution never rested on A3. What A3 supplied was the phrase *"a
disabled component turned back on"*, which is now wrong — the component was always on.

## D6 instance 15 — mine, field = Quantities

**P7's rationale argued about pruning; P7 was registered against peak frontier.**

The rationale reasoned that the aggregated estimate is partition-invariant and that
`reduce_frontier` prunes on it at insertion — which governs **which nodes are explored**, i.e.
**nodes expanded**. That quantity behaved exactly as the rationale predicted: **1.03x at length
4, 0.99x at length 3 — near-invariant.** The rationale was *right*.

It was registered against **peak frontier**, which additionally counts re-insertions of nodes
already explored, and that moved 1.19-1.39x. **Right mechanism, wrong dependent variable.**

P7 would have been SUPPORTED on the quantity its own rationale described.

**Recorded with the aggravating detail:** this was committed while writing a registration whose
entire purpose was to prevent that class of error, in a file that by then contained thirteen
logged instances of it. Having the checklist did not make me apply it to the thing I was
writing at the time.

---

# A3's MAGNITUDE FAILURE — the proposed cause is wrong, and the correct one is narrower

The suggested cause was: *line 17 re-inserts on `UpdatedNode.max_iou > MinIoU`, not on the
estimate having changed, so every surviving aggregate node bounces once under either
partition.*

**Checked against the control flow at `optimal.py:685-709` (pinned SHA), and it does not hold.**

```python
if node_heuristic != "sample":
    new_max, _ = path_heuristic.update_paths_iou(heuristic_name="sample", ...)   # :687-698
    if new_max < -e_node:                    # :699  <- TRIGGER: the estimate DECREASED
        if new_max >= minimum_threshold:     # :701  <- inner guard, not the trigger
            heapq.heappush(...,"sample")     # :702-707
```

The re-insert is gated **first** on `new_max < -e_node` — the refined estimate being strictly
lower than the current one. `>= minimum_threshold` is the *inner* guard deciding whether the
node is worth keeping, not the condition that fires the bounce. **A node whose refined estimate
does not decrease is never re-inserted.**

**The data refutes it independently.** If every surviving aggregate node bounced once,
re-insertions would track distinct nodes expanded (~475) under both partitions. Observed:

```
                    re-insertions (median)   nodes expanded (median)   pushes (median)
flat                        72                       477                  ~1,700   (~4% of pushes)
per-sentence               622                       474                  ~2,140   (~29% of pushes)
```

**Most flat nodes never bounce** — 72 against 477 expanded. The claim requires ~475.

**Correct cause, narrower:** re-insertion requires the *sample* estimate to come out strictly
tighter than the *sum* estimate. At |D| = 1 that happens on roughly 4% of pushes; per-sample it
happens on roughly 29%. Refinement is not a no-op at |D| = 1 — it simply finds a strictly
tighter bound far less often, because per-sample minima can bind where a single pooled sample
cannot. A3's error was reading "aggregated and sample coincide at |D| = 1" as "the refinement
step does nothing", when the two estimators are different heuristic families and disagree even
on one sample.

## Named failure mode — asserting code behaviour from a name, an expression, or an intent

Recorded separately from the numbered instances, because **the remedy is not a checklist
field.** No comparison precondition would have caught any of these. The remedy is: **read the
control flow.**

**Four occurrences this session, all mine:**

1. **`can_improve_or_iou_disjoint_case` as root cause** — asserted from the function's name and
   its comment, without checking what the branch computed or whether it was reachable.
   Falsified by F1.
2. **C2, voiding the `in_grammar_max` column** — asserted `max_length` was an exact length
   without reading `:566` / `:575`. It is a maximum; the column was a working one-sided
   detector I nearly discarded.
3. **`exp_noprune` substitution 3b** — disabled upstream's `ValueError` on the assumption the
   invariant "no longer applies", without checking whether it was load-bearing. It was; the
   build left the length-3 space and the experiment was void.
4. **A3** — asserted refine-on-pop was a no-op at |D| = 1 from what aggregation *means*, not
   from the guard at `:699`.

**The pattern:** in each case a plausible reading of a name, an expression, or a design intent
was substituted for the branch condition actually controlling execution, and in each case the
code was available and short. Three of the four were caught only when a measurement disagreed.

**This session's fifth occurrence is the one above** — the proposed line-17 cause — which is
not mine, and which I record only to note that the failure mode is not personal to me and that
checking it cost one `sed` of 25 lines.

## Decomposition of the 8.6x — one sentence, not scheduled

There are **two** re-insertion sites both tagged `"sample"` — the estimate-decrease path at
`:702-707` and the distributive-property path at `:753-758` — and the counter did not separate
them, so which one carries the excess is **unmeasured**; the trigger `new_max < -e_node` fires
on ~4% of flat pushes against ~29% per-sample, which is sufficient to produce the 8.6x without
invoking the second site at all.

---

# ITEM 4 — THE EVENT-ORDERING HARM MEASUREMENT. Gate passed.

2026-08-02. `src/exp_event_ordering.py`, output `results/event_ordering_L3.txt`.
Length 3, K=15, M=24,199, all 27 pairs, both partitions.

```
measure                                             flat     per-sentence
OPPORTUNITY  inadmissible ceilings (set-based)     14/27        13/27
OPPORTUNITY  all optimal prefixes pruned (bound)    4/27         4/27
OPPORTUNITY  optimal leaf ancestors pruned      31 / 24 pairs      -
HARM         pairs that actually lost the optimum      2            0
SENTINEL     optimal formula never CREATED            0            0
SENTINEL     created, never scored or dropped         2            0
SENTINEL     pair timed out                           0            0
```

**14 opportunities produce 2 harms; 13 produce 0. The tightest bound overstates by 2x and by
4 -> infinity.** That gap is the result.

**Margin redefined as THRESHOLD HEADROOM** — ceiling minus incumbent threshold at the scoring
pop. Copy-independent and continuous. It replaces the pop-distance margin, which was not a
clean quantity: it went negative (-25, -183, -217) whenever copies of a node interleaved.

```
flat          25 scored optima   min +0.00093000  median +0.06700778  max +0.39505257
per-sentence  27 scored optima   min +0.00027657  median +0.12325959  max +0.39668945
```

Comparable to the two prefix-level drop margins (0.00025690, 0.00033742): **the tightest
surviving optimum cleared the threshold by 0.00093 flat and 0.00028 per-sentence** — the same
order as the margins by which the lost prefixes failed.

## The loss site is NOT `reduce_frontier`, and the assumed headline does not hold

Event dump, optimal formula on trained a=0.2 unit88:

```
t=348  CREATED  heappush   ceiling=0.4175506268081003  threshold=0.25056904400606983
t=349  POPPED   heappop    ceiling=0.4175506268081003  threshold=0.25056904400606983
(no SCORED, no DROPPED — the node ends here)
```

**Popped with an aggregated ceiling of 0.4176, sitting well ABOVE both the threshold (0.2506)
and its own exact IoU (0.2545), then discarded before scoring.** Not dropped by
`reduce_frontier`; not skipped by the `:697` threshold test, since 0.4176 > 0.2506.

The only remaining path is `optimal.py:699-709`: the node is re-estimated with the **sample**
heuristic, and if `new_max < -e_node` and `new_max < minimum_threshold` it is neither re-pushed
nor scored — it is silently dropped.

**So the killing bound is the REFINED estimate, not the aggregated one.** The proposed headline
— *"a complete formula discarded on an upper bound below its own IoU"* — is **NOT supported as
stated**: the bound it carried when popped is *above* its IoU and therefore admissible. The
refined estimate that replaced it is what fell below the threshold, and **its value is not
captured by the current instrumentation**, which sees only pushes, pops, drops and scores. One
further hook on `path_heuristic.update_paths_iou` would settle it.

**Nothing is written into `UPSTREAM_REPORT.md` on this.** The claim needs the `new_max` value,
and asserting it from control flow without measuring it is the named failure mode from earlier
in this session.

## "Prefix never CREATED" 1/3 -> 0/0 was an id-cache artifact — and the ties conclusion survives

The exposure audit reported *"P* never entered the frontier"* on **1 of 27 flat and 3 of 27
per-sentence**. Re-measured with corrected instrumentation, both are **0**. The old counts came
from `cache[id(f)]`: formula objects are short-lived, CPython reuses ids after GC, and the
cache returned stale answers non-deterministically.

**That false signal was the trigger for the entire ties / `n_prefixes` investigation.** The
independence must be stated rather than left to a reader:

> **The `n_prefixes` conclusion does not depend on the event hooks.** It rests entirely on the
> brute-force prefix enumeration — pure numpy over concept masks, no instrumentation, no
> `id()`. The 1-3 prefixes per optimum, the 2/3 signature-space share, the operator-repetition
> derivation, and the finding that both losses have `n_prefixes = 1` all come from that
> enumeration. **A false trigger led to a true investigation.** The conclusion survives its own
> origin.

## STANDING RULE — never key on object identity

Second identity defect in this project, and worse than the first.

```
first  : F.Or(a,b) vs F.Or(b,a)   structurally equal, distinct objects   deterministic, VISIBLE
second : cache[id(f)]             ids reused after GC                    non-deterministic, SILENT
```

**Key on the evaluated mask or a canonical string. Never on `id()`, and never on object
identity for formula objects.** The first produced an obviously wrong answer — 0 of 27 nodes
found — and was caught in minutes. The second produced *plausible* answers, 1 and 3, small
numbers in the right range, and survived long enough to launch an investigation.

---

# W2 FIRES — the refined estimate is admissible. Mechanism UNIDENTIFIED, to D7.

2026-08-02. `src/exp_refined_estimate.py`, output `results/refined_estimate.txt`.
Registered before the hook existed (`aac37d5`).

```
trained a=0.2  unit88   refined 0.4175506268081003  threshold 0.25056904400606983  exact IoU 0.25454105110196174
trained a=0.05 unit86   refined 0.546485260770975   threshold 0.1771041084962106   exact IoU 0.21660649819494585
```

**Both refined estimates are above the threshold AND above the exact IoU they bound.** The
sample heuristic is admissible on these two complete formulas. **W1 is refuted; W2 fires.**

**The winning child did not die at `optimal.py:699-709`.** Per the registration: the mechanism
is **UNIDENTIFIED**, nothing is written to `UPSTREAM_REPORT.md`, and it goes to **D7**.

What is now excluded, each by measurement rather than reading:

```
reduce_frontier threshold prune   excluded -- no DROPPED event on the formula
the :697 incumbent skip           excluded -- ceiling 0.4176 > threshold 0.2506
the :699-709 refinement discard   excluded -- refined 0.4176 is NOT < threshold
```

`recent_nodes` dedup (`:749-757`) and the distributive-property path (`:731-767`) remain
un-excluded. **They are not asserted as the cause** — that inference is exactly the named
failure mode, and this would be its seventh occurrence.

## The FINAL flag fired, and it is a separate finding

On **every** refinement event, on both cases:

```
next_op = INDIVIDUAL   FINAL = True   heuristic = sample
```

**A complete formula was estimated rather than evaluated.** The node was flagged final — its
exact IoU was computable in one mask operation — and the search spent a heuristic estimate on
it instead. This is a **control-flow observation, not a bound defect**, and it gets its own
sentence as registered in advance.

It does not by itself cause the loss (the estimate produced was admissible), but it means the
search carries complete formulas through an estimation path where an exact evaluation was
available.

## Retraction — "discarded on an upper bound below its own IoU"

**Mine, and it was drafted before the measurement.** It came from extrapolating the
prefix-level trace, where the ceiling genuinely was below `true_max(P)`, down to the child,
where it is not. Measured:

```
ceiling  0.4175506268081003
IoU      0.25454105110196174
threshold 0.25056904400606983
                                 ceiling > IoU > threshold
```

The bound is **above** the value it bounds — admissible. And the sharper point the wrong
wording obscured: **the formula's own IoU exceeds the incumbent threshold by 0.00397201, so
had it been scored it would have become the new incumbent.** It was discarded while strictly
better than the incumbent it was compared against.

**Sixth assert-from-intent occurrence, and the first where the wording was supplied in advance
of the measurement.** That is the distinguishing feature: the previous five were assertions
about code I had read; this one was a sentence written to be filled in by a number, and the
number contradicted it. Drafting the conclusion before the measurement is a distinct hazard
from misreading the code, and it is the one that scales worst — the sentence was already
well-formed, quotable, and pointed at the right file.

## Margin comparison — carried with every appearance of 0/27

**Wherever `0/27 losses per-sentence` appears, this goes with it:**

```
per-sentence minimum headroom   +0.00027657
flat drop margins               0.00025690  and  0.00033742
```

**The closest per-sentence survivor cleared the threshold by less than one of the flat losses
failed by.** `0/27` is not a safety margin. It is one favourable pop-ordering away from being
`1/27`, and the distance is measured, not inferred.

---

# D6 (continued) — REGISTRATION REVERSAL, logged as a reversal

2026-08-02. This is **D6, not D7.** The mechanism question is the unfinished half of D6's own
question — *is the optimality guarantee sound* — so it belongs here. D7 remains the length-4
oracle, the fork-only rerun at `optimal_utils.py:271`, D/E/F, and queue items 3/4/5.

**The reversal, stated as one:**

```
registered   W2 fires -> "mechanism UNIDENTIFIED, nothing written, goes to D7"
outcome      W2 fired (refined estimates 0.4176 and 0.5465, both admissible)
action taken deferral REVERSED, same day
grounds      instrumentation had narrowed the exit paths from five to two
```

**This is a reversal of a registered outcome, and it is recorded as one rather than
relabelled as new work.** The registration was honoured at the moment it fired — the deferral
was written, committed, and reported — and then overturned on new information about tractability,
not on a new preference about the answer.

**The stop condition is unchanged, and it is M3.** M3 is not a deliverable boundary; it is a
halt: if neither enumerated exit fires, that is the result, and **no further hooks are added
until something fires.** The distinction matters because the previous deferral was a boundary
and this one is a halt, and only the second survives being close to an answer.

**`UPSTREAM_REPORT.md` still requires its own registration before anything from this enters it.**

---

# D6 item 0 — READ BEFORE INSTRUMENTING. Source, verbatim.

Pinned SHA `70805299`, `compositional/optimal.py`.

## Exit A — distributive/equivalence re-push

```
:712   transformed_label = apply_distributive_property(node)
:713   if transformed_label != label_node:            <- guards the whole block
:732       if new_max < -e_node:                      <- trigger for re-push + continue
:733           if new_max >= minimum_threshold:       <- inner guard on the re-push
:734-737          heapq.heappush(current_frontier,
                      (-new_max, next_op_node, label_node, node[3], "sample"))
:740           if new_min > minimum_threshold:        <- threshold raise + reduce_frontier
:746-747       done = len(current_frontier) == 0 ; continue
```

Note the re-push at `:734-737` re-inserts **`label_node`**, the *original* label — not
`transformed_label`. The transform is used only to compute a tighter estimate.

## Exit B — `recent_nodes` memory skip

```
:750   if -e_node >= recent_e_iou:
:751       if node in recent_nodes:
:752-753       done = len(current_frontier) == 0 ; continue
:755       else: recent_nodes.append(node)
:757-758   else: recent_nodes = [node] ; recent_e_iou = -e_node
```

## How "Node in Memory" tests membership — and the candidate defect is REFUTED

`recent_nodes` is a **list**, so `node in recent_nodes` is a linear scan using `==` on the
whole 5-tuple `(e_iou, next_op, label, paths_to_expand, heuristic)`. Element `[2]` dispatches
to `F.Or.__eq__` / `F.And.__eq__`.

**Those are explicitly commutativity-aware** (`formula.py:266`, `:343`): the mono-operator case
sorts `get_vals()`, and the general case tests `left==other.left and right==other.right` **or**
`left==other.right and right==other.left`. `compute_hash_value` (`formula.py:28-48`) sorts
flattened operands for AND and OR by design.

Tested directly rather than read:

```
Or : x==y True   hash(x)==hash(y) True   contract OK   y in {x} True   y in [x] True
And: x==y True   hash(x)==hash(y) True   contract OK   y in {x} True   y in [x] True
```

**No `F.Or(a,b)` vs `F.Or(b,a)` defect exists in upstream. The hash/eq contract holds.** The
membership test is sound on the formula element. It is *not* a candidate defect and is not
reported as one.

### Correction to my own standing rule

The standing rule *"never key on object identity"* cited two instances, the first being
*"`F.Or(a,b)` vs `F.Or(b,a)` — structurally equal, distinct objects — deterministic, visible"*.
**That attribution is wrong and is withdrawn.** Upstream's equality and hash both handle
commutativity correctly, as just tested.

The failure that attribution described was real — an early exposure audit reported *"P* never
entered the frontier"* on all 27 pairs using `n[2] == _P`, where a later mask-based test found
events. But **the cause of that failure is unestablished.** The most likely explanation is that
`_P` was one arbitrary optimal prefix while the node carried a *different* formula with the
same mask — which is instance 8, the under-specified subject, not an equality defect. **I am
not asserting that either.** What is established: commutativity is not the cause, and the
second instance (`cache[id(f)]`) stands unaffected.

The standing rule itself survives on the `id()` instance alone, and is narrowed to that.

---

# D6 item 1 — HOOK COVERAGE AUDIT. Two uncovered classes found.

Pinned SHA, `compositional/optimal.py`. Hooks in place: CREATED
(`HeapProbe.heappush` + `HeapProbe.heapify`), POPPED (`HeapProbe.heappop`),
DROPPED (wrapper on `optimal.reduce_frontier`), SCORED
(`mask_utils.get_formula_mask_and_tree`).

## Frontier-insertion and pop sites — all covered

```
site   enclosing function              op         covered by
:98    update_frontier_by_ancestors    heapify    CREATED
:427   reduce_frontier                 heapify    CREATED (re-heapify of survivors)
:490   update_frontier                 heappush   CREATED
:494   update_frontier                 heapify    CREATED
:674   perform_search                  heappop    POPPED
:704   perform_search (refine re-push) heappush   CREATED
:735   perform_search (distributive)   heappush   CREATED
```

`optimal.heapq` is replaced by the `HeapProbe` instance, so every `heapq.*` call inside the
module dispatches to a hooked method. **No insertion or pop site is uncovered.**

## Removal sites — TWO CLASSES UNCOVERED

```
reduce_frontier filtering   :408 :485 :743 :794 :828 :866   COVERED (all 6 call sites)

UNCOVERED 1 -- silent non-append during estimation
  :389   `if node_path_max_iou > 0:` in estimate_iou_frontier
         A path whose estimate is <= 0 is never appended to frontier_estimates. It is
         removed with NO heapq call and NO reduce_frontier call. Invisible to every hook.

UNCOVERED 2 -- popped-then-continued
  :679 :709 :747 :753 :804 :871
         Six `continue` statements in perform_search. A node popped at :674 and reaching any
         of these is gone unless separately re-pushed. POPPED fires; nothing distinguishes
         "popped and processed" from "popped and discarded", and nothing says WHICH continue.
```

**The audit's own answer to the question that prompted it:** the two exits under investigation
(`:747` distributive, `:753` memory) are both in UNCOVERED class 2. That is precisely why they
cannot be separated from the existing log, and precisely what M1/M2/M3 will hook.

## Consequence — the three exclusions are requalified

The exclusions in `UPSTREAM_REPORT.md` rest on **absence of events**, and absence is evidence
only under complete coverage. Coverage is not complete. Restated in the honest form:

```
was      "reduce_frontier threshold prune EXCLUDED -- no DROPPED event"
is now   "no DROPPED event under hooks covering all six reduce_frontier call sites
          (:408 :485 :743 :794 :828 :866)"
```

The `:697` and `:699-709` exclusions are **unaffected** — they rest on positive measurements
(ceiling 0.4176 > threshold 0.2506; refined 0.4176 above both threshold and the IoU it bounds),
not on absence.

---

# M2 RESTATED — Exit A is not a discard

`:734-737` re-pushes **`label_node`**, the original label, not `transformed_label`. So Exit A
returns the node to the frontier with a lowered estimate. **It would appear as a SECOND
CREATED, not as a disappearance.**

The existing log for the optimal child is:

```
t=348  CREATED   ceiling=0.4175506268081003
t=349  POPPED    ceiling=0.4175506268081003
       (nothing after)
```

**No second CREATED. Conditional on hook completeness — which the audit above now bounds — that
already points at Exit B.** The run confirms or refutes it; the audit is what the inference
rests on, and the audit says insertion coverage is complete while `continue`-discard coverage
is not. So a missing second CREATED is meaningful, and a silent exit is not attributable
without the new hooks.

---

# OPEN ITEM — the early "0/27" via `n[2] == _P`, cause unestablished

Promoted from a parenthetical to an open item, because it was used as the basis of a standing
rule and its cause has never been established.

**Observed:** an early exposure audit compared frontier nodes to one enumerated optimal prefix
with `n[2] == _P` and reported *"P* never entered the frontier"* on **all 27 pairs**. A later
mask-keyed test found events on 26 of 27.

**Excluded:** commutativity. `Or.__eq__`/`And.__eq__` and `compute_hash_value` are
commutativity-aware, verified by direct test — `==`, hash equality, and both set and list
membership all hold for `Or(a,b)` vs `Or(b,a)`.

**Not established:** why the comparison failed. The leading candidate is instance 8 — `_P` was
one arbitrary member of a 1-3 element optimal-prefix set while nodes carried other members —
but that has not been tested and is not asserted.

**Status: OPEN.** It is not load-bearing for any current claim; the exposure figures were all
recomputed mask-keyed.

---

# The standing rule propagated an unverified diagnosis — mine

Recorded as its own failure, distinct from the numbered instances and from
assert-from-intent.

The rule *"never key on object identity"* was written from **two** cited instances. The first,
`F.Or(a,b)` vs `F.Or(b,a)`, was **a diagnosis I never verified** — I inferred it from a symptom
(0/27 matches) and generalised it into a rule in the same commit, without running the
three-line equality test that refutes it.

**The shape: a diagnosis was promoted to a general rule before the diagnosis itself was
checked.** It is worse than a wrong diagnosis, because a rule outlives the case that produced
it and carries its error into unrelated work. This one would have had me avoid a correct
upstream equality implementation.

**The narrowed form is correct and stands:** `cache[id(f)]` is unsound because CPython reuses
ids after GC — verified, non-deterministic, and the actual cause of a measured artifact
(`never CREATED` 1/3 → 0/0). **Key by evaluated mask or canonical string. The `id()` half of
the rule is the whole rule.**

---

# B1 CONFIRMED — the E.2.2 violation is OUR CONFIGURATION, not the domain

2026-08-02. Counting only, no search. Same corpus (M = 24,199, 2,000 sentences), same masks,
per-sentence partition, neuron `trained a=0.2 unit88`.

```
     K  concepts  common%   sentences with Bott_1 == 0    min  median  max
    15        15     83.8%                       83.5%      0       0    1
    50        50     96.6%                      100.0%      0       0    0
   100       100     99.3%                      100.0%      0       0    0
  1168      1168    100.0%                      100.0%      0       0    0

K = 1168 : per-concept dataset-wide E^C totals  min 0  median 10  max 13,698
           concepts with dataset-wide total 0    24 of 1,168
           Bott^A_1(E^C) = 0
```

**B1 holds. B2 does not.** At the full `min_support >= 5` vocabulary — the paper's own
vocabulary scale — `Bott_1(E^C)_x = 0` on **every sentence**, and the aggregated
`Bott^A_1(E^C) = 0` as well. The precondition is satisfied at paper scale.

It is already satisfied at **K = 50**.

**Mechanism, and it is the obvious one once measured:** with a large vocabulary most concepts
are absent from any given sentence, so some concept contributes zero extras-in-common and the
minimum is 0. **24 of 1,168 concepts have a dataset-wide total of zero.** Top-K on
high-support features removes exactly the concepts that would have supplied the zero.

## What this does to the finding

**The E.2.2 violation is a property of our configuration, not of text.** Two contributions,
now separable:

```
single-sample partition   dominant  : Bott_1 = 859, 0% compliant       (our reshape(1, M))
top-K = 15 selection      residual  : 16.5% of sentences non-compliant at K = 15
paper-scale K = 1168      NONE      : 100% compliant on every sentence
```

The earlier framing — *"on a token-level corpus with min_support + top-K selection we measure
`Bott_1 = 859` on 100% of samples"* — presented the violation as a domain characteristic. **It
is not.** It is what our `K = 15` and our single sample produce. At the paper's own vocabulary
scale and its own sample definition, the condition holds everywhere we can measure it.

## The limit, recorded either way — precondition testable at paper scale, consequence not

`Bott_1` is measurable at K = 1168 because it is **counting**. The 2/27 misses are **not**
verifiable there: an exhaustive in-grammar oracle at K = 1168 is `K * (3K)^2 = 1168 * 3504^2`
~ **1.4e10 formulas per unit**, times 27 units.

So: **the precondition can be tested at paper scale; the consequence cannot.** We can say the
condition holds at K = 1168. We cannot say whether the search is optimal there, and nothing
here licenses extrapolating the 2/27 either way.

This asymmetry is itself worth reporting — it is the reason the finding cannot simply be
re-run at paper scale to settle it.

## Consequence for `UPSTREAM_REPORT.md` — claim (a) is demoted

Claim (a) was *"the aggregated bound is inadmissible when `Bott_1(E^C)_x != 0`"*, supported by
`Bott_1 = 859` on 100% of samples. The **inadmissibility demonstration stands** — the ceiling
0.232677 against a reachable 0.254541 is a measured fact about the estimator given a violating
input.

**What no longer stands is the implication that the violating input is what a text corpus
looks like.** It is what K = 15 on a single sample looks like. The report must say that, and
must say that at K >= 50 with per-sentence samples we cannot produce a violation at all.

That materially weakens the report as a criticism of the method and strengthens it as a
statement about a precondition that is easy to violate by configuration — which is a different
paper, and a fairer one.

---

# D6 closing — registrations, a retracted run, and two records

2026-08-02.

## The K=50 oracle: FIRST RUN RETRACTED — float32 precision artifact

The first K=50 run reported **14 of 27 misses, including both K=15 known misses**, and would
have been reported as **K1 firing**. **It is retracted before reporting.**

**Every one of the 14 had a gap of `+0.0000%`:**

```
returned 0.26220362622036264   "optimum" 0.2622036337852478   difference 7.5e-09
float32 epsilon 1.19e-07
```

The oracle computed intersections by **float32 GEMM**. Counts up to 24,199 are not exactly
representable in float32, so the GEMM result lands a few ULPs *above* the true integer count,
and the `> 1e-12` test flagged every pair. The search returns float64 computed from integer
counts. **The oracle was less precise than the thing it was auditing.**

Re-run in **float64**, where counts <= 24,199 are exactly representable, with the test at
`1e-9`. Nothing from the first run is reported.

This is the same class as the rounded-`exact_IoU` error that produced 17/27 instead of 2/27,
and the `nan`-as-success error: **a reference computed to lower precision than the measurement
it judges.**

## Registered before the K=50 result lands — THE CONFOUND

**K = 50 changes two things at once.** The precondition holds there (B1), **and** the search
space is **37x larger** (1,125,000 vs 30,375 formulas). **K1/K2 as registered cannot separate
them.** If the misses vanish, either the restored precondition or the different space could
explain it, and this run cannot say which.

Recorded before the number exists so the ambiguity is not discovered after seeing it.

## Registered — 2b, the matched-K de-confounding run

Oracle at **K = 15 with a different 15 concepts**, chosen so `Bott^A_1 = 0` — including at
least one concept with zero dataset-wide common extras. **Same space size (30,375), same cost.
Only the precondition differs.**

```
D1  misses vanish   -> the precondition is the driver
D2  misses persist  -> K itself is, and the precondition story is incidental to them
```

The chosen vocabulary is to be reported explicitly so the selection is auditable.

## Item 3 — hook coverage audit: already done, restated

Committed at `90ef8be`. All seven insertion/pop sites covered (`:98 :427 :490 :494` heapify /
heappush, `:674` heappop, `:704 :735` re-pushes); all six `reduce_frontier` call sites covered.
**Two removal classes UNCOVERED:** the silent non-append at `estimate_iou_frontier:389` when a
path estimate is `<= 0`, and six `continue` statements (`:679 :709 :747 :753 :804 :871`) after
which a popped node is gone with no event. The `reduce_frontier` exclusion in
`UPSTREAM_REPORT.md` is already requalified to *"no DROPPED event under hooks covering sites
:408 :485 :743 :794 :828 :866"*.

## Item 5 prerequisite — the transcription discrepancy is RESOLVED, and the target moves

The hypothesis was right. `optimal_utils.py:477-521` (`compute_max_iou_from_label_info`) is
reached from `estimate_min_max_iou_from_label_info`, which `path_heuristic.estimate_paths_iou`
calls at `:389-396` to score the **INDIVIDUAL path** — the final-path label dIoU, Eq 3 / Def
3.5. **It is not the aggregated path heuristic.**

The aggregated **path** quantities live in `compositional/optimal_sum_heuristic.py`:

```
:10   estimate_disjoint_label_info
:148  estimate_label_info
:368  or_chain_estimation
:509  and_chain_estimation
:581  and_not_chain_estimation
```

**And this relocates the target.** On unit88 the ceiling that got the prefix dropped was
`0.23267674991206472`, which **equals `IoU(P)` exactly** — the INDIVIDUAL-path value. So the
node's key was its own dIoU, meaning **the AND-chain path estimated LOWER than the true
AND-reachable 0.254541**. The suspect function is **`and_chain_estimation` at
`optimal_sum_heuristic.py:509`**, not `optimal_utils.py:477-521`.

Eq (50)/(51) are to be hand-computed against **that** function's output. Nothing is computed
yet, and no verdict is recorded.

---

# RECORD against item 1's length ladder — K = 8 is the most non-compliant configuration measured

```
K       sentences with Bott_1 == 0     max Bott_1     Bott^A_1
 8            35.4 - 40.1%                  8         2,665 - 2,920
15            62.4 - 75.9%                  1         1,202 - 1,576
50                   100%                   0                 0
```

(units = the length-ladder set, trained a=0.1, 88/92/396/413/510)

**The length ladder ran at K = 8**, where roughly 38% of sentences satisfy the precondition —
the worst configuration in this project. An under-computed ceiling **prunes more than a sound
bound is entitled to**, and that cuts item 1's two conclusions in opposite directions. Both
belong with the results:

**COST CONCLUSION — STRENGTHENS.** Length 5 failed to terminate on 2 of 5 units *while pruning
more than it was entitled to*. A sound search explores strictly more and is slower still. The
"length 5 does not terminate" finding is a lower bound on the difficulty.

**GAIN CONCLUSION — WEAKENS, and the caveat is load-bearing.** The measured gains
(+0.000% / +0.055% / +0.921% IoU from length 4 to 5) are **lower bounds**. An over-pruning
search finds worse optima, and the length-5 space offers more opportunities to over-prune than
length 4 — so the difference is biased downward. **"Length 5 buys nothing" is exactly what an
over-pruning search manufactures.** At 38% precondition compliance the bias is not a remote
possibility, and the conclusion should not be quoted without this attached.

---

# `UPSTREAM_REPORT.md` — DO NOT SEND, and why

The current draft asserts a precondition violation as a property of token data. **B1 refuted
that**: at K >= 50 with per-sentence samples the condition holds on every sentence, and at
K = 1168 `Bott^A_1 = 0`. The draft is **wrong as written**.

It stays untouched until items 2, 2b, 3 and 5 have answers. Item 3 is answered; 2, 2b and 5
are not. **Anything arising outside the closing list goes to D7.**

---

# RETRACTION — the inadmissibility demonstration was a CATEGORY ERROR

2026-08-02. Item 3's pre-filter log settles it, and the answer is not the one the elimination
argument pointed at.

## The four path estimates at node P, pre-filter

```
trained a=0.2 unit88     IoU(P) = 0.23267674991206472   true reachable from P = 0.25454105110196174
    INDIVIDUAL  0.23267674991206472
    OR          0.5360534646500176   <- MAX
    AND         0.4175506268081003
    NOT         0.40520673813169983

trained a=0.05 unit86    IoU(P) = 0.11240400279264604   true reachable from P = 0.21660649819494585
    INDIVIDUAL  0
    OR          0.20339771933907377
    AND         0.546485260770975    <- MAX
    NOT         0.45609065155807366
```

**Every extension path bounds the true reachable value from above.** AND is 0.4176 against a
reachable 0.2545 on unit88, and 0.5465 against 0.2166 on unit86. **`and_chain_estimation:509`
is exonerated. So are `or_chain_estimation:368` and `and_not_chain_estimation:581`. No chain
estimate is inadmissible on either pair.**

## What the "inadmissible ceiling" actually was

`estimate_paths_iou` emits **four separate frontier entries per node** — one per path. That is
why the event log showed **four CREATED events at t=10** for P.

The entry dropped at t=206 with ceiling `0.23267674991206472` is the **INDIVIDUAL-path entry**,
and that number is exactly `IoU(P)`. The INDIVIDUAL path means *stop here*, so its correct
bound **is** `IoU(P)`. Dropping it when the threshold reached `0.23293365307753797` is
**correct behaviour**.

The OR, AND and NOT entries — the ones that bound *extending* P — were never dropped. They
survived and were expanded at t=231, t=348 and t=366.

**The comparison `ceiling 0.232677 < true_max(P) 0.254541` compared the bound for "stop at P"
against the best value reachable by "extend P". Those are different quantities.** It is a
category error, and it is mine.

## What this retracts

**Retracted:** that the aggregated bound was demonstrated inadmissible on this data. It was
not. The demonstration compared a stop-here bound against an extend-from-here reachable set.

**Also retracted, pending recomputation:** every exposure figure built on that comparison —
`6/27` and `5/27` inadmissible ceilings, and the set-based `14/27` and `13/27`. All were
computed by taking a node's key at drop time and comparing it to `true_max(P)` over the whole
subtree. If those drops were INDIVIDUAL-path entries, they are correct drops and the counts
are meaningless.

**Not retracted:**
- **The 2/27 misses.** Measured by exhaustive enumeration against what the search returned.
  Independent of any bound argument.
- **The unevaluated-FINAL observation.** A complete formula, exact IoU
  `0.25454105110196174`, exceeding the incumbent `0.25056904400606983` by
  `0.003972007095891905`, carried through an estimation path and never evaluated. Independent.
- **B1.** The precondition holds at K >= 50. Counting only.
- **The K = 8 non-compliance record.** Counting only.

## Consequence for `UPSTREAM_REPORT.md`

Claim (a) as written — *"the aggregated bound is inadmissible when `Bott_1(E^C)_x != 0`"* — has
**no surviving demonstration**. The `Bott_1` violation is real in our configuration and the
estimator's admissibility is *conditioned* on it, but **we have not shown a single inadmissible
estimate.** The report cannot assert (a) and must not be sent.

What survives for upstream is the lead observation and the 2/27, with the cause unidentified.

## The error shape, recorded

**Elimination pointed at `and_chain_estimation:509` and elimination was wrong**, because the
enumeration of candidates was itself incomplete: it never included "the dropped entry is a
different path of the same node". Two entries carrying the same label are not the same node in
any sense that matters to a bound.

Both prior framings were mine, and both were built on a quantity I had not decomposed. The
instruction to log the three chain estimates *before* hand-computing anything is what caught
it — the hand-computation would have compared Eq (50)/(51) against a function that was never
implicated, matched, and been reported as "the estimator behaves as specified", which is true
and would have concealed that the premise was wrong.

---

# K2 FIRES — 0/27 misses at K = 50, per-sentence

2026-08-02. `results/k50_oracle.txt`, float64 re-run.

```
misses at K = 50 per-sentence : 0/27      sentinels : 0
the 2 K = 15 known misses     : NEITHER misses
-> K2
```

**The 2/27 misses do not survive at K = 50.** Per the registration, K2 means they are a
K = 15 artifact.

## Resolution floor, reported alongside the verdict

Counts are integers <= 24,199, exactly representable in float64 (2^53 headroom), so the GEMM
is exact and only the final division rounds.

```
relative rounding of the IoU division   2.22e-16
absolute, on IoU ~0.25                  5.55e-17
test threshold                          1.00e-09      ~1.8e07x ABOVE the floor
smallest real gap the test could miss   1e-9 absolute = 4e-7 relative
the two K=15 gaps, for scale            2.3e-3 and 9.8e-3 absolute
```

**The threshold is six orders of magnitude below the effects it needs to detect and eight
above the instrument floor.** It does not repeat the `1e-12` mistake in either direction.

## THE CONFOUND STANDS — this does not identify the driver

Registered before the result: **K = 50 changes the precondition AND makes the space 37x
larger.** K2 firing is consistent with either. **2b (matched-K, `Bott^A_1 = 0`, same 30,375
space) has not been run**, so nothing here says whether the precondition or K itself is the
driver.

## The unevaluated-FINAL observation REPRODUCES, and is configuration-free

```
445,644 of 525,933 refinement calls carried next_op == INDIVIDUAL   (84.7%)   at K = 50
```

**It is not a K = 15 artifact.** Complete formulas are estimated rather than evaluated at
K = 50 as well, in the regime where the precondition holds and no misses occur. This is now
the only finding in D6 that survives at paper-adjacent settings.

---

# RETRACTION, EXTENDED

## Partition-invariance — fact kept, consequence removed

The derivation and the byte-identical ceilings stand as a **fact about the aggregated
estimate**. **Its only role in this project was the consequence** — *"repartitioning cannot
repair the inadmissibility"* — and there is no demonstrated inadmissibility to repair. **The
consequence is withdrawn.** The fact is retained because it is measured and may matter later;
it currently supports nothing.

## Item 4's table — the left column is retracted

```
OPPORTUNITY  inadmissible ceilings (set-based)     14/27   13/27   RETRACTED
OPPORTUNITY  all optimal prefixes pruned (bound)    4/27    4/27   RETRACTED
OPPORTUNITY  optimal leaf ancestors pruned      31/24 pairs    -   RETRACTED
HARM         pairs that actually lost the optimum      2       0   STANDS
```

All three opportunity counts were computed by comparing a node's key at drop time against
`true_max(P)` over the whole subtree — the category error. **The opportunity-vs-harm gap was
the point of that table, and it is gone.** What remains is two harm counts, 2 and 0, which are
correct and are no longer contrasted with anything.

## `n_prefixes` — correlation kept, mechanism removed

**Kept:** 2 of 2 losses have `n_prefixes = 1`; 0 of 18 pairs with `n_prefixes >= 2` lost.
Kept also: the structural derivation of `n_prefixes` from the operator signature, and the 2/3
signature-space share. Those are combinatorics and are independent.

**Removed:** the explanation — *"`n_prefixes` is the number of independent chances to be
expanded before an unsound prune fires"*. There is no established unsound prune. **It is now a
correlation on n = 2 with no mechanism**, and it must be stated that way wherever it appears.
The inspection-only diagnostic built on it is correspondingly weakened: it predicts which
units were vulnerable in our runs, without a reason.

---

# INSTANCE 17 — the K=50 "reproduction" measured a different event

**The event** is: a formula flagged FINAL, **popped**, **never scored**, whose **exact IoU
exceeded the incumbent** at that moment.

**The test used** was: refinement calls carrying `next_op == "INDIVIDUAL"` — 445,644 of
525,933, 84.7%.

**That is normal behaviour.** `estimate_paths_iou` estimates every node on all four paths, and
nodes are scored on pop (Alg 1:38). The 84.7% figure says only that most estimation happens on
nodes whose INDIVIDUAL path is live. **It contains neither "never scored" nor "exceeded the
incumbent."** Field = **Quantities**.

**It was elevated to "the only finding that survives at paper-adjacent settings" before being
checked.** That elevation is what made it load-bearing.

## The measuring version — R1 fires at K = 15

`src/exp_final_discard.py`. Counts formulas that were flagged FINAL, popped, never scored, and
whose exact IoU exceeded the incumbent threshold at the moment of the pop.

```
K = 15, per-sentence, 27 pairs :  125 events, 23 of 27 pairs affected
```

Examples:

```
untrained a=0.05 unit396   IoU=0.07518796992481203 > incumbent=0.06860632183908046
                           ((tag=IN OR lemma=.) AND (NOT const=VP))
untrained a=0.05 unit413   IoU=0.08409250175192712 > incumbent=0.07840440165061899
                           ((tag=IN OR dep=pobj) AND (NOT const=PP))
```

**So the event is not a single anecdote — it happens 125 times across 23 of 27 pairs at
K = 15.** The K = 50 count is running.

**Gate, reported rather than assumed:** the `estimate_iou_frontier` non-append counter fired
468,645 times, so **nodes demonstrably can vanish before ever being popped.** A zero from this
metric would therefore be uninterpretable. A non-zero is safe — it is a lower bound. (The
counter itself is crude: it diffs input frontier length x4 against returned estimates and will
overcount; it is used only as a yes/no on whether non-appends occur.)

---

# THE EVENT AND THE HARM ARE DIFFERENT THINGS — and the report conflated them

**The event:** a final-flagged formula, popped, never scored, exact IoU above the incumbent.

**The harm:** the search's returned answer ends up below a formula it discarded.

**These coincide only at K = 15, and only on the two missing pairs:**

```
trained a=0.2 unit88, K=15
   discarded         0.25454105110196174
   incumbent then    0.25056904400606983
   FINAL RETURNED    0.2522022213711222     <- below the discarded formula. HARM.
```

**At K = 50 there were 0/27 misses — the search returned the true in-grammar optimum on every
pair.** So whatever discards occur there, **none of them cost anything, by construction.** An
unevaluated formula is only harmful if nothing better is found later, and at K = 50 something
better always was.

**Section 1 of `UPSTREAM_REPORT.md` presented the event and borrowed the force of the harm.**
That is corrected: the event may be widespread and configuration-free; **the harm is K = 15
specific and is 2 pairs.**

---

# S1 fires — but on a configuration where no harm occurs, and that is the result

2026-08-02. `src/exp_final_discard.py`, K = 15.

## The mask-level filter, with the source hypothesis verified first

`optimal.py:762` — `if label_node not in visited:` — skips a node whose label equals an
already-visited one (`visited` at `:652`, appended `:782` and `:816`, compared by
commutativity-aware `__eq__`). **Skipping an already-explored equivalent is correct
behaviour.** "This copy never scored" and "this formula's value was never computed" are
different quantities; the first version measured the first, the claim needs the second.

Filter applied at **mask** level, which is broader than the code's label-equality dedup,
because IoU depends only on the mask.

```
K = 15, per-sentence, 27 pairs
  UNFILTERED (upper bound, known overcount)   125 events, 23 of 27 pairs
  FILTERED   (mask never scored anywhere)      76 events, 17 of 27 pairs
```

Tightening `scored_masks` from "every label in the returned tree" to "only the scored label"
changed **nothing** — 76 either way.

**S1 fires: 76 >> 2.** The event is not the 2/27 seen from inside the search.

## The configuration mismatch — mine, and it changes what the number means

`exp_final_discard.py` runs **per-sentence**. **At K = 15 per-sentence there are 0/27 misses**
(P3). The known harm — unit88 discarding `0.25454105110196174` and returning
`0.2522022213711222` — is from the **flat** run.

So `trained a=0.2 unit88` is absent from the filtered list not because the filter is broken,
but because **in this configuration that formula was scored and the search returned the
optimum.** I read its absence as a false negative; it is not.

**What the number therefore says, and it is stronger than what I was trying to show:**

> **76 events, across 17 of 27 pairs, in a configuration with ZERO misses.** Formulas flagged
> final are popped, never evaluated anywhere in the run, and carry an exact IoU above the
> incumbent at that moment — **while the search still returns the true optimum on every pair.**

That is the event/harm separation measured directly rather than argued: **the event is common
and, here, entirely harmless.** Something better is always found by another route.

## What is still missing before section 1 can be written

The **flat K = 15** count, so the event can be compared against the configuration where the
harm actually occurs, and the **filtered K = 50** count. The unfiltered K = 50 number is 245
events across 15 of 27 pairs; **it is not comparable to a filtered figure** and is not quoted
against one — that would be the Reference field failing, which is the whole reason the filter
was added.

Until both land: **"at most 125 unfiltered, 76 filtered, per-sentence K = 15 only"** — not
"the event is common".

---

# THE EVENT/HARM TABLE — filtered and unfiltered, both partitions, one table

2026-08-02. `src/exp_final_discard.py`. Same metric throughout: a formula flagged FINAL,
popped, whose evaluated mask was **never scored anywhere in the run**, and whose exact IoU
**exceeded the incumbent** at the moment of the pop.

```
partition       K    unfiltered   filtered   pairs w/ event   MISSES   miss pairs in filtered set
--------------------------------------------------------------------------------------------
flat           15        67          45         16 / 27         2      BOTH  (unit88 a=0.2,
                                                                              unit86 a=0.05)
per-sentence   15       125          76         17 / 27         0      n/a
per-sentence   50       245         pending     pending         0      n/a
```

Unfiltered and filtered are never compared across rows; the columns are kept separate for
exactly that reason.

**The flat row is what connects the event to the harm.** In the configuration where the 2
misses occur, the event fires **45 times across 16 pairs, and both miss pairs are among them.**
The other 14 pairs had the event and no harm.

## Section 1 REFRAMED — one finding, two consequences

The report had two observations, and section 1 borrowed the force of section 2. Merged:

> **The search routinely discards complete formulas without evaluating them.** A formula
> flagged final is popped, its exact IoU is never computed anywhere in the run, and that IoU
> exceeds the incumbent at the moment it is discarded. This happens 45 times across 16 of 27
> pairs at K = 15 flat, and 76 times across 17 of 27 at K = 15 per-sentence.
>
> **Usually something better is found later**, by another route, and the discard costs nothing.
> That is why 14 of the 16 affected flat pairs still return the true in-grammar optimum, and
> why all 17 affected per-sentence pairs do.
>
> **In 2 of 27 flat K = 15 runs nothing better was found, and the discarded formula beat the
> returned answer:**
>
> ```
> trained a=0.2 unit88   discarded 0.25454105110196174   returned 0.2522022213711222
> trained a=0.05 unit86  discarded 0.21660649819494585   returned 0.20679723502304148
> ```
>
> **Cause unknown.** Five candidate explanations have been tested and all five failed.

**ONE finding with two consequences, not two findings.**

## The 45 and the 76 are LOWER bounds

Both counts are over **popped** nodes. The `estimate_iou_frontier:389` non-append counter fires
(478,967 flat, 468,645 per-sentence), so **formulas can vanish before ever being popped**, and
those are additional and uncounted. The metric cannot see them.

So: **at least 45 flat, at least 76 per-sentence.** Never "exactly", and never an upper bound.
(The unfiltered figures are upper bounds on a *different* quantity — copies rather than
formulas — and the two must not be mixed.)

## Mechanism hunt REORDERED

**Previously:** hook the two uncovered removal classes, then trace the exit path on the 2 miss
pairs.

**Now:** hook the two uncovered removal classes, then log the exit path over **all 45 flat
events** (and the 76 per-sentence), not the 2 miss pairs.

**Rationale:** 2 events is a case study and requires the miss pairs to be representative, which
nothing establishes. 45 events across 16 pairs is a sample, and **the exit distribution answers
M1/M2/M3 directly** without that assumption.

**M3 still stands, and is now sharper:** if a substantial share of the 45 exit through no
logged path, report that and stop. With a sample rather than two cases, "substantial share" is
measurable rather than a judgement call.

---

# CROSS-CHECK — the event log and the exhaustive oracle AGREE

2026-08-02. For each of the 45 filtered flat events, the discarded IoU compared against the
run's **final returned IoU** for that pair — not the incumbent at discard time.

```
pairs: 2    events: 2

trained a=0.2  unit88   returned 0.2522022213711222
                        discarded 0.25454105110196174  (+0.9274%)  ((dep=ROOT OR dep=nsubj) AND const=NP)
trained a=0.05 unit86   returned 0.20679723502304148
                        discarded 0.21660649819494585  (+4.7434%)  ((dep=ROOT OR dep=nsubj) AND tag=NN)

expected exactly the 2 known miss pairs  ->  MATCHES the oracle
```

**Exactly 2, and exactly the two known miss pairs.** Two independent instruments — an
exhaustive 30,375-formula enumeration and a per-node event log built from entirely different
hooks — agree on which pairs lost and by how much. This is the first cross-instrument
agreement in this section, and it is the reason the remaining 43 events can be read as
harmless rather than as unmeasured.

**A precision defect was caught while building this.** `run_one` returns `best_iou` **rounded
to 4dp** (5e-05 resolution) while the discarded IoUs are full precision, and the gaps under
test are as small as 1e-03. Comparing them directly would have been the third instance of the
precision class in this project. The returned IoU is reconstructed from the integer counts
(`n_inter`, `n_fires`) on both paths instead.

---

# FILTERED K = 50 — the count, and the rate does not predict harm

```
partition       K    unfiltered   filtered   pairs w/ event   MISSES
--------------------------------------------------------------------
flat           15        67          45         16 / 27          2
per-sentence   15       125          76         17 / 27          0
per-sentence   50       245         191         12 / 27          0
```

**R1 holds at K = 50 on the filtered count: 191 events across 12 of 27 pairs.** The event is
**not** a K = 15 artifact. It is the one thing in D6 that survives at K = 50, where the
precondition holds and no miss occurs.

## THE EVENT RATE DOES NOT PREDICT HARM — and this belongs in the report

```
per-sentence K=15    76 events   0 misses
per-sentence K=50   191 events   0 misses
flat         K=15    45 events   2 misses
```

**The configuration with the FEWEST events is the only one with any harm.** A reader given the
raw counts will infer the opposite ordering, so the line has to be stated explicitly rather
than left to inference.

**It also rules out the obvious remedy.** "Discard fewer formulas" is not indicated: the
configurations that discard most are the ones that lose nothing. Whatever makes a discard
harmful is not its frequency, and a fix aimed at the discard rate would be aimed at the wrong
quantity.

What distinguishes the harmful two is unknown — the same "cause unknown" that closes the
five-failed-candidates list.

---

# THE EVENT TABLE WITH DENOMINATORS — and the harm sentence is WRONG as written

2026-08-02. Raw counts across searches of different sizes are not comparable: K = 50 explores
37x the space of K = 15 and pops 5.7x as many nodes.

```
config       events      pops     per-pop      per-formula    MISSES
flat K=15        45    37,096   1.213e-03       5.487e-05        2
part K=15        76    49,599   1.532e-03       9.267e-05        0
part K=50       191   209,964   9.097e-04       6.288e-06        0

per-pop rate ascending      : part K=50  <  flat K=15  <  part K=15
per-formula rate ascending  : part K=50  <  flat K=15  <  part K=15
```

Per-formula, K = 50 is **14.7x rarer** than K = 15 per-sentence while its raw count is 2.5x
higher — the estimate that prompted this check, confirmed.

## The sentence is corrected

**Was:** *"the configuration with the FEWEST events is the only one with any harm."* True on
raw counts, and **misleading on rates**.

**Is:** **flat K = 15 sits in the MIDDLE on both normalised rates** — neither the highest nor
the lowest — **and is the only configuration with harm.** The highest-rate configuration
(part K = 15) has zero harm. The lowest-rate configuration (part K = 50) has zero harm.

> **The event rate does not order the configurations by harm, on any normalisation tried:
> raw count, per node popped, or per enumerated formula.** The single harmful configuration is
> unremarkable on all three.

## The remedy conclusion survives, with a changed reason

**"Discard fewer formulas" is still not indicated**, but not because "the configuration that
discards most is harmless" — that was the raw-count reading. The rate-based reason is stronger:
**moving the rate in either direction lands on a configuration with zero harm.** Nothing in
these three points suggests harm is a function of the rate at all, so a fix targeting the rate
has no measured effect to target.

**Recorded as a correction to my own line**, added to the report one commit earlier and wrong
on the quantity that matters. It is the same shape as instance 15 — right mechanism, wrong
dependent variable — this time caught before the report was sent rather than after.
