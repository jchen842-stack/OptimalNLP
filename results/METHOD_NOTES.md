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
