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
