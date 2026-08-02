# D6 — the measurement-defect class

**Record, not a conclusion.** The instances and the shape they share are set out below;
the entry itself is yours to write.

## Opening — the suite's correctness coverage was one check, and it was defective

`verify/run_all.sh` runs 10 checks. **Nine examine inputs or plumbing** — alignment, padding,
patch no-op, IoU-vs-upstream, masks-vs-raw-`.feats`, two stub checks, model reproduction,
binarisation. **Exactly one examines a result**: the brute-force oracle. That one check was
**instance 2** — it sampled 3 cases out of 27 and had a 14.5% chance of firing.

So the suite's entire correctness coverage was a single check, and that check was defective.
**"11/11, no CANNOT VERIFY" was true, and near-uninformative about correctness.** It said the
right data went in. It did not say the right answer came out, and could not have.

Every result cited here is in `results/METHOD_NOTES.md` with its raw output committed.

## The shape

**In each case the reference stopped matching the measured quantity, and nothing in the
pipeline flagged it.**

Not one of the six is a coding error in the thing being measured. In every one, the *thing
compared against* silently drifted out of correspondence with the thing measured — different
units, different population, different depth, different sample axis, or a sentinel that is not
a number at all — and the comparison still returned a value, so the pipeline reported a
verdict rather than an error.

## The instances

| # | instance | the reference | the measured quantity | how it surfaced |
|---|---|---|---|---|
| 1 | **D5.4 density artifact** | trained arm sampled the full [0.15, 0.85] density band | untrained arm did not | the trained/untrained IoU gap was read as a training effect |
| 2 | **Oracle at 2 units** | 3 cases (2 real units + proxy), all among the 25 that tie | 27 pairs, 2 of which miss | check 10 passed honestly on a sample that could not see the failure |
| 3 | **Expanded-count vs formula-space** | `K*(3K)^(L-1)` formula-space ratios (24.0x, 12.4x) | measured **wall-clock** ratios (83.2x, 36.6x) | the P1 "cancellation" finding, withdrawn |
| 4 | **`nan` sentinel** | `true - x > tol` | `x` was `nan` / `None` / no-solution | 12 no-label runs scored as successes; 10 comparison sites still unguarded |
| 5 | **Treatment-dependent median membership** | all-27 median | timed-out peaks are truncated, and membership moves with the treatment | corrected to a matched set before the L4 run (A1) |
| 15 | **P7 registered against the wrong dependent variable** (assistant) | rationale: pruning -> nodes **expanded** (1.03x, near-invariant as predicted) | registered on **peak frontier** (1.39x), which also counts re-insertions | measuring refinement churn to explain the 1.39x/1.03x split |
| 14 | **One diary consequence hung off a disjunction** | P5 = (a) OR (b), consequence "D5.0 superseded" | (a) and (b) fired in **opposite directions**; the scorer emitted a verdict the data contradicts | L4 landed and the frontier had *grown* while timeouts fell |
| 13 | **"optima biased toward protected signatures"** (assistant) | 47.4% observed | 66.7% space share; Wilson CI on 9/19 is [27.3%, 68.3%] and **contains** it | computed when instance 12 forced a power check one level up. **Parent of instance 12.** |
| 12 | **P9 registered without checking it could fire** | bands separated by 9.0 points | SE 9.7 points at n=26; n~236 needed for 80% power | computed when the baseline correction forced a re-registration |
| 11 | **P9 scored a different subject than it named** | named: length-4 **optima** | scored: **returned formulas**, whose correctness is the open question | caught on review before the L4 file existed; circular as registered |
| 10 | **Two L4 sources, two extraction paths** | flat CSV stores integer counts | partition CSV stores `repr(float)` | logged pre-emptively; no discrepancy produced, but the rounded-`exact_IoU` version of this bit once (17/27 vs 2/27) |
| 9 | **P3 read as a fix without a power check** | "both misses recovered, 0/27" | 0 of **9** vulnerable pairs; p ~ 0.10 under the flat rate | applying the Power field retroactively to a registration that predates the checklist |
| 8 | **Exposure metric tracked one arbitrary prefix** | one representative prefix of the optimum | the optimum is a **set** of 1-3 prefixes | a "never entered frontier" pair lost nothing, which the 3-column shape could not express |
| 7 | **Item 3 blocked on an uncomputed effect size** | asserted "order 7 drops, enough to move a between-arm comparison" | measured ~0.21% against a 64.7% gap | computed when the block was challenged |
| 6 | **Depth-3 reference under a depth-4 search** | `in_grammar_max` enumerated at hardcoded depth 3 | a `PART_LENGTH=4` search | `part=0.146338 > true=0.145326` in the run log |

### Detection power for instance 2, since it is the one with a number

The oracle drew 2 real units. Two of the 27 pairs miss. Probability of drawing at least one
bad pair:

```
1 - C(25,2)/C(27,2) = 1 - 300/351 = 0.1453
```

**14.5%.** The check was not weak by accident of which units were drawn; it had a one-in-seven
chance of ever firing.

## HEADLINE — the defects that agreed with us were the ones nothing caught

**Four of the seven produced a result *favourable* to the hypothesis under test.** Instances
3, 4, 5 and 6 would each have been reported as supporting evidence.

**Every one of those four was caught by an unrelated later measurement disagreeing — never by
a check designed to catch it.** Instance 2 surfaced only by running upstream's own
`beam_optimal`; 3 only by learning what `expanded` actually counts; 6 only by reading a run
log while waiting for something else; 4 only when a scoring pass produced an impossible
verdict.

> **The pipeline had no mechanism for detecting a defect that agreed with us.**

That is the finding. Pre-registration fixes what is predicted; it does not check that the
comparison is well-formed. Every guard in `verify/run_all.sh` tests inputs and plumbing, and
the one check that examines a result (check 10) was itself instance 2. Nothing anywhere
compares a reference against the quantity it is standing in for.

## Instance 7 — the class covers reasoning, not only code

Item 3 was blocked on the assertion that OR-of-same-category-then-narrowing "is not
guaranteed equally frequent across the trained and untrained arms", of "order 7 drops at 100
units, enough to move a between-arm comparison". **No magnitude was computed, and the
computation was available.** When done:

```
worst case, all losses on one arm : 0.0741 x 0.0284 = 0.210%
trained/untrained gap             : (3.41-2.07)/2.07 = 64.7%
                                    ~1 part in 308
```

The confound could not have moved the comparison, and the exposure in fact splits 3 trained /
3 untrained. An experiment was deferred on an unquantified assertion where a derivation was
possible.

Same class as the other six: **the reference (an asserted effect size) never matched the
measured quantity (an actual one), and nothing required it to.** The class is not about code.
It is about comparisons made without checking that the two sides are the same kind of thing —
and a blocking decision is a comparison.

## Not on this list

The 2/27 exact-search misses are **not** an instance. They are a real property of an
inadmissible estimator, measured correctly, and they survive every correction above. Their
attributed cause changed twice; their existence did not.

---

## THE DELIVERABLE — comparison preconditions

The taxonomy above is the diagnosis. This is what it is for. **Every comparison that produces
a verdict carries these six fields, stated before the comparison runs.**

**A prediction can be perfectly discriminating in principle and unfireable in practice. Only
Power separates those.** Discrimination asks whether different outcomes give different
conclusions; Power asks whether the data can tell those outcomes apart — and satisfying
Discrimination is what makes a Power failure invisible.

| field | the question |
|---|---|
| **Quantities** | What is on each side, **with units**? Are they the same kind of thing? |
| **Membership** | Which set is being compared over? Is that set **fixed across the conditions**, or does it move with the treatment? |
| **Reference** | Does the reference's **configuration match the run being scored** — same length, same K, same alpha, same partition, same unit set? |
| **Discrimination** | **What input would change the answer?** A test no input can fail is not a test. **Extended to decisions:** any claim that an effect is large enough or small enough to matter requires a **computed magnitude**, not an assertion. |
| **Power** | **What is the smallest effect this comparison can resolve at the available `n`, and is the predicted effect larger than it?** (Restated from a sampling-only form, which did not reach instances 12 and 13.) |
| **Sentinels** | What **non-numeric values** can reach the comparison — `nan`, `None`, no-solution, truncated-by-timeout — and where are they bucketed? |

### Retro-validation: which field catches each instance

| # | instance | field that fires |
|---|---|---|
| 1 | D5.4 density artifact | **Membership** — the unit set moved between arms |
| 2 | Oracle at 3 cases | **Power** — 14.5% chance of firing |
| 3 | Expanded-count vs formula-space | **Quantities** — a time ratio compared against a count ratio |
| 4 | `nan` sentinel | **Sentinels** |
| 5 | Treatment-dependent median membership | **Membership** — truncated peaks, membership moves with treatment |
| 6 | Depth-3 reference under a depth-4 search | **Reference** — enumeration depth did not match search length |
| 7 | Item 3 blocked on an uncomputed effect size | **Discrimination** (decision form) — no magnitude computed |

**15 of 15 caught. All six fields fire at least once.**

Instance 15 fires on **Quantities**: right mechanism, wrong dependent variable. P7 would have
been SUPPORTED on the quantity its own rationale described. It was committed while writing a
registration whose purpose was to prevent that error, in a file already holding thirteen
logged instances of it — **having the checklist did not make me apply it to what I was
writing at the time.**

Instance 14 fires on **Discrimination**: the field was satisfied for P5's own verdict but
never applied to the *consequence* hung off it. A consequence attached to a disjunction needs
its own Discrimination check, one per disjunct.

Instance 13 is the **parent** of instance 12: P9's bands presupposed that 47.4% and 66.7%
were distinguishable, and they are not at n = 19. The unsupported premise propagated into a
registered prediction and survived three registrations and three joint reviews — the
predictions were scrutinised, the premise was not, because it had stopped being the thing
under test.

Recording 12 and 13 as a lineage rather than two independent entries is itself the finding.

Instance 12 fires on **Power**, and is instance 2 one level up: instance 2 was a *check* that
sampled too few cases; instance 12 is a *prediction* never tested for resolvability at the
available n. P9 was registered three times and reviewed by both participants each time, and
**none of the three asked whether 27 pairs could resolve a 9-point difference** — because
Discrimination was satisfied at every stage. A prediction can be perfectly discriminating in
principle and unfireable in practice; only Power separates those.

Instance 11 fires on **Quantities** primarily (subject substitution: optima vs returned
formulas) and on **Reference** independently (the 47% baseline was computed on optima, so
even the restated version compares returned-L4 against optimal-L3). Two fields catching one
instance is mildly stronger evidence than one.

Instance 10 fires on **Reference**, and is the first logged *before* it caused an error —
two data sources reached by two extraction paths, values agreeing, logged because the same
structure already produced a 17/27-vs-2/27 error once.

Instance 9 fires on **Power**, and is the clearest case of the checklist doing work
retroactively: P3 was registered, run, reported and reviewed by both participants with no
power field, and the sharpened reading — 0 of 9 vulnerable pairs, p ~ 0.10 — only appeared
when the field was applied to it after the fact.

Instance 8 fires on **Membership**: *which set is being compared over?* The exposure metric
never specified whether its subject was one prefix or the set of optimal prefixes, and the
answer changes the figure from 6/27 to 14/27.

**This is evidence FOR the checklist under the commitment made above, not against it.**
Instance 8 was found *by applying an existing field*, not by an eighth failure escaping all
six. The commitment was that an uncaught instance becomes a seventh field; this one was
caught, so no field is added. Recorded explicitly because "the checklist worked" is a claim
that needs the same scrutiny as any other.

### The retro-validation is weak evidence, and saying so is the point

**The checklist was derived from these seven instances.** A 7/7 pass on the set that generated
it is close to tautological — it demonstrates the taxonomy is self-consistent, not that it is
complete. The fields were written by reading the failures backwards.

**The real test is prospective**, on defects not yet seen. No claim is made here that the six
fields are exhaustive. The commitment is the process one: when an eighth instance appears and
none of the six fields catches it, **the gap is recorded as a seventh field rather than forced
into an existing one.** That is the only way this stays a test instead of becoming a ritual.

---

## CONCLUSION — what the checklist is actually evidence for

**The file above reads stronger than the evidence supports. Corrected here.**

All fifteen instances were caught. **None was caught by its author at the time of writing.**

The three most recent are the clearest test, because by then the log was open, maintained, and
being actively cited:

- **Instance 7** — a decision blocked on an uncomputed effect size, while the file already
  documented six instances of exactly that shape.
- **Instance 13** — an unsupported bias claim asserted with no resolvable-effect check, then
  used as the premise for a registered prediction.
- **Instance 15** — a registration written against the wrong dependent variable, *inside a
  registration whose stated purpose was preventing that class of error*.

**The supportable claim is narrow:**

> **The checklist catches defects on review. It does not catch them on authorship.**

Every instance was found by a later measurement disagreeing, or by someone applying the fields
to work already committed. Not one was prevented at the moment of writing, including by an
author who had the list open. A checklist is a review instrument; treating it as a prophylactic
is the same error as treating `11/11` as evidence of correctness.

There is also a class it cannot reach at all: **asserting code behaviour from a name, an
expression, or an intent rather than from the control flow** (four occurrences this session,
recorded in `results/METHOD_NOTES.md`). No comparison precondition addresses it. The remedy is
to read the flow, and it is cheap — the four cost between five and twenty-five lines of source
each to settle.
