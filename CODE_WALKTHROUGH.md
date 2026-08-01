# Code walkthrough — mechanism, not API

Written to be defensible in a meeting. Every claim has a `file:line` you can point at.

> **Written 2026-07-31. Deleted in `733ff0c`, restored 2026-08-01.**
>
> **Citation status, measured 2026-08-01 by `verify/check_walkthrough_citations.py`
> (output: `results/walkthrough_citations.txt`):**
>
> | | |
> |---|---|
> | **39 of 99** | verified at the cited line — the range still holds the identifier the prose names |
> | **35 of 99** | resolve, but the content has **moved** to a different line (new line reported by the checker) |
> | **25 of 99** | **unverifiable by any tooling** — no identifier is extractable from the surrounding prose |
> | 0 | missing files |
>
> **Do not read this document as "every claim has a `file:line` you can point at" without
> that table.** Fewer than half the citations are confirmed to point where they say. The
> 25 unverifiable ones are a permanent blind spot in the checker, not a backlog: they can
> only be settled by a human reading each one.
>
> **Two corrections to the original header, which read "All 98 `file:line` citations
> re-verified against commit `f1bace0`":**
>
> 1. **The count was 98; there are 99.** The self-count was wrong.
> 2. **"Upstream citations are pinned at `70805299` and do not drift" is misleading as
>    written.** Upstream citations resolve against the **patched** tree, not the pinned one —
>    `patches/0001-frontier-beam-fallback.patch` inserts ~21 lines into `optimal.py`, so the
>    same symbol sits at different line numbers in the two trees (`expand_node` is at
>    `optimal.py:537` patched, `:516` pinned). They are stable *given the patch*, which is a
>    weaker claim than "pinned and do not drift".
>
> Citations below have **not** been hand-fixed. Drift is reported, not silently repaired, so
> re-run the checker after any edit to `src/`.

**Path convention.** Two trees are involved:

| prefix | tree | who wrote it |
|---|---|---|
| `src/`, `verify/`, `tests/` | this repo (`optimalce-nlp`) | us |
| `compositional/`, `utils/` | upstream `optimal-compositional-explanations` @ `70805299`, plus `patches/0001-frontier-beam-fallback.patch` | upstream, except the patch |
| `models.py` | `neuron-explanations-nli/nli/code` (Mu & Andreas 2020) | upstream |

**Everything below was re-run, not recalled.** The unit-413 numbers in §1 come from an
actual execution on 2026-07-31 (`~/miniconda3/envs/compexp/bin/python`), and they match the
recorded row in `results/beam_vs_exact_L3_K15.csv:2` exactly.

---

## 1. One run, end to end: trained unit 413, alpha 0.2, length 3

The command:

```sh
python src/real_token_search.py --arms all --K 15 --lengths 3 --min_support 5 \
    --max_sents 2000 --neuron real --beam_list none \
    --acts results/acts2k_trained_a0.2.npz --unit_ids 413 \
    --dmin 0 --dmax 1 --min_fire 200 --cap 200000 --time_budget 600
```

The answer it returns:

```
formula   ((const=NP AND (NOT const=VP)) AND (NOT const=PP))
best_iou  0.4689     visited 65   expanded 260   estimated 2682   peak_frontier 428   0.12 s
```

Sanity: the formula fires on 5,519 tokens, the unit on 4,840, they overlap on 3,307.
3307 / (4840 + 5519 − 3307) = 0.46894. That is the IoU.

### Stage A — `.feats` file → concept matrix

**What a row of `.feats` is.** One line = one sentence. Whitespace splits it into tokens;
each token is seven `|`-separated fields in the order fixed by the header
(`src/real_token_masks.py:40`):

```
text | lemma | tag | dep | ent | synset | const
Two  | two   | CD  | nummod | CARDINAL | two.n.01 | NP
```

`const` is the only multi-valued field — `;`-separated constituent labels
(`src/real_token_masks.py:41`). Fields can be empty; an empty field contributes *nothing*,
it is not the concept "no entity" (`src/real_token_masks.py:71-72`). That choice is what
keeps `uncoverable` meaningful downstream.

**How one line becomes K mask entries.** `load_sentences` (`src/real_token_masks.py:44`)
turns the line into a list of `(text, {category: [values]})` pairs; the parse itself is
`src/real_token_masks.py:65-74`. Each token becomes one **column** of the concept matrix,
and in that column between 0 and 7 of the K rows are set — one per (category, value) pair
the token carries that survived concept selection. So "one line → K mask entries" is really
*one token → a K-bit column*, and a 10-token sentence contributes 10 columns.

Concretely, the first sentence of the dev set is
`Two women are embracing while holding to go packages .` The token `Two` carries
`lemma=two, tag=CD, dep=nummod, ent=CARDINAL, synset=two.n.01, const=NP`. Of the fifteen
concepts admitted at K=15, only `const=NP` is one of them, so `Two`'s column has exactly
one bit set. `embracing` carries `const=['VP','VP']` — duplicated in the raw file — and its
column has `const=VP` set (see §4 for why the duplicate is harmless).

**Which K concepts.** `select_concepts` (`src/real_token_masks.py:84`) counts every
(category, value) pair across all tokens, drops anything below `min_support=5`
(`:96`), sorts by descending frequency with the pair itself as tie-break (`:97`), and takes
the first K (`:98`). At K=15 over 2,000 sentences that is:

```
0 const=NP   1 const=VP   2 const=PP    3 tag=NN    4 tag=DT
5 dep=det    6 tag=IN     7 dep=prep    8 dep=pobj  9 lemma=a
10 synset=angstrom.n.01   11 dep=punct  12 dep=ROOT 13 dep=nsubj  14 lemma=.
```

Note the ranking is **global across categories**, not per-category — that is why the three
constituent labels crowd the top and `ent` gets zero slots. See §4 for the alternative.

**The matrix.** `build_dense` (`src/real_token_masks.py:101`) allocates a
`(K, M)` boolean array (`:104`) and sets `dense[k, m] = True` for each concept a token
carries (`:106-110`).

> **Data now:** `dense`, `numpy.ndarray`, shape `(15, 24199)`, dtype `bool`.
> Rows are concepts, columns are tokens, in reading order. 2,000 sentences → 24,199 tokens.

**Overlap diagnostics** — `diagnostics` (`src/real_token_masks.py:114`) computes
`sum_elements` = concepts per token (`:118`) and from it:

```
mean_overlap 3.189   common_frac 0.858   unique_frac_all 0.139
disjoint_pairs 108/210 (0.514)   max_overlap 7   coverage 0.977
```

`common_frac = 0.858` is the number the whole thesis rests on: 86% of covered tokens sit in
more than one concept. Hold on to it — §2 explains why it wrecks the heuristic.

### Stage B — SNLI sentences → encoder → activations → binary mask

`real_activations.py` was run once, earlier, to produce `results/acts2k_trained_a0.2.npz`.
The search just loads that file. Here is what produced it.

**Same parser, deliberately.** `real_activations.main` calls
`rtm.load_sentences` (`src/real_activations.py:184`) — the *identical* function the mask
side uses. That is the only reason the two axes cannot drift (`src/real_token_masks.py:50-53`).

**Vocabulary.** For the trained arm the `stoi` comes out of the checkpoint, not from the
corpus (`src/real_activations.py:191-192`). This matters: rebuilding a vocabulary here would
map every token to the wrong embedding row and produce plausible-looking activations with no
relationship to the model — a silent failure. The comment at `:187-189` says so. OOV rate is
8.8% (2,137 / 24,199), reported at `:197-198`, all mapped to UNK.

**Batching and padding.** `extract_states` (`src/real_activations.py:66`):

- batches of 64 sentences (`:93-94`);
- `maxlen` = longest sentence in *that* batch (`:96`);
- an `(maxlen, batch)` long tensor pre-filled with **1** (`:98`) — 1 because
  `TextEncoder`'s embedding is built with `padding_idx=1` (`models.py:244`);
- real ids written in at `[t, b]` (`:100-101`), so the layout is time-major.

The forward call is `enc.get_states(ids, torch.tensor(lengths))` (`:102`). Inside,
`models.py:259` does `pack_padded_sequence(semb, slen, enforce_sorted=False)` — torch sorts
by length internally and `pad_packed_sequence` (`models.py:262`) restores the original batch
order. So no un-permutation is needed on our side.

**Where padding is stripped:** `src/real_activations.py:104`,
`out.append(states[:n, b, :].numpy())` — `n` is that sentence's true length, so rows
`n..maxlen` are dropped and never reach the token axis.

> Do not take that on faith. `verify/check_padding.py` runs the shortest (3 tokens) and
> longest (53 tokens) sentences together in one batch, then each alone in a batch of 1 where
> no padding exists, and compares: max difference 6.5e-8. It also includes a negative control
> (deliberately taking the wrong slice) which comes out at 4.8e-1, proving the test can fail.

**Order is asserted, not assumed.** `row_tokens` collects the surface token behind every
output row (`:105-106`), and `verify_alignment` (`:123`) compares 50 random rows against what
the mask side has at the same index (`:147-154`). A row-count check alone cannot detect a
reordering — `:140` says exactly that.

> **Data now:** `states`, shape `(24199, 512)`, dtype float32. One row per token, one column
> per LSTM hidden unit.

**Binarization.** `binarize` (`src/real_activations.py:110`) with `alpha=0.2` takes the
per-unit 80th percentile (`:119`, `np.quantile(states, 1-alpha, axis=0)` — `axis=0` means
*per unit*, this is not a global threshold), compares, and transposes (`:120`).

> **Data now:** `acts`, shape `(512, 24199)`, dtype bool. Row 413 is our neuron; it fires on
> 4,840 tokens (density 0.20001 — alpha 0.2 delivers what it promises).

### Stage C — both → the search

`load_real_neurons` (`src/real_token_search.py:60`) opens the `.npz`, filters units to a
density band and a minimum firing count (`:73-74`), and with `--unit_ids 413` skips random
selection entirely and takes exactly that unit (`:86-91`). A length check at `:356` refuses
to proceed if the activation file and the mask token count disagree.

`run_one` (`src/real_token_search.py:220`) converts everything into the shapes upstream
expects:

| line | what it builds | type / shape |
|---|---|---|
| `:231` | `masks` — one sparse row per concept | `list[scipy.csr_matrix]`, 15 × `(1, 24199)` |
| `:232` | `common, unique, uncoverable` via `compute_quantities` | 3 × `torch.bool`, `(1, 24199)` |
| `:233` | `disjoint_info` | `torch.bool`, `(15, 15)` |
| `:234` | `bitmaps` — the neuron | `torch.bool`, `(1, 24199)` |

`compute_quantities` lives at `src/synthetic_overlap_sweep.py:202` and deliberately
replicates upstream's `utils/mask_utils.py:160-162` (see §2's last subsection).

Three module globals are then swapped into upstream:

```
src/real_token_search.py:237   optimal.heapq = probe          # frontier instrumentation
src/real_token_search.py:253   optimal.expand_node = capped   # only if --expand_budget
src/real_token_search.py:254   optimal.MAX_FRONTIER_SIZE = beam_cap   # None here (exact)
```

**The handoff line is `src/real_token_search.py:265.`** That is where our code stops and
upstream's begins:

```python
best_label, best_iou, visited, expanded, estimated = optimal.compute_optimal_explanations(
    bitmaps=bitmaps, masks=masks, masks_info=(common, unique, uncoverable),
    disjoint_info=disjoint_info, config=cfg,
)
```

**What crosses it, in:** the four objects in the table above, plus `cfg`, a `StubConfig`
(`src/synthetic_overlap_sweep.py:158`) that answers only the three accessors upstream
actually calls — `get_length()` → 3, `get_mask_shape()` → `(1, 24199)`, `get_device()` → cpu.
The vision config's file paths, dataset names and caches are never touched.

**What crosses it, out:** `(best_label, best_iou, visited, expanded, estimated)`.
`best_label` is a `compositional.formula` tree object, not a string;
`best_iou` a float; the other three ints (`compositional/optimal.py:921`).

Everything is restored in a `finally` (`src/real_token_search.py:273-277`), including
`MAX_FRONTIER_SIZE = None`, so a swept run cannot leak a beam cap into the next.

`formula_stats` (`src/real_token_search.py:193`) then renders the tree and re-evaluates it
independently (`eval_formula`, `:126`) to get coverage and intersection counts. Those
`formula*` columns are permanent for a reason stated at `:280-281`: a score without the label
it came from is what let a near-universal winner sit undetected across two diary entries.

**Summary of the whole pipeline as shapes:**

```
snli_1.0_dev.feats  → 2000 sentences → 24,199 (text, feats) pairs → dense (15, 24199) bool
same sentences → ids (maxlen, 64) long → states (24199, 512) f32 → acts (512, 24199) bool
                                                                  → row 413, 4840 bits set
both → optimal.compute_optimal_explanations → ((NP AND NOT VP) AND NOT PP), IoU 0.4689
```

---

## 2. How the search actually works

All of this is `compositional/optimal.py`, upstream code.

### What is on the frontier

A **node is a 5-tuple**, built at `optimal.py:392-400`:

```
( -max_iou_estimate,   # float. NEGATED — see ordering below
  next_op,             # "OR" | "AND" | "NOT" | "INDIVIDUAL" | None
  label,               # a formula tree (F.Leaf / F.And / F.Or)
  paths_to_expand,     # [[], or_paths, and_paths, and_not_paths] — which op sequences remain legal
  heuristic )          # "sum" or "sample" — which estimator produced field 0
```

Index constants are at `utils/constants.py:22-25`.

Crucial and easy to miss: **a node is not a formula, it is a (formula, next-operator) pair.**
The same label appears on the frontier several times, once per operator it could still take.
`estimate_paths_iou` returns four path-nodes per label (`path_heuristic.py:665-670`), and each
one with a positive ceiling is pushed separately (`optimal.py:392`). That is why the initial
frontier for 15 concepts is 43 nodes, not 15.

`paths_to_expand` is the bookkeeping that keeps the search from wasting slots: it records
which operator *sequences* are still reachable inside the remaining length budget. It is
recomputed on every expansion at `optimal.py:597-626`.

### How the frontier is ordered

It is a plain Python `heapq` min-heap over the tuple. Field 0 is the **negated** maximum
estimated IoU (`optimal.py:392`), so the smallest tuple is the **highest ceiling** — best-first
on an admissible upper bound. Ties fall through to field 1 (the `next_op` string, so
`"AND" < "INDIVIDUAL" < "NOT" < "OR"` alphabetically) and then field 2 via `Formula.__lt__`
(`compositional/formula.py:118`, `:247`).

Here is the first dozen pops of the actual unit-413 run:

```
ceiling  next_op  label        frontier size after pop
0.9655   AND      const=NP     42
0.9655   AND      const=VP     63
0.9655   AND      dep=nsubj    74
0.9655   OR       const=NP     73
0.9655   OR       const=VP    124
0.9655   OR       dep=nsubj   151
0.9649   AND      dep=ROOT    150
0.9649   OR       dep=ROOT    149
0.9633   NOT      dep=ROOT    134
0.9622   AND      tag=NN      137
...
```

**Look at that column of 0.9655s.** The true best achievable IoU is 0.4689. The ceiling for
`const=NP` (whose own IoU is 0.2185) and for `const=VP` (0.0598) and for `dep=nsubj` (0.2060)
are *identical to four decimals*. The heuristic is not distinguishing a good branch from a
bad one; it is telling the search that everything is potentially near-perfect. That is the
frontier explosion in one screenful — and it is why the frontier grows from 43 to 428 in a
problem with only 15 concepts and 3 slots. §2's heuristic subsection explains where the
looseness comes from.

### One pop-expand-push cycle

Loop body is `optimal.py:693-916`. In order:

1. **Pop** — `optimal.py:695`. Highest ceiling wins.
2. **Stale check** — `:697-700`. If `-e_node < minimum_threshold`, the node's ceiling has
   fallen below a value we have already *achieved*, so it is discarded. It is not removed
   eagerly when the threshold rises; it is left to rot and dropped on pop.
3. **Estimator upgrade** — `:708-730`. If the node was scored by the cheap `"sum"` heuristic,
   rescore with the tighter `"sample"` heuristic. If the tighter score is *lower*, the node is
   **re-pushed with the new score** (`:725-728`) and the cycle restarts. It is not expanded on
   this visit. This is a lazy-refinement pattern: pay for the expensive bound only for nodes
   that actually reach the front.
4. **Distributive re-check** — `:733-768`. For labels of length ≥ 3, `apply_distributive_property`
   (`:102`) rewrites e.g. `(A OR B) AND NOT C` into `(A AND NOT C) OR (B AND NOT C)`, which the
   pairwise estimator can bound more tightly. If the rewrite lowers the ceiling, the node is
   re-pushed and the cycle restarts.
   **Subtlety worth knowing:** the *rewritten* label is never pushed. Line `:758` pushes
   `label_node`, the original. The transform exists purely to tighten a number.
5. **Recent-node dedupe** — `:771-779`. A small buffer of nodes seen at the current ceiling
   value, to avoid re-expanding a node that keeps returning at the same score.
6. **Terminal node** — `:782-892`. If `next_op == "INDIVIDUAL"`, the formula is done. Its
   actual mask is materialised (`:786`, `mask_utils.get_formula_mask_and_tree`), its true IoU
   computed (`:792-799`), best-so-far updated (`:807-812` — ties broken toward the *shorter*
   formula), and if it beats `minimum_threshold` the whole frontier is pruned (`:813-817`).
   Then **propagation** (`:819-889`): every intermediate subtree computed along the way gets
   its exact quantities recorded and fed back so that other frontier nodes containing that
   subtree can be re-bounded with exact rather than estimated numbers
   (`update_frontier_by_ancestors`, `:14`).
7. **Expand** — `:895-897`. `expand_node` generates children. `:898-899` bumps the counters.
8. **Push** — `:902-914`. `update_frontier` scores the children, merges them into the heap
   (`:508-509`), and applies the beam cap (`:516`).

For unit 413 the counters were: 260 expansions, 2,682 children generated, 65 formulas
actually materialised and scored. The gap between 2,682 and 65 is the pruning doing its job;
the gap between 43 and a peak of 428 is the pruning failing to do enough of it.

### How formulas are built — `expand_node`

`optimal.py:537`. This is the part reviewers ask about, so here it is precisely.

```python
_, next_op, label, paths_to_expand, _ = frontier_node       # :548
for candidate_term in candidate_labels:                     # :554  candidate_labels are all Leafs
    if candidate_term.val in label.get_vals(): continue     # :556  no concept twice
    ... ordering constraints ...                            # :558-574
    if   next_op == "OR":  candidate_formula = F.Or(label, candidate_term)          # :577-578
    elif next_op == "AND": candidate_formula = F.And(label, candidate_term)         # :579-580
    elif next_op == "NOT": candidate_formula = F.And(label, F.Not(candidate_term))  # :581-582
```

Three structural facts follow, and they bound the entire hypothesis space:

**(a) The new term is always a bare leaf, always attached on the right.** `candidate_labels`
is built once as `[F.Leaf(c) for c in range(len(masks))]` (`optimal.py:654`) and never
extended. So every formula is a **left-deep chain**: the right child of every operator is a
leaf or a NOT-leaf, forever.

**(b) NOT only ever appears as `AND NOT`.** Line `:582` is the only construction site, and
`F.Not.__init__` asserts its argument is a `Leaf` (`compositional/formula.py:229`). There is
no `OR NOT`, and no negation of a compound.

**(c) Length counts leaves, not operators.** `Leaf.__len__` returns 1
(`formula.py:90`), `BinaryNode.__len__` sums its children (`formula.py:189`), and
`UnaryNode.__len__` forwards to its value (`formula.py:152`). So NOT is **free** — it costs
no length budget. `length=3` means "at most three concepts", and `((NP AND NOT VP) AND NOT PP)`
is a length-3 formula with two negations in it.

**Constructible** (all real winners from `results/`):

```
const=NP                                                  len 1
(NOT dep=pobj)                                            len 1   ← unit413 @ alpha 0.5
((const=NP AND (NOT const=VP)) AND (NOT const=PP))        len 3   ← unit413 @ alpha 0.2
(((dep=det OR dep=ROOT) AND (NOT const=VP)) OR dep=nsubj) len 4
```

**Not constructible, at any length:**

```
(const=NP OR tag=NN) AND (dep=det OR tag=DT)   — right operand is a compound; :578/:580 only
                                                 ever put a Leaf there
NOT (const=NP OR const=VP)                     — formula.py:229 asserts Not wraps a Leaf
const=NP OR (NOT const=VP)                     — :582 is the only NOT site and it hard-codes AND
const=NP AND const=NP                          — :556 blocks a repeated concept
```

The transient exception: `apply_distributive_property` (`:102-140`) *can* build a formula
whose operands are compound — but as established above, that object is used for scoring only
and never enters the frontier.

The ordering constraints at `:558-574` are dedup, not restriction: since `A OR B` ≡ `B OR A`,
only the ordered version is generated. `:559-565` handles the leaf case, `:566-574` compares
against the last operator and value to avoid regenerating equivalents one level up. The
comment at `:567-568` is honest that it only looks back two operators — "looking beyond the
last two would be costly" — so some logically equivalent formulas *are* generated twice. The
`visited` list at `:783` catches those at scoring time.

### The heuristic's quantities, in tokens

Fix a neuron mask **N** (tokens where unit 413 fires) and a candidate formula's mask **L**.
Every token is first classified once, globally, by *how many concepts cover it*
(`utils/mask_utils.py:160-162`):

| name | definition | tokens here |
|---|---|---|
| **unique** | covered by exactly one of the 15 concepts | 3,369 (13.9%) |
| **common** | covered by two or more | 20,280 (83.8%) |
| **uncoverable** | covered by none | 550 (2.3%) |

Then each of `unique` and `common` is crossed with three relationships between L and N
(`utils/optimal_utils.py:142-153`) — the **I / E** decomposition:

| quantity | in tokens | code |
|---|---|---|
| `unique_intersection` | tokens in exactly one concept, where the formula fires **and** the unit fires — a correct hit | `optimal_utils.py:143-145` |
| `common_intersection` | same, but the token is in several concepts | `:146-148` |
| `unique_extras` | tokens in exactly one concept, where the formula fires but the unit does **not** — a false positive | `:151` |
| `common_extras` | same, multi-concept token | `:150` |
| `unique_uncovered` | tokens the unit fires on that the formula misses, single-concept | `:153` |
| `common_uncovered` | same, multi-concept | `:152` |

IoU = intersection / union, so the ceiling is built as
**largest possible intersection ÷ smallest possible union** (`optimal_utils.py:512-520`;
the generic form is `path_heuristic.py:47`, `max_iou = max_intersection / min_union`). The
floor inverts it (`optimal_utils.py:470-472`).

**Why split by unique/common at all — this is the load-bearing idea.**
`utils/optimal_utils.py:40-42` states it outright: *for the unique quantities there is no max
and min, since they can be computed exactly.*

The reason, in tokens: take a token covered by exactly one concept. Whatever you `OR` onto
the formula next, that token's status is fully determined — no other admitted concept touches
it, so no other term can newly cover it. Its contribution to the future intersection is a
known number, not a range. Now take a token covered by five concepts. `OR`-ing on any of the
other four flips it into the formula's mask; the estimator cannot know which term will be
chosen, so it must bracket the outcome: `(max, min)` (`optimal_utils.py:75-97` builds those
tuples, degenerate for unique, genuinely wide for common).

`extract_max_min_quantity_improvement` (`optimal.py:143`) then sums the top-`length` and
bottom-`length` per-concept values (`:186-189`) to get, per remaining slot, "the most this
could possibly improve" and "the least". Those brackets are clipped by hard physical limits
(`:203-208`) — you cannot intersect more than the neuron fires, cannot add more extras than
there is space.

**So:** when unique dominates (vision — one pixel, one segmentation label), the brackets
collapse and the ceiling is nearly the true value, which makes best-first search almost
greedy and cheap. When common dominates — **85.8% of covered tokens here** — nearly every improvement term is a
wide interval, the top-`length` sums are near their physical limits for every candidate, and
every ceiling converges to roughly the same near-1.0 value. That is the 0.9655 column above.
Best-first search with a bound that does not discriminate degenerates toward breadth-first,
and breadth-first over 15 concepts × 3 operators × 3 slots is where the memory goes.

That is the whole finding. The overlap statistic in Stage A and the flat ceiling column in
Stage C are the same fact seen from two ends.

### Which comparison makes pruning safe

**`compositional/optimal.py:697`:**

```python
if -e_node < minimum_threshold:
    # Unuseful node, skip it
```

and its bulk equivalent, `reduce_frontier`'s **`optimal.py:444`**, `if -iou >= threshold`.

Soundness rests on exactly two properties:

- `-e_node` is an **admissible upper bound**: an over-estimated intersection over an
  under-estimated union (`optimal_utils.py:512-520`), clipped by `neuron_coverable_sum` so it
  can never exceed what is physically achievable (`:512-513`). Nothing reachable by extending
  this node can score higher.
- `minimum_threshold` is a value that is **actually attainable** — either a realised IoU of a
  fully-evaluated formula (`optimal.py:813-814`, and the same for propagated ancestors at
  `:847-848`), or a proven *lower* bound from the heuristic's min branch
  (`path_heuristic.py:380`, `optimal.py:761-763`).

Upper bound below attainable value ⟹ the node cannot contain the optimum ⟹ dropping it is
safe. Everything else in the file — the sum→sample upgrade, the distributive rewrite, the
ancestor propagation — exists to *lower* ceilings and *raise* the threshold so that this one
comparison fires more often. Upstream even asserts the invariant explicitly at
`optimal.py:401-404`: a node whose max IoU came out below the threshold raises `ValueError`,
"This should not happen."

> **What "the optimum" means here.** Optimal *within the method's formula grammar*.
> `expand_node` only ever appends a bare literal (`optimal.py:554-582`), so formulas are
> left-deep, negation appears only as AND-NOT, and there is **no `OR NOT`**. A brute-force
> oracle over the unrestricted formula space confirms the search attains the in-grammar
> optimum exactly — and measures what the grammar cannot express: **+0.0000% at length 3**
> on all three cases tested, **+0.1586% at length 4** on one unit
> (`tests/test_bruteforce_oracle.py`, `results/oracle_L{3,4}.txt`, `VERIFICATION.md` check 10).
> No reported number changes; the wording does.

**And this is precisely what the beam cap gives up.** `_apply_beam_cap` drops nodes whose
ceiling is still above the threshold. That is not a sound prune, it is a heuristic one — the
result is a lower bound on the optimum, not the optimum. `results/beam_vs_exact_L3_K15.csv`
measures the cost: 20 of 27 pairs agree exactly.

### `mask_utils.py:160` — `common = sum_elements > 1`

**What it computes.** `sum_elements` is built at `mask_utils.py:153-157` by summing all K
concept masks elementwise. It is the per-token count of admitted concepts. Line 160 marks
every token covered by **two or more**; `:161` marks exactly one (`unique`); `:162` zero
(`uncoverable`). Three disjoint boolean masks over the token axis, partitioning it.

**A necessary caveat: we do not execute that line.** `get_dataset_quantities` pickles to a
`config.get_info_dir()`, and the config is the vision one. `src/synthetic_overlap_sweep.py:202-208`
reimplements it, line for line, and hands the result in as `masks_info` at the boundary
(`src/real_token_search.py:232` → `:266`). The docstring at `:203` says so. If a reviewer asks
"are you sure it matches?" — compare `mask_utils.py:160-162` against
`synthetic_overlap_sweep.py:205-207`; they are the same three predicates on the same sum.

**Who reads it downstream.** It enters upstream at
`compute_optimal_explanations(masks_info=...)` (`optimal.py:924`) and fans out to four places
that make decisions:

1. **`get_neuron_quantities`** (`optimal_utils.py:304-311`) — splits the neuron's own firing
   tokens into common/unique counts, and the non-firing space into common/unique extras. These
   become the clipping limits at `optimal.py:268-315`.
2. **`compute_quantities_vector`** (`optimal_utils.py:143-153`) — the six per-concept
   quantities of the previous subsection. Called for every concept once up front
   (`:362-371`) and for every terminal formula during the search (`:545-552`).
3. **`compute_max_iou_from_label_info` / `compute_min_iou_from_label_info`**
   (`optimal_utils.py:477` / `:431`) — the actual ceiling and floor, hence the heap key, hence
   the pop order, hence the prune at `optimal.py:697`.
4. **`extract_max_min_quantity_improvement`** (`optimal.py:143`) — the per-slot improvement
   brackets whose width is the whole mechanism.

So: line 160 decides which tokens get *bracketed* estimates rather than *exact* ones, and
therefore how loose every bound in the search is. It is one comparison, and it is the hinge.

---

## 3. What the patch does

`patches/0001-frontier-beam-fallback.patch`, against upstream `70805299`. Three hunks, 25
added lines, zero deleted apart from one changed `return`.

```diff
+MAX_FRONTIER_SIZE = None
+
+def _apply_beam_cap(frontier):
+    if MAX_FRONTIER_SIZE is None or len(frontier) <= MAX_FRONTIER_SIZE:
+        return frontier
+    kept = heapq.nsmallest(MAX_FRONTIER_SIZE, frontier, key=lambda node: node[0])
+    heapq.heapify(kept)
+    return kept

 def reduce_frontier(frontier, threshold):
     ...
     heapq.heapify(reduced_frontier)
-    return reduced_frontier
+    return _apply_beam_cap(reduced_frontier)

 def update_frontier(...):
     ...
+    # Enforce the beam cap after growth (no-op when MAX_FRONTIER_SIZE is None).
+    sorted_frontier = _apply_beam_cap(sorted_frontier)
     return sorted_frontier, global_min_threshold
```

Live at `compositional/optimal.py:412-428`, `:447`, `:515-516`.

**What `_apply_beam_cap` does to the heap.** `heapq.nsmallest(N, frontier, key=node[0])`
selects the N nodes with the smallest first field — which, since field 0 is the negated IoU
estimate, means the N **highest ceilings** (`:426`). `heapify` then restores the heap
invariant on the survivors (`:427`), because `nsmallest` returns a sorted list, and a sorted
list happens to satisfy the heap property but the code should not depend on that.

The `key=` is deliberate and worth defending: without it, `nsmallest` would compare whole
tuples, so nodes with equal ceilings would be ordered by `next_op` and then by formula
identity. Selection would then silently depend on which operator sorts first alphabetically.
With `key=lambda node: node[0]`, ties are broken by `nsmallest`'s own stable ordering over
input position, and the *selection criterion* is exactly and only the ceiling. The docstring
at `:422-423` states this.

### Why both call sites

`update_frontier` is the one that is strictly **required**: it is the only place the frontier
grows. `:506-509` pushes every newly generated child onto the past frontier, one at a time,
with no size check. After a wide expansion the merged heap can be far larger than the cap, so
the cap has to be applied to the merged result (`:516`).

`reduce_frontier` is capped for two reasons, one weak and one real:

- *Weak (invariant hygiene):* `reduce_frontier` is called directly from `perform_search` at
  `:764`, `:815`, `:849`, and `:888`. Capping there means **every function that returns a
  frontier list returns one that respects the cap**, so no future edit can reintroduce an
  uncapped path. On its own this is belt-and-braces: `reduce_frontier` only ever removes
  nodes, so it cannot breach a cap that already held.
- *Real (and it changes behaviour):* `reduce_frontier` is also called from *inside*
  `estimate_iou_frontier` at `:408`, where the list being reduced is the **new nodes only**,
  before they merge with the past frontier. Capping there truncates the fresh children against
  *each other* rather than against the incumbent frontier, and it bounds that intermediate
  list — which for a wide expansion can be thousands of nodes — before the merge allocates.

Be straight about this in the meeting: the second effect means the beam is applied twice with
different reference sets (new-nodes-only, then merged), which is a slightly different beam
semantics than a single post-merge truncation. It bounds memory in both places, which is what
it was for, but it is not the textbook single-truncation beam. `results/beam_sweep.csv` and
`results/beam_vs_exact_*.csv` measure the end-to-end consequence empirically rather than
arguing about it.

### Why `MAX_FRONTIER_SIZE = None` is a genuine no-op

Not "usually harmless" — structurally identical. Three arguments, in increasing strength:

1. **Short-circuit before any mutation.** `optimal.py:424-425`:
   ```python
   if MAX_FRONTIER_SIZE is None or len(frontier) <= MAX_FRONTIER_SIZE:
       return frontier
   ```
   `None` is checked *first*, and the function returns **the same list object**. No copy, no
   re-heapify, no re-ordering, no comparison of node tuples. Both call sites become `x = x`.
   Contrast a plausible alternative — `heapq.nsmallest(cap or len(frontier), ...)` — which
   would rebuild the list every call and could reorder equal-ceiling ties, silently changing
   which of several tied nodes gets expanded first and therefore which of several tied optima
   is reported. This implementation cannot do that.

2. **`None` is the module default** (`:417`), and our harness restores it in a `finally`
   (`src/real_token_search.py:277`, `src/synthetic_overlap_sweep.py:262`). A run that forgets
   to set it gets exact search, not a silently capped one.

3. **Measured, not argued.** `verify/check_patch_noop.py` runs one identical configuration
   (unit 413, K=15, length **4** — the harder case) against whichever tree
   `OPTIMALCE_UPSTREAM` points at, and prints the winning formula, `best_iou` **in hex float**
   (`check_patch_noop.py:53` — hex so a last-bit difference cannot hide behind `%.4f`),
   `visited`, `expanded`, `estimated`, and `peak_frontier`. Point it at a clean checkout, point
   it at the patched one, diff. `VERIFICATION.md` check 3 records the PASS.

---

## 4. The parts I wrote — mechanism, choices, and luck

### `src/real_token_masks.py` (204 lines)

**The lines that do the work:**

- **`:65-74`** — the parse. `tok.split("|")` into 7 fields; `cat in MULTI` decides whether the
  value is `;`-split into a list or wrapped in a singleton list. Everything downstream is a
  consequence of this shape.
- **`:96-98`** — concept selection. `min_support` filter, then frequency sort, then top-K.
  This single expression determines the entire hypothesis space of every search in the repo.
- **`:104-110`** — `build_dense`. The `(K, M)` bool matrix; the double loop over
  `feats.items()` and its values with an `index.get` that silently ignores unselected concepts.
- **`:118-120`** — `sum_elements` and `common_frac`. The overlap number the whole thesis rests
  on, computed in three lines.

**Choices that could have gone another way:**

| choice | where | the alternative |
|---|---|---|
| empty field → no concept | `:71-72` | treat `""` as the concept "no entity". Would collapse `uncoverable` to ~0 and inflate coverage to 100%, changing every bound in the heuristic. |
| frequency ranking is **global** across categories | `:97` | per-category quota. Global gives K=15 = 3 const / 6 dep / 3 tag / 2 lemma / 1 synset / **0 ent** — entities never get tested. A quota would give a more balanced but less frequency-motivated vocabulary. |
| malformed token silently skipped | `:68` | raise. `continue` means a corrupted file degrades quietly. Defensible for a fixed checked-in artifact; not defensible if the parser is reused on new annotations. |
| `min_support=5` | `:157` | any value. Drops the hapax tail that could never form a useful explanation, but it is a free parameter and it was not swept. |
| `diagonal = 1` before counting disjoint pairs | `:126` | a concept is never "disjoint with itself" — otherwise `disjoint_pairs` would be inflated by K. |

**Things that work by luck rather than by design — say these before a reviewer finds them:**

1. **Duplicate constituent labels.** The raw file has tokens like `embracing|…|VP;VP` — the
   same label twice. `:73` splits it into `['VP','VP']` and `:110` sets the same matrix cell
   `True` twice. It is correct only because boolean assignment is idempotent. Nothing
   deduplicates, and nothing checks. If `build_dense` were ever changed to a *count* matrix
   (a natural refactor for weighted concepts), those tokens would silently get weight 2.
2. **Two of the fifteen concepts are the same mask.** `lemma=a` and `synset=angstrom.n.01`
   both cover exactly 2,701 tokens, and I verified the masks are **bitwise identical** — spaCy
   lemmatises the article "a" and WordNet maps it to the angstrom noun sense. So K=15 is really
   14 distinct concepts plus a duplicate. This costs a slot, inflates `mean_overlap` slightly,
   and guarantees a pair of nodes with identical ceilings on every expansion. Nothing in
   `select_concepts` deduplicates by mask. **This is a real weakness, not a rounding detail.**
3. `:190` writes the CSV using `rows[0].keys()` for the header. If an arm were ever skipped at
   `:178` such that it produced a differently-shaped row, later rows would be written against
   the wrong header. Fine today because every arm builds `row` identically.

### `src/real_activations.py` (231 lines)

**The lines that do the work:**

- **`:98-101`** — the padded id tensor. `torch.full((maxlen, len(batch)), 1)` then real ids
  written in. Time-major, PAD=1.
- **`:102`** — `enc.get_states(ids, torch.tensor(lengths))`. The forward pass.
- **`:104`** — `states[:n, b, :]`. **The padding strip.** One slice; if `n` were wrong every
  downstream index would shift.
- **`:119-120`** — `np.quantile(states, 1-alpha, axis=0)` then compare and transpose.
  `axis=0` is per-unit; that one argument is the difference between "top 20% of each unit" and
  "top 20% of all activations everywhere".

**Choices that could have gone another way:**

| choice | where | the alternative |
|---|---|---|
| checkpoint vocab for the trained arm | `:191-192` | rebuild from the corpus, as the untrained arm does (`:200`). This one is not really a choice — it is a **bug that was avoided**, and the comment at `:187-189` explains why: a rebuilt vocabulary maps every token to the wrong embedding row and produces plausible-looking garbage. 8.8% OOV is the price. |
| `alpha=None` → threshold at 0 | `:117-118` | This is upstream's default (`quantile_features`). A float alpha gives sparser, more neuron-like masks. Both are supported; the sweep uses alphas. |
| corpus vocabulary for the untrained arm | `:51-63` | random per-occurrence ids. The comment at `:54-56` is the justification: with random weights the table is arbitrary anyway, but a token must map to the *same* row every time or the activations carry no lexical structure at all. |
| one forward pass serves all alphas | `:215-217` | re-run the encoder per alpha. The encoder does not depend on alpha, only the threshold does — so this is free, and it also guarantees every alpha sees bit-identical states. |
| density band `[0.15, 0.85]` default | `real_token_search.py:321-322` | the sweeps override to `[0, 1]` with `--min_fire` instead. Worth knowing the default is not what the recorded results used. |

**Things that work by luck rather than by design:**

1. **`PAD=1`, `UNK=0` is a hard-coded cross-repo coupling.** `:58` writes
   `stoi = {PAD: 1, UNK: 0}` and `:101` falls back to `stoi.get(text, 0)`. Both numbers must
   match `nn.Embedding(..., padding_idx=1)` in `models.py:244`, which lives in a *different
   repository*. Nothing asserts it. If upstream changed `padding_idx`, padding would become a
   real token and every activation would shift, silently. (I checked the trained checkpoint:
   its `stoi` also has `UNK→0, PAD→1`, 33,671 types — so the assumption holds for the
   checkpoint too, but again by inspection, not by assertion.)
2. **`verify_alignment` degrades to a warning.** `:139-141`: if `row_tokens` is not passed, it
   prints a warning and checks row *counts* only, which cannot detect a reordering. `main`
   always passes it (`:206-208`), but any other caller can silently get the weak check.
3. `:87` uses `load_state_dict(..., strict=False)` and only *prints* the missing/unexpected
   counts (`:88`) — it does not fail. A checkpoint with a renamed layer would load partially
   and run happily. In practice `real_activations.main` takes the vocab from the checkpoint at
   `:191`, so a truly mismatched checkpoint would blow up on vocab size first — but that is
   luck, not a guard.

### `src/real_token_search.py` (405 lines)

**The lines that do the work:**

- **`:231-234`** — the entire input surface: sparse concept masks, the common/unique/uncoverable
  triple, the disjoint matrix, the neuron bitmap. Four lines that convert our numpy world into
  upstream's expected types.
- **`:240`** — `optimal.heapq = probe`. Frontier instrumentation with zero edits to upstream.
- **`:254`** — `optimal.MAX_FRONTIER_SIZE = beam_cap`. The beam knob.
- **`:265`** — the handoff. Discussed in §1.
- **`:273-277`** — the `finally` restore. Without it, a sweep would leak the probe and the beam
  cap into every subsequent run in the same process.

**Choices that could have gone another way:**

| choice | where | the alternative |
|---|---|---|
| monkey-patch `optimal.heapq` | `:240` | fork upstream and add counters. The patch approach keeps `UPSTREAM` a clean pinned SHA and makes the diff auditable — but see the luck section. |
| `--expand_budget` returns `[]` instead of raising | `:242-253` | raise and abort. Returning empty lets the search **drain its queue and return its incumbent normally**, so a budgeted run yields a usable answer rather than a halt. Scoring is untouched; only exploration is truncated. That is what a work budget should mean, and it is stated at `:239-241`. |
| stdout redirected to devnull around the call | `:261-264`, `:273` | let it print. Upstream streams a `\r` progress line per pop (`optimal.py:702-705`) which destroys sweep output. **Cost: it also swallows any genuine upstream message during the search.** Restored in `finally` (`:273`), so an exception still surfaces. |
| hand-rolled `render()` | `:180-190` | use `F.to_str`. Not a preference — upstream's is **broken**: `BinaryNode.to_str` forwards `sort=` (`formula.py:181-184`) to `Leaf.to_str`, which takes only `namer` (`formula.py:87-88`). It raises `TypeError` on any formula with a leaf child, i.e. all of them. The comment at `:183-184` says so. |
| units filtered by density band **before** any scoring | `:73-74` | filter after. Doing it first is what makes `min_fire` a genuine pre-registered floor rather than a post-hoc filter — stated at `:70-72`. |
| concepts passed but explicitly diagnostic-only | `:224-225` | pass concept names into the search. They do not enter it; the search sees integer indices only. |

**Things that work by luck rather than by design:**

1. **The `heapq` swap depends on how upstream wrote its import.** `optimal.py:4` does
   `import heapq` and then calls `heapq.heappush(...)`, resolving the module attribute at call
   time — which is the only reason assigning `optimal.heapq = probe` intercepts anything. Had
   upstream written `from heapq import heappush`, the probe would be **silently bypassed**:
   no error, no crash, `peak_frontier` would just read a small wrong number forever and every
   frontier-explosion claim in the repo would be unfalsifiable. The comment at
   `synthetic_overlap_sweep.py:113-115` notes the mechanism; it does not note the failure mode.
   *If I were hardening one thing in this repo, it would be an assertion that the probe was
   actually called.*
2. **`HeapProbe.__getattr__` is load-bearing for the patch, by accident.**
   `synthetic_overlap_sweep.py:153-155` delegates unknown attributes to the real `heapq`. That
   was written for generality — but `_apply_beam_cap` calls `heapq.nsmallest`
   (`optimal.py:426`) through the *same swapped module global*. Without that three-line
   `__getattr__`, every beam run would die with `AttributeError`. Two features written weeks
   apart happen to compose. (Pleasant side effect: `heapq.heapify` at `:427` routes through
   `HeapProbe.heapify` (`:148-151`), so the post-cap frontier size *is* counted in
   `peak_frontier`.)
3. **`optimal.expand_node` is rebound on the module, not on the caller.** `:253` works only
   because `perform_search` calls the bare global name `expand_node` at `optimal.py:895`. A
   local alias anywhere would break it silently, same failure mode as (1).
4. `:390` writes the CSV header from `rows[0].keys()`. Same fragility as in
   `real_token_masks.py`; benign because `row` is constructed identically at `:376-385` on
   every iteration.
5. `:291` — `round(best_iou, 4) if best_iou == best_iou else None` uses the NaN-is-not-itself
   trick to detect a halted run. Correct, and idiomatic, but it reads as a typo. Expect to be
   asked.

---

## 5. Five questions you will be asked

**Q1. "Your beam and your exact search agree on 20 of 27 pairs. So is the beam doing anything?"**

Yes, but not at length 3 — and that is the honest framing. **The 20/27 figure is a
length-3 result** (`results/beam_vs_exact_L3_K15.csv`), and at length 3 the exact search is
cheap: unit 413 solves in 0.12 s with a peak frontier of 428, so a 200-node beam barely
binds. Ratio-of-averages over all 27 pairs is +0.96% for exact, +5.05% restricted to the 7
pairs that actually differ (`REPRODUCE.md` §5).

**The beam earns its keep at length 4**, one slot further, same unit
(`results/beam_vs_exact_K15.csv`, joined with `results/alpha_sweep_K15.csv`):

| arm, alpha | exact IoU | exact time | exact peak | beam-200 IoU | beam time | beam peak |
|---|---|---|---|---|---|---|
| trained 0.2 | 0.4894 | 23.2 s | 8,239 | 0.4174 | 0.61 s | 245 |
| trained 0.1 | 0.0962 | 200.7 s | 6,878 | 0.0909 | 1.22 s | 247 |
| untrained 0.1 | 0.1545 | 1,069.6 s | 26,049 | 0.1228 | 1.29 s | 250 |
| untrained 0.05 | 0.1084 | 1,309.9 s | 22,120 | 0.0860 | 1.54 s | 250 |

38× to 850× faster, giving up 5–21% of the IoU, with the frontier held at ~250 nodes instead of
6.9k–26k. And this is the *survivable* end of length 4 — `REPRODUCE.md` §6 records that of
the 27 length-4 runs, **4 hit the cap outright**, so for those the comparison is not
"worse IoU" but "an explanation versus none".

One caveat to volunteer before someone else raises it: at length 3 the beam is close to
free *and* close to lossless, so if a reviewer only reads `beam_vs_exact_L3_K15.csv` the
patch looks unnecessary. It is the length-4 column that justifies it.

**Q2. "How do you know the concept masks and the activations are indexed by the same tokens
in the same order?"**

Three layers. (a) **Structural:** both sides call the *same parser*,
`rtm.load_sentences` — the mask side at `src/real_token_masks.py:81`, the activation side at
`src/real_activations.py:184`. (b) **Asserted at runtime:** `verify_alignment`
(`src/real_activations.py:123`) compares the surface token behind 50 random activation rows
against the token the mask side holds at the same index (`:147-154`) and raises on any
mismatch. Row counts alone cannot catch a reordering — `:140` says so. (c) **Audited
independently:** `verify/check_alignment.py` re-derives the token stream with a hand-written
parser that never imports `real_token_masks`, and matches 24,199/24,199 on both arms
(`VERIFICATION.md` check 1). The padding question — does a short sentence get offset inside a
mixed-length batch — is answered separately by `verify/check_padding.py` at 6.5e-8 with a
working negative control (check 2).

**Q3. "You patched the upstream search. How do I know your exact-search numbers are still
upstream's?"**

`_apply_beam_cap` returns **the same list object, unmodified**, when `MAX_FRONTIER_SIZE is
None` — the `None` check is first in the condition at `compositional/optimal.py:424`, before
any `len`, copy, or heap operation. So both call sites (`:447`, `:516`) reduce to `x = x`. It
is measured too: `verify/check_patch_noop.py` prints the winning formula, `best_iou` **as a
hex float** (`:53`), and all four search counters, so a clean checkout and a patched one can be
diffed bit-for-bit. `VERIFICATION.md` check 3 records PASS. The harness also restores `None` in
a `finally` (`src/real_token_search.py:277`) so a beam cap cannot leak between sweep rows.

**Q4. "IoU 0.4689 is the best over thousands of candidates. Isn't that just best-of-N noise?"**

Partly, and the repo says so rather than hiding it. Three guards. (a) A **pre-registered
statistical floor**: `--min_fire 200`, applied before any scoring (`src/real_token_search.py:73-74`),
because a unit firing on a handful of tokens cannot support a formula drawn from ~1.4M
candidates — the reasoning is written at `:70-72`. (b) The recorded results carry a **`lift`
column** (IoU over the independence baseline) rather than raw IoU alone —
`results/alpha_sweep_K15.csv` line 15 gives unit 413 @ alpha 0.2 lift 3.14. (c) A
**permutation test** in `phaseB_report.py`, seeded `random.Random(0)` (`REPRODUCE.md`, seeds
table). The honest residual: the alphas, K values and the density band were chosen by us, and
`min_support=5` was never swept.

**Q5. "You asked for length 3 and got a length-3 formula here — but I see length-1 winners in
your CSVs. Which is it?"**

`length` is a **maximum, not a target**, and shorter answers are actively preferred. Two
mechanisms. (a) When a terminal formula is evaluated, upstream also evaluates **every
intermediate subtree** on the path (`optimal.py:822-838`) and lets any of them become the
incumbent (`:841-846`) — so a length-3 run can return a length-1 answer without ever having
searched for one. (b) Ties break toward the shorter formula, at both the terminal site
(`:809-812`) and the ancestor site (`:843-846`). That is why unit 413 at alpha 0.5 returns
plain `(NOT dep=pobj)` (`results/alpha_sweep_K15.csv:20`) — at that density the neuron fires
on half the corpus and nothing longer beat it. **Related trap, expect it:** length counts
*leaves*, not operators (`formula.py:90`, `:152`, `:189`), so NOT is free and
`((const=NP AND (NOT const=VP)) AND (NOT const=PP))` is length **3**, not 5.

---

## Appendix: the numbers a reviewer may spot-check

| quantity | value | where |
|---|---|---|
| sentences / tokens | 2,000 / 24,199 | `REPRODUCE.md` §2 |
| K=15 concept overlap | mean 3.189, common_frac 0.858, unique 13.9%, max 7 | `src/real_token_masks.py:114` |
| disjoint concept pairs | 108 / 210 (51.4%) | ditto |
| unit 413 @ alpha 0.2 | fires 4,840 tokens (density 0.20001) | `results/acts2k_trained_a0.2.npz` |
| exact L3 winner | `((const=NP AND (NOT const=VP)) AND (NOT const=PP))` | reproduced 2026-07-31 |
| its IoU | 0.4689 = 3307 / (4840 + 5519 − 3307) | ditto |
| search cost | visited 65, expanded 260, estimated 2,682, peak 428, 0.12 s | ditto; matches `results/beam_vs_exact_L3_K15.csv:2` |
| same unit at L4, exact | IoU 0.4894, 23.2 s, peak 8,239 | `results/beam_vs_exact_K15.csv:2` |
| same unit at L4, beam 200 | IoU 0.4174, 0.61 s, peak 245 | `results/alpha_sweep_K15.csv:15` |
| L4 runs that hit the cap | 4 of 27 | `REPRODUCE.md` §6 |
