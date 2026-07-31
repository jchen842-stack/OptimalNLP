# Research diary — index

Chronological. Each entry revises the one before it; **nothing is deleted when it is
superseded**, because the corrections are why the surviving claims are credible. Entries that
were later corrected carry a `⚠️` banner at the top naming what changed and where.

Read newest-first if you want the current state; oldest-first if you want how it got there.

| entry | date | topic | revises | status |
|---|---|---|---|---|
| [D5](summer_d5.md) | 2026-07-24 | Reproducing the NLP failure synthetically; the beam-fallback fix (`MAX_FRONTIER_SIZE`) | — | **current** — the fix and the frontier mechanism stand |
| [D5.1](summer_d5.1.md) | 2026-07-26 | Real SNLI token masks; overlap is structural, not a knob | D5 | ⚠️ **partly superseded** — the wall is on the **length** axis, not the K axis (D5.5 §3); `disjoint_pairs` is the wrong statistic, unique elements is the right one (D5.5 §1) |
| [D5.2](summer_d5.2.md) | 2026-07-26 | Real per-token unit activations replace the proxy neuron | D5.1 | ⚠️ **IoU results withdrawn** (D5.4) — measured at density ~0.50 where IoU ≈ density. Frontier and timing results stand |
| [D5.3](summer_d5.3.md) | 2026-07-26 | Trained Bowman SNLI encoder (0.7934 dev); trained vs untrained units | D5.2 | ⚠️ **IoU comparisons withdrawn** (D5.4). The "vocabulary is the binding constraint" conclusion is unsupported — it was the activation range (D5.5 §5) |
| [D5.4](summer_d5.4.md) | 2026-07-31 | The IoUs were tracking density, not explanation quality | D5.2, D5.3 | **current** — the diagnosis |
| [D5.5](summer_d5.5.md) | 2026-07-31 | Activation range corrected; corpus scaled 10x; alpha sweep; matched-length band comparison; correctness audit | D5.4 | **current** — the state of the work |

## Where the current numbers live

Diary entries record *how* things were found. For the numbers themselves:

- **[`../results/MANIFEST.md`](../results/MANIFEST.md)** — every results file, its producer, its
  diary entry, and whether it is current or superseded.
- **[`../results/METHOD_NOTES.md`](../results/METHOD_NOTES.md)** — pre-registered predictions and
  their outcomes, plus the recurring methodological errors. **Read this before quoting any
  number.**
- **[`../VERIFICATION.md`](../VERIFICATION.md)** — the correctness audit. `../verify/run_all.sh`
  re-runs it.

## Format

`.md` is the source of truth. To render one for Google Docs (landscape .docx):

```sh
scripts/make_diary_docx.sh diary/summer_d5.5.md
```

Rendered `.docx` files are not tracked — they duplicate the `.md` and drift from it silently
once the `.md` is edited.

## Adding an entry

1. `diary/summer_d5.N.md`, same structure: a `*Revises …*` italic preamble stating what
   changed, then `## 0. Headline`, the sections, `## Scope`, `## Next steps`, `## Artifacts`.
2. Add a row to the table above.
3. If it supersedes an earlier claim, put a `⚠️` banner at the top of **that** entry —
   immediately above the claim it corrects, never below — and update that entry's status here.
4. Do not rename or move existing entries; their paths are cited in the top-level `README.md`
   and in external write-ups.
