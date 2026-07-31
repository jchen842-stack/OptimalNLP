# Checkpoints (gitignored) — how to reproduce them

The `.pth` files in this directory are **not in version control** (`.gitignore` excludes
`models/*`). `bowman_snli_best.pth` is the model behind every trained-arm number in D5.3 and
in the Phase B / alpha-sweep results, so if it is lost those numbers cannot be regenerated
without retraining. This file records everything needed to retrain it.

## Files

| file | what it is |
|---|---|
| `bowman_snli_best.pth` | best-dev checkpoint — **this is the one used for all results** |
| `bowman_snli_e0.pth` | end of epoch 0 |
| `bowman_snli_e1.pth` | end of epoch 1 |
| `bowman_snli_e2.pth` | end of epoch 2 (identical dev acc to `best`; epoch 2 *was* the best) |

Each is ~54 MB.

## Invocation

```sh
python src/train_snli_encoder.py --max_data 0
```

`--max_data 0` is **required** and is not the default. Verified: the checkpoint stores a
33,671-type vocabulary, and rebuilding the training set gives 33,671 types only with
`--max_data 0` (549,367 pairs). The default `--max_data 100000` yields 99,889 pairs and a
16,669-type vocabulary, which cannot have produced this checkpoint. See
`VERIFICATION.md` check 7.

## Hyperparameters

Mostly `train_snli_encoder.py` argparse defaults; `--max_data` is the exception. The four
values the checkpoint stores internally match.

| parameter | value | source |
|---|---|---|
| epochs | 3 | default |
| batch size | 100 | default |
| optimiser | Adam | hardcoded (`optim.Adam`) |
| learning rate | 1e-3 | default |
| embedding dim | 300 | default, **confirmed in checkpoint** |
| hidden dim | 512 | default, **confirmed in checkpoint** |
| `--max_data` | **0** (= use all 549,367 pairs) | **NOT the default**; recovered from vocab size |
| `--data` | `data/snli_1.0/` relative to the NLI code dir | default |
| seed | 0 (assumed) | default — see caveat, this one is NOT verified |

Training corpus: full SNLI, 549,367 train pairs. Encoder is
`models.TextEncoder` from the `neuron-explanations-nli` codebase (`NLI_CODE`), an LSTM over
300-d embeddings with 512 hidden units; per-token states come from `get_states()`.

Runtime was ~13 min/epoch on CPU.

## Accuracy trace

| epoch | train loss | train acc | val acc |
|---|---|---|---|
| 0 | 0.699 | 0.701 | 0.772 |
| 1 | 0.537 | 0.783 | 0.792 |
| 2 | 0.461 | 0.817 | **0.793** |

Final dev accuracy stored in the checkpoint: **0.7934362934362934** (`val_acc`), at
`epoch: 2`. Above the task7 Bowman reference of 0.775. Trained for 3 epochs, not to
convergence.

## Vocabulary — the trap

**Vocabulary size: 33,671 types**, built from the SNLI *training* split and stored in the
checkpoint as `stoi` / `itos`.

Any consumer of this checkpoint **must take `stoi` from the checkpoint**, never rebuild it
from the annotation corpus. The embedding table is indexed by the training vocabulary; a
vocabulary rebuilt from a different corpus maps each token to the wrong embedding row and
**fails silently** — activations are still produced, they are simply meaningless.
`src/real_activations.py` does this correctly when `--ckpt` is supplied. Against the 2,000
sentence annotation corpus (24,199 tokens) the OOV rate is 8.8% (2,137 tokens → `<unk>`).

Checkpoint format matches the upstream NLI codebase: `state_dict`, `stoi`, `itos`,
`val_acc`, `epoch`, `embedding_dim`, `hidden_dim`. Encoder weights are stored under an
`encoder.` prefix because the checkpoint is saved from the classifier.

## Seed and provenance — what is asserted and what is not

`train_snli_encoder.py` calls `torch.manual_seed(args.seed)` with `--seed` defaulting to
**0**, so a defaults-only run is seeded and reproducible.

**Recovered:** `--max_data 0`. An earlier version of this file claimed the run used
defaults throughout; that was wrong. The vocabulary size is a fingerprint of `--max_data`
and it rules the default out (see above).

**Still NOT verified: the seed.** It leaves no trace in the checkpoint, and no log of the
invocation survives. `--seed 0` is the default and is what a defaults-run would have used,
but the run demonstrably did *not* use all defaults, so the seed cannot be inferred from
that. Retraining with `--seed 0 --max_data 0` should reproduce 0.7934 dev accuracy;
**if it does not, the original seed was different and this record is incomplete.**

**What IS verified:** the stored checkpoint re-evaluates to dev accuracy
0.7934362934362934, exactly matching its own stored `val_acc` to 1e-9, on a from-scratch
evaluation (`VERIFICATION.md` check 7).

## Retraining

```sh
# from the repo root, with NLI_CODE pointing at the neuron-explanations-nli checkout
python src/train_snli_encoder.py --seed 0 --max_data 0
```

Then regenerate activations (the sweep binarises at several alphas from one forward pass):

```sh
python src/real_activations.py --max_sents 2000 --ckpt models/bowman_snli_best.pth \
    --alphas 0.5 0.2 0.1 0.05 0.005 --out results/acts2k_trained.npz
```
