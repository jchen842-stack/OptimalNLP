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
python src/train_snli_encoder.py
```

No flags were recorded in the diary for the run that produced `bowman_snli_best.pth`. See
"Seed and provenance" below for what can and cannot be asserted about that.

## Hyperparameters

These are the `train_snli_encoder.py` argparse defaults, and the four that the checkpoint
stores internally match them exactly.

| parameter | value | source |
|---|---|---|
| epochs | 3 | default |
| batch size | 100 | default |
| optimiser | Adam | hardcoded (`optim.Adam`) |
| learning rate | 1e-3 | default |
| embedding dim | 300 | default, **confirmed in checkpoint** |
| hidden dim | 512 | default, **confirmed in checkpoint** |
| `--max_data` | 100000 | default (see note below) |
| `--data` | `data/snli_1.0/` relative to the NLI code dir | default |
| seed | 0 | default — see caveat |

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

**What cannot be asserted:** the exact command line used for this checkpoint was never
logged. Every parameter the checkpoint records internally (`embedding_dim` 300,
`hidden_dim` 512, `epoch` 2 of 3) is consistent with an all-defaults run, and the diary's
reported corpus size and vocab match too — but consistency is not proof, and the seed in
particular leaves no trace in the checkpoint. Retraining with defaults should reproduce
0.7934 dev accuracy; **if it does not, the original run used non-default flags and this
record is incomplete.** Treat an exact-match retrain as confirmation, and a mismatch as
evidence that the provenance is lost rather than that something is broken.

`--max_data 100000` is the default but the diary records the full 549,367-pair corpus being
used, so this flag was either overridden or is not applied to the training split. Check
`train_snli_encoder.py` before assuming the default was in force.

## Retraining

```sh
# from the repo root, with NLI_CODE pointing at the neuron-explanations-nli checkout
python src/train_snli_encoder.py --seed 0
```

Then regenerate activations (the sweep binarises at several alphas from one forward pass):

```sh
python src/real_activations.py --max_sents 2000 --ckpt models/bowman_snli_best.pth \
    --alphas 0.5 0.2 0.1 0.05 0.005 --out results/acts2k_trained.npz
```
