"""Real per-token neuron activations from a Bowman SNLI encoder, on the token axis.

Why this file exists instead of calling the upstream analysis
-------------------------------------------------------------
Two obstacles in `neuron-explanations-nli/nli/code` rule out reusing its pipeline:

1. `data/snli.py:load_for_analysis` unconditionally `torch.load`s a checkpoint and takes
   **both** the weights and the vocabulary (`stoi`/`itos`) from it. The `RANDOM_WEIGHTS`
   flag lives in `loader.py` (the machine-translation loader) and does not reach the NLI
   path, so there is no checkpoint-free route through their code.
2. Their NLI neurons live on the **example** axis: `get_final_reprs` returns one vector per
   sentence *pair*. Our concept masks are per **token**.

`models.TextEncoder.get_states()` resolves both — it returns the per-timestep LSTM outputs
`(seq_len, batch, hidden_dim)`, i.e. exactly one activation vector per token — so we build
the encoder ourselves, with a vocabulary derived from the corpus rather than a checkpoint.

Element axis: tokens, matching the OptimalCE analogy (pixels are sub-image units, tokens are
sub-sentence units). This is finer-grained than the upstream NLI setup by design.

Untrained mode (`--untrained`) is the near-free control: no corpus download and no training
run, and a random-weights baseline is standard in this literature. It is a genuine test
rather than a smoke test, because the proxy neuron used so far is an OR of three concepts
plus noise and therefore has a near-perfect short explanation by construction. A strong
best-so-far IoU is what prunes this search, so a target *without* a good short explanation
should prune worse and grow the frontier. Untrained units supply exactly that weak-floor
case.

**What untrained activations cannot support:** any claim about what trained neurons encode.
They bound the search-tractability question only.
"""

import argparse
import os
import sys

import numpy as np

NLI_CODE = os.environ.get("NLI_CODE", os.path.expanduser(
    "~/projects/neuron-explanations-nli/nli/code"))
sys.path.insert(0, NLI_CODE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import env_info  # noqa: E402
import real_token_masks as rtm  # noqa: E402

PAD = "<pad>"
UNK = "<unk>"


def build_vocab(sents):
    """Vocabulary over the corpus itself, so there are no OOV tokens.

    With random weights the embedding table is arbitrary anyway; what matters is that a
    given token always maps to the *same* row, so activations carry real lexical structure
    instead of being independent noise per occurrence.
    """
    stoi = {PAD: 1, UNK: 0}
    for sent in sents:
        for text, _ in sent:
            if text not in stoi:
                stoi[text] = len(stoi)
    return stoi


def extract_states(sents, stoi, hidden_dim, embedding_dim, seed, ckpt=None, batch_size=64,
                   row_tokens=None):
    """Per-token LSTM hidden states, flattened to (n_tokens, hidden_dim).

    Tokens come back in exactly the reading order of `rtm.load_sentences`, which is the
    order the concept masks use.

    If `row_tokens` is a list, the surface token behind every output row is appended to it,
    so `verify_alignment` can assert the ORDER rather than only the count.
    """
    import torch
    import models  # from the NLI codebase

    torch.manual_seed(seed)
    enc = models.TextEncoder(len(stoi), embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu")
        sd = state.get("state_dict", state)
        # Checkpoint keys are prefixed with "encoder." when saved from the classifier.
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
        print(f"[ckpt] loaded {ckpt} (missing={len(missing)}, unexpected={len(unexpected)})")
    enc.eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(sents), batch_size):
            batch = sents[i:i + batch_size]
            lengths = [len(s) for s in batch]
            maxlen = max(lengths)
            # (seq_len, batch) with PAD=1, matching TextEncoder's padding_idx.
            ids = torch.full((maxlen, len(batch)), 1, dtype=torch.long)
            for b, sent in enumerate(batch):
                for t, (text, _) in enumerate(sent):
                    ids[t, b] = stoi.get(text, 0)
            states = enc.get_states(ids, torch.tensor(lengths))  # (seq, batch, hidden)
            for b, n in enumerate(lengths):
                out.append(states[:n, b, :].numpy())
                if row_tokens is not None:
                    row_tokens.extend(text for text, _ in batch[b][:n])
    return np.concatenate(out, axis=0)


def binarize(states, alpha=None):
    """(n_tokens, n_units) float -> (n_units, n_tokens) bool neuron masks.

    `alpha=None` reproduces the upstream default (`quantile_features`: threshold at 0).
    A float alpha keeps the top-alpha fraction per unit, giving sparser, more neuron-like
    masks; density matters here because a very dense target is trivially coverable.
    """
    if alpha is None:
        return (states > 0).T
    thresh = np.quantile(states, 1 - alpha, axis=0)
    return (states > thresh[np.newaxis, :]).T


def verify_alignment(sents, states, row_tokens=None, n_sample=50, seed=0):
    """The masks and the activations must index the same tokens in the same ORDER.

    Row counts alone cannot detect a reordering. Order is guaranteed upstream by
    `pack_padded_sequence(..., enforce_sorted=False)` in `TextEncoder.get_states`, which
    sorts internally and restores the original order -- but that is upstream's promise, not
    ours, and it would break silently if upstream changed. So when `row_tokens` is supplied
    this compares the actual surface token behind `n_sample` random rows against the token
    the mask side has at the same index.
    """
    n_tokens = sum(len(s) for s in sents)
    if states.shape[0] != n_tokens:
        raise AssertionError(
            f"token-axis mismatch: {n_tokens} tokens from the .feats parse vs "
            f"{states.shape[0]} activation rows — masks and neurons would not align")

    if row_tokens is None:
        print("[align] WARNING: row counts only; pass row_tokens= to assert ORDER")
        return n_tokens

    expected = [text for sent in sents for text, _ in sent]
    if len(row_tokens) != n_tokens:
        raise AssertionError(
            f"row_tokens length {len(row_tokens)} != {n_tokens} tokens")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_tokens, size=min(n_sample, n_tokens), replace=False)
    bad = [(int(i), expected[i], row_tokens[i]) for i in idx if expected[i] != row_tokens[i]]
    if bad:
        detail = "; ".join(f"row {i}: masks={m!r} acts={a!r}" for i, m, a in bad[:5])
        raise AssertionError(
            f"TOKEN ORDER MISMATCH at {len(bad)}/{len(idx)} sampled rows — masks and "
            f"activations are not aligned: {detail}")
    print(f"[align] {n_tokens} rows; token ORDER verified on {len(idx)} sampled indices")
    return n_tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feats", default=os.path.expanduser(
        "~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats"))
    ap.add_argument("--max_sents", type=int, default=200)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--embedding_dim", type=int, default=300)
    ap.add_argument("--untrained", action="store_true",
                    help="random weights (no checkpoint, no training) — the weak-floor control")
    ap.add_argument("--ckpt", default=None, help="Bowman SNLI checkpoint, e.g. models/bowman_snli/6.pth")
    ap.add_argument("--alpha", type=float, default=None,
                    help="top-fraction threshold per unit; omit for the upstream >0 default")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="sweep several alphas from ONE forward pass; --out is treated as a "
                         "template and gets an _a<alpha> suffix per value")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--align_sample", type=int, default=50,
                    help="rows to spot-check for token-order alignment")
    ap.add_argument("--out", default="results/real_activations.npz")
    args = ap.parse_args()
    env_info.print_banner('acts')

    if not args.untrained and args.ckpt is None:
        ap.error("pass --untrained for the random-weights control, or --ckpt PATH for a trained model")

    sents = rtm.load_sentences(args.feats, args.max_sents)

    if args.ckpt is not None:
        # The embedding table is indexed by the TRAINING vocabulary. Rebuilding a vocabulary
        # from the annotation corpus here would map every token to the wrong embedding row and
        # fail silently — plausible-looking activations with no relationship to the model.
        import torch
        _ck = torch.load(args.ckpt, map_location="cpu")
        stoi = _ck["stoi"]
        hidden_dim = _ck.get("hidden_dim", args.hidden_dim)
        embedding_dim = _ck.get("embedding_dim", args.embedding_dim)
        oov = sum(1 for s in sents for text, _ in s if text not in stoi)
        n_tok = sum(len(s) for s in sents)
        print(f"[acts] using checkpoint vocab ({len(stoi)} types) | "
              f"OOV {oov}/{n_tok} ({oov/max(n_tok,1):.1%}) -> UNK")
    else:
        stoi = build_vocab(sents)
        hidden_dim, embedding_dim = args.hidden_dim, args.embedding_dim

    print(f"[acts] {len(sents)} sentences | {sum(len(s) for s in sents)} tokens | "
          f"vocab {len(stoi)} | {'UNTRAINED (random weights)' if args.untrained else args.ckpt}")

    row_tokens = []
    states = extract_states(sents, stoi, hidden_dim, embedding_dim,
                            args.seed, ckpt=args.ckpt, row_tokens=row_tokens)
    n_tokens = verify_alignment(sents, states, row_tokens=row_tokens,
                                n_sample=args.align_sample)
    print(f"[acts] states {states.shape} — aligned to {n_tokens} mask tokens")

    # The forward pass does not depend on alpha -- only the binarization does -- so a sweep
    # reuses one set of states rather than re-running the encoder per alpha.
    alphas = args.alphas if args.alphas is not None else [args.alpha]
    base, ext = os.path.splitext(args.out)
    for alpha in alphas:
        acts = binarize(states, alpha)
        density = acts.mean(axis=1)
        fired = acts.sum(axis=1)
        out = args.out if args.alphas is None else f"{base}_a{alpha}{ext}"
        print(f"[acts] alpha={alpha} -> {acts.shape} | density mean {density.mean():.4f} "
              f"min {density.min():.4f} max {density.max():.4f} | firing tokens/unit: "
              f"mean {fired.mean():.1f} min {int(fired.min())} max {int(fired.max())}")
        np.savez_compressed(out, acts=acts, density=density,
                            untrained=args.untrained, alpha=alpha if alpha else -1)
        print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
