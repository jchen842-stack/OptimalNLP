"""Train a Bowman SNLI entailment classifier, to supply *trained* units to the search.

Why our own trainer instead of `nli/code/snli_train.py`
-------------------------------------------------------
Theirs exposes only `--debug` (1 000 pairs) or the full 550 k corpus for 50 epochs, with no
subset knob. We want a middle setting: enough training that units acquire linguistic
structure, without hours of CPU. Everything substantive is imported from their code —
`SNLI`, `pad_collate`, `models` — so this is a driver, not a fork.

The checkpoint is written in their format (`state_dict` / `stoi` / `itos`), so it loads with
`real_activations.py --ckpt`.

**Vocabulary matters.** The embedding table is indexed by the *training* vocabulary, so
`real_activations.py` must use `ckpt["stoi"]` when a checkpoint is supplied rather than a
vocabulary rebuilt from the annotation corpus. Mismatched indices would map every token to the
wrong embedding and fail silently — activations would still be produced, just meaningless.
"""

import argparse
import os
import sys
import time

import numpy as np

NLI_CODE = os.environ.get("NLI_CODE", os.path.expanduser(
    "~/projects/neuron-explanations-nli/nli/code"))
sys.path.insert(0, NLI_CODE)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/snli_1.0/", help="relative to the NLI code dir")
    ap.add_argument("--max_data", type=int, default=100000,
                    help="training pairs to use (None-like: pass 0 for the full 550k)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--embedding_dim", type=int, default=300)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/projects/optimalce-nlp/models"))
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from data.snli import SNLI, pad_collate
    import models

    torch.manual_seed(args.seed)
    os.chdir(NLI_CODE)  # their loader resolves `data/snli_1.0/` relative to cwd
    os.makedirs(args.out, exist_ok=True)

    max_data = args.max_data if args.max_data > 0 else None
    t0 = time.time()
    train = SNLI(args.data, "train", max_data=max_data)
    val = SNLI(args.data, "dev", vocab=(train.stoi, train.itos))
    print(f"[train] {len(train.s1s)} train / {len(val.s1s)} val pairs | "
          f"vocab {len(train.stoi)} | loaded in {time.time()-t0:.1f}s", flush=True)

    loaders = {
        "train": DataLoader(train, batch_size=args.batch_size, shuffle=True,
                            collate_fn=pad_collate),
        "val": DataLoader(val, batch_size=args.batch_size, shuffle=False,
                          collate_fn=pad_collate),
    }

    enc = models.TextEncoder(len(train.stoi), embedding_dim=args.embedding_dim,
                             hidden_dim=args.hidden_dim)
    model = models.BowmanEntailmentClassifier(enc)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        for split in ("train", "val"):
            model.train() if split == "train" else model.eval()
            losses, correct, total = [], 0, 0
            t1 = time.time()
            for i, (s1, s1len, s2, s2len, label) in enumerate(loaders[split]):
                with torch.set_grad_enabled(split == "train"):
                    preds = model(s1, s1len, s2, s2len)
                    loss = criterion(preds, label)
                    if split == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                losses.append(loss.item())
                correct += (preds.argmax(1) == label).sum().item()
                total += label.numel()
                if split == "train" and i % 100 == 0:
                    print(f"  epoch {epoch} batch {i}/{len(loaders[split])} "
                          f"loss {np.mean(losses[-100:]):.4f} acc {correct/max(total,1):.4f} "
                          f"({time.time()-t1:.0f}s)", flush=True)
            acc = correct / max(total, 1)
            print(f"[{split}] epoch {epoch}: loss {np.mean(losses):.4f} acc {acc:.4f} "
                  f"({time.time()-t1:.0f}s)", flush=True)

            if split == "val":
                ckpt = {"state_dict": model.state_dict(),
                        "stoi": train.stoi, "itos": train.itos,
                        "val_acc": acc, "epoch": epoch,
                        "embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim}
                torch.save(ckpt, os.path.join(args.out, f"bowman_snli_e{epoch}.pth"))
                if acc > best_acc:
                    best_acc = acc
                    torch.save(ckpt, os.path.join(args.out, "bowman_snli_best.pth"))
                    print(f"  -> new best ({acc:.4f}), saved bowman_snli_best.pth", flush=True)

    print(f"\n[train] done. best val acc {best_acc:.4f} | checkpoints in {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
