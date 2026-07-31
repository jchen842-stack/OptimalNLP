"""Check 7: does the checkpoint reproduce 0.7934 dev accuracy? And what invocation made it?

Also: OOV rate with checkpoint stoi (claimed 8.8%) vs a corpus-rebuilt stoi, to show the
check is sensitive.
"""
import os, sys, time
NLI = os.path.expanduser('~/projects/neuron-explanations-nli/nli/code')
sys.path.insert(0, NLI); sys.path.insert(0, os.path.abspath('src'))
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
REPO = os.getcwd()
os.chdir(NLI)
from data.snli import SNLI, pad_collate
import models

ck = torch.load(os.path.join(REPO,'models/bowman_snli_best.pth'), map_location='cpu')
print(f"checkpoint: val_acc={ck['val_acc']!r} epoch={ck['epoch']} "
      f"emb={ck['embedding_dim']} hid={ck['hidden_dim']} vocab={len(ck['stoi'])}")

# --- which invocation? vocab size is a fingerprint of max_data --------------------
import contextlib, io
for md, lab in ((100000,'--max_data 100000 (DEFAULT)'), (None,'--max_data 0 (full corpus)')):
    t0=time.time()
    with contextlib.redirect_stderr(io.StringIO()):
        tr = SNLI('data/snli_1.0/', 'train', max_data=md)
    print(f"  {lab:<32} pairs={len(tr):>7} vocab={len(tr.stoi):>6} "
          f"matches checkpoint vocab 33671: {len(tr.stoi)==len(ck['stoi'])}  ({time.time()-t0:.0f}s)")
    if len(tr.stoi)==len(ck['stoi']): train_ref = tr

# --- re-evaluate dev --------------------------------------------------------------
with contextlib.redirect_stderr(io.StringIO()):
    val = SNLI('data/snli_1.0/','dev', vocab=(ck['stoi'], ck['itos']))
enc = models.TextEncoder(len(ck['stoi']), embedding_dim=ck['embedding_dim'],
                         hidden_dim=ck['hidden_dim'])
model = models.BowmanEntailmentClassifier(enc)
model.load_state_dict(ck['state_dict']); model.eval()
loader = DataLoader(val, batch_size=100, shuffle=False, collate_fn=pad_collate)
correct=tot=0
with torch.no_grad():
    for s1, s1len, s2, s2len, label in loader:
        preds = model(s1, s1len, s2, s2len)
        correct += (preds.argmax(1)==label).sum().item(); tot += label.numel()
acc = correct/tot
print(f"\nRE-EVALUATED dev accuracy: {acc!r}  ({correct}/{tot})")
print(f"stored val_acc           : {ck['val_acc']!r}")
print(f"match to 1e-9            : {abs(acc-ck['val_acc'])<1e-9}")

# --- OOV sensitivity --------------------------------------------------------------
os.chdir(REPO)
import real_token_masks as rtm, real_activations as ra
FEATS = os.path.expanduser('~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats')
sents = rtm.load_sentences(FEATS, 2000)
n_tok = sum(len(s) for s in sents)
oov_ck = sum(1 for s in sents for t,_ in s if t not in ck['stoi'])
corpus_stoi = ra.build_vocab(sents)
oov_corpus = sum(1 for s in sents for t,_ in s if t not in corpus_stoi)
print(f"\nOOV with CHECKPOINT stoi ({len(ck['stoi'])} types): {oov_ck}/{n_tok} = {oov_ck/n_tok:.1%}")
print(f"OOV with CORPUS-REBUILT stoi ({len(corpus_stoi)} types): {oov_corpus}/{n_tok} = {oov_corpus/n_tok:.1%}")
print("  -> rebuilding gives 0% OOV, which is exactly why it fails SILENTLY: every token")
print("     maps to SOME row, just the wrong one.")
