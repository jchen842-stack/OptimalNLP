"""Check 2: padding does not offset short sentences inside a mixed-length batch.

Independent method: run each sentence ALONE (batch of 1 -> zero padding) and compare to the
same sentence's slice from a mixed-length batched run. If padding leaked, the short
sentence's states would differ.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.expanduser('~/projects/neuron-explanations-nli/nli/code'))
sys.path.insert(0, 'src')
import real_token_masks as rtm, real_activations as ra

FEATS = os.path.expanduser('~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats')
sents = rtm.load_sentences(FEATS, 2000)
stoi = ra.build_vocab(sents)

lens = [len(s) for s in sents]
short_i = int(np.argmin(lens)); long_i = int(np.argmax(lens))
print(f"shortest sentence idx {short_i} len {lens[short_i]}; "
      f"longest idx {long_i} len {lens[long_i]}  (ratio {lens[long_i]/lens[short_i]:.1f}x)")

pair = [sents[short_i], sents[long_i]]

# (a) batched together -> requires padding the short one to the long one's length
batched = ra.extract_states(pair, stoi, 512, 300, seed=0, batch_size=64)
# (b) each alone -> batch of 1, maxlen == its own length, NO padding at all
alone_s = ra.extract_states([sents[short_i]], stoi, 512, 300, seed=0, batch_size=64)
alone_l = ra.extract_states([sents[long_i]],  stoi, 512, 300, seed=0, batch_size=64)

ns, nl = lens[short_i], lens[long_i]
print(f"batched rows {batched.shape[0]} (expect {ns+nl}); alone {alone_s.shape[0]} + {alone_l.shape[0]}")
bs, bl = batched[:ns], batched[ns:ns+nl]
d_s = np.abs(bs - alone_s).max(); d_l = np.abs(bl - alone_l).max()
print(f"max |batched - alone|  short: {d_s:.3e}   long: {d_l:.3e}")

# Order check: reversing the batch must produce the same per-sentence states.
rev = ra.extract_states([sents[long_i], sents[short_i]], stoi, 512, 300, seed=0, batch_size=64)
d_rev_l = np.abs(rev[:nl] - alone_l).max(); d_rev_s = np.abs(rev[nl:nl+ns] - alone_s).max()
print(f"reversed batch order    short: {d_rev_s:.3e}   long: {d_rev_l:.3e}")

# Negative control: if we deliberately DID leak padding (take rows 0..ns from the padded
# tensor of the LONG sentence), the diff should be large -- proves the test can fail.
leak = np.abs(alone_l[:ns] - alone_s).max()
print(f"negative control (deliberate wrong slice): {leak:.3e}  <- must be LARGE")

tol = 1e-5
ok = max(d_s, d_l, d_rev_s, d_rev_l) < tol and leak > tol
print("RESULT:", "PASS" if ok else "FAIL")
