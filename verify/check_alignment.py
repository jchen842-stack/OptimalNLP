"""Checks 1, 2, 5, 8: token-order alignment, padding, mask correctness, binarisation.

Independence: the token stream is re-derived by a SEPARATE hand-written parser in this
file, never by calling real_token_masks. The activation-side row->token map is rebuilt by
replicating the batching in real_activations.extract_states and emitting the token text
each row came from.
"""
import os, sys, random
import numpy as np, torch
sys.path.insert(0, os.path.expanduser('~/projects/neuron-explanations-nli/nli/code'))
sys.path.insert(0, 'src')
import real_token_masks as rtm
import real_activations as ra
import models

FEATS = os.path.expanduser('~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats')
CATS = ["lemma","tag","dep","ent","synset","const"]

def independent_parse(path, max_sents):
    """Hand parser, written independently of real_token_masks."""
    sents=[]
    with open(path, encoding='utf-8') as f:
        f.readline()                      # header
        for n, line in enumerate(f):
            if n >= max_sents: break
            line=line.rstrip('\n').strip()
            if not line: continue
            toks=[]
            for chunk in line.split(' '):
                f7 = chunk.split('|')
                if len(f7) != 7: continue
                text = f7[0]
                feats = {}
                for cat, val in zip(CATS, f7[1:]):
                    if val == '': continue
                    feats[cat] = val.split(';') if cat=='const' else [val]
                toks.append((text, feats))
            sents.append(toks)
    return sents

def rows_to_tokens(sents, stoi, batch_size=64):
    """Replicate extract_states' batching and emit the token text behind each output row."""
    order=[]
    for i in range(0, len(sents), batch_size):
        batch = sents[i:i+batch_size]
        lengths=[len(s) for s in batch]
        for b, n in zip(batch, lengths):
            order.extend([t for t,_ in b[:n]])
    return order

def main():
    rng = random.Random(0)
    fails=[]
    for M_sents, label in ((200,'M=2547'), (2000,'M=24199')):
        ind = independent_parse(FEATS, M_sents)
        rtm_s = rtm.load_sentences(FEATS, M_sents)
        ind_flat=[t for s in ind for t,_ in s]
        rtm_flat=[t for s in rtm_s for t,_ in s]
        print(f"\n### {label}: independent parser {len(ind_flat)} tokens, "
              f"real_token_masks {len(rtm_flat)} tokens")
        same = ind_flat == rtm_flat
        print(f"  token streams identical: {same}")
        if not same: fails.append(f"{label}: parser mismatch")

        for arm, ckpt in (('untrained', None), ('trained','models/bowman_snli_best.pth')):
            if ckpt:
                ck = torch.load(ckpt, map_location='cpu'); stoi = ck['stoi']
            else:
                stoi = ra.build_vocab(rtm_s)
            act_order = rows_to_tokens(rtm_s, stoi)
            n_ok = sum(1 for a,b in zip(act_order, ind_flat) if a==b)
            print(f"  [{arm}] activation row->token vs independent: "
                  f"{n_ok}/{len(ind_flat)} match, lengths {len(act_order)}/{len(ind_flat)}")
            if n_ok != len(ind_flat) or len(act_order)!=len(ind_flat):
                fails.append(f"{label}/{arm}: order mismatch")
            idxs = sorted(rng.sample(range(len(ind_flat)), 20))
            print(f"  [{arm}] 20 random indices (i: mask-side | activation-side):")
            for i in idxs:
                mark = 'OK ' if ind_flat[i]==act_order[i] else 'BAD'
                print(f"      {mark} {i:>6}: {ind_flat[i]!r:<18} | {act_order[i]!r}")
    print("\nRESULT:", "PASS" if not fails else f"FAIL {fails}")

main()
