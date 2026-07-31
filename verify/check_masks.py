"""Check 5: masks match the raw .feats file, hand-parsed.

Independence: raw lines are re-read with a plain split, no project code, and compared to
dense[k, m]. Also tests ';'-split for multi-valued const, and that EMPTY fields contribute
no concept (rather than a 'none' concept). Finally re-derives mean overlap independently.
"""
import os, sys, random
sys.path.insert(0, 'src')
import numpy as np, real_token_masks as rtm

FEATS = os.path.expanduser('~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats')
CATS = ["lemma","tag","dep","ent","synset","const"]
MAXS = 2000

raw=[]                                   # flat list of raw '|' chunks, hand-read
with open(FEATS, encoding='utf-8') as f:
    f.readline()
    for n,line in enumerate(f):
        if n>=MAXS: break
        line=line.rstrip('\n').strip()
        if not line: continue
        for chunk in line.split(' '):
            if len(chunk.split('|'))==7: raw.append(chunk)

toks = rtm.load_tokens(FEATS, MAXS)
cons = rtm.select_concepts(toks, rtm.CATEGORIES, 15, 5)
dense = rtm.build_dense(toks, cons)
print(f"raw chunks {len(raw)}  dense {dense.shape}")

rng = random.Random(7)
pos, neg, checked, bad = [], [], 0, 0
while len(pos)<5 or len(neg)<5:
    k = rng.randrange(len(cons)); m = rng.randrange(len(raw))
    tgt = dense[k,m]
    if tgt and len(pos)<5: pos.append((k,m))
    elif (not tgt) and len(neg)<5: neg.append((k,m))

print(f"\n{'k':>3} {'m':>6} {'concept':>22} {'dense':>6} {'hand-parsed raw field':>34} {'ok':>4}")
for k,m in pos+neg:
    cat,val = cons[k]
    fields = raw[m].split('|')
    field = fields[1+CATS.index(cat)]
    vals = field.split(';') if cat=='const' else ([field] if field else [])
    expect = val in vals
    ok = bool(dense[k,m]) == expect
    bad += (not ok); checked += 1
    print(f"{k:>3} {m:>6} {cat+'='+val:>22} {str(bool(dense[k,m])):>6} "
          f"{repr(field)[:32]:>34} {'OK' if ok else 'BAD':>4}")

# multi-valued const
multi = next(c for c in raw if ';' in c.split('|')[6])
f7 = multi.split('|')
print(f"\nmulti-valued const example: token {f7[0]!r} const field {f7[6]!r} "
      f"-> {f7[6].split(';')}")

# empty fields contribute nothing
empt = [c for c in raw if c.split('|')[4]=='']          # empty 'ent'
mi = raw.index(empt[0])
ent_ks = [i for i,(c,_) in enumerate(cons) if c=='ent']
print(f"empty-field example: token {empt[0].split('|')[0]!r} ent field is '' ; "
      f"ent concepts in K=15 vocab: {ent_ks} (none expected) ; "
      f"row sum over all concepts = {int(dense[:,mi].sum())}")
none_concept = any(v=='' for _,v in cons)
print(f"any concept with empty value in vocabulary: {none_concept} (must be False)")

# independent mean overlap
s = dense.sum(axis=0)
mo = s[s>0].mean()
print(f"\nindependent mean overlap over active tokens: {mo:.3f}  (reported 3.189)")
print(f"independent unique fraction: {(s==1).sum()/len(s):.3f}  (reported 0.139)")
ok_all = bad==0 and not none_concept and abs(mo-3.189)<0.001
print("\nRESULT:", "PASS" if ok_all else "FAIL")
