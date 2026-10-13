"""Pure NumPy reporting, matching the historical full-source text definitions."""
import math
from collections import Counter
import numpy as np
from final_compare_v1.metrics import caption_metrics


def metrics(source,covered,received):
    if source!=covered: raise ValueError('V8 never scores a silently shortened target')
    full=caption_metrics(source,received); a=source.lower().strip().split(); b=received.lower().strip().split(); precisions=[]
    for n in (1,2):
        aa=Counter(tuple(a[i:i+n]) for i in range(max(0,len(a)-n+1)))
        bb=Counter(tuple(b[i:i+n]) for i in range(max(0,len(b)-n+1)))
        precisions.append((sum(min(v,aa[k]) for k,v in bb.items())+1.)/(sum(bb.values())+1.))
    bleu=0. if not a or not b else min(1.,math.exp(1.-len(a)/len(b)))*math.sqrt(precisions[0]*precisions[1])
    return dict(full,bleu2=bleu,covered_wer=full['normalized_wer'],covered_cer=full['text_cer'],
        covered_word_errors=full['normalized_word_errors'],covered_source_words=full['normalized_source_words'])


def summarize(rows):
    if not rows: raise ValueError('Empty metric group')
    def mean(k): return float(np.mean([r[k] for r in rows]))
    return dict(entries=len(rows),full_wer=mean('normalized_wer'),full_exact_fraction=mean('text_exact'),
        full_corpus_wer=sum(r['normalized_word_errors'] for r in rows)/max(1,sum(r['normalized_source_words'] for r in rows)),
        covered_wer=mean('covered_wer'),covered_cer=mean('covered_cer'),bleu2=mean('bleu2'),
        capacity_stop_fraction=mean('stopped_by_capacity'),truncated_fraction=mean('input_truncated'))


def paired(values,seed):
    x=np.asarray(values,float)
    if not len(x) or not np.isfinite(x).all(): raise ValueError('Invalid paired differences')
    rng=np.random.default_rng(seed); sample=x[rng.integers(len(x),size=(2000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(sample,[.025,.975]).tolist(),pairs=len(x),
        scope='paired image clusters, one trained replicate, fixed held-out validation paths; exploratory NOT independent test')
