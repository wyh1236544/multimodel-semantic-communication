"""Exact Levenshtein, legacy scoring, source-length diagnostics; no Torch/RX hints."""
from collections import Counter
import math
import re
import statistics
import unicodedata


def distance(a,b):
    # Myers bit-vector algorithm: exact unit-cost Levenshtein, not approximate.
    if len(a)>len(b): a,b=b,a
    n=len(a)
    if not n: return len(b)
    eq={}
    for i,x in enumerate(a): eq[x]=eq.get(x,0)|(1<<i)
    mask=(1<<n)-1; high=1<<(n-1); vp=mask; vn=0; score=n
    for x in b:
        e=eq.get(x,0); xv=e|vn; xh=(((e&vp)+vp)^vp)|e
        ph=vn|~(xh|vp); mh=vp&xh
        score+=int(bool(ph&high))-int(bool(mh&high))
        ph=((ph<<1)|1)&mask; mh=(mh<<1)&mask
        vp=(mh|~(xv|ph))&mask; vn=ph&xv
    return score


def words(text): return re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold(),re.UNICODE)


def metrics(source,received):
    a,b=words(source),words(received); errors=distance(a,b)
    cer=distance(source,received)/max(len(source),1)
    rawwer=distance(source.split(),received.split())/max(len(source.split()),1)
    x,y=source.lower().strip().split(),received.lower().strip().split(); ps=[]
    for n in (1,2):
        aa=Counter(tuple(x[i:i+n]) for i in range(max(0,len(x)-n+1)))
        bb=Counter(tuple(y[i:i+n]) for i in range(max(0,len(y)-n+1)))
        ps.append((sum(min(v,aa[k]) for k,v in bb.items())+1)/(sum(bb.values())+1))
    bleu=0. if not x or not y else min(1.,math.exp(1.-len(x)/len(y)))*math.sqrt(ps[0]*ps[1])
    return dict(text_cer=cer,text_wer=rawwer,wer=rawwer,cer=cer,text_exact=source==received,
        normalized_word_errors=errors,normalized_source_words=len(a),normalized_wer=errors/max(len(a),1),
        normalization='NFKC+casefold+Unicode_word_tokens; raw WER/CER also reported',bleu2=bleu,
        covered_wer=errors/max(len(a),1),covered_cer=cer,covered_word_errors=errors,covered_source_words=len(a))


def body(seq):
    if not seq or seq[0]!=2: raise ValueError('Missing decoder start')
    result=[]; ended=False
    for token in seq[1:]:
        if token==2: ended=True; break
        if token not in (0,1): result.append(token)
    return result,ended


def diagnostic(rows):
    groups={k:[] for k in ('full62','partial','empty')}; first=[]
    for r in rows:
        source=r['source_tokens']-2
        if len(r['chunk_generated_token_ids'])!=9: raise ValueError('Nonpublic chunk count')
        for j,seq in enumerate(r['chunk_generated_token_ids']):
            expected=min(62,max(0,source-j*62)); generated,eos=body(seq)
            key='full62' if expected==62 else 'partial' if expected else 'empty'
            pairs=max(1,len(generated)-1)
            row=dict(expected=expected,generated=len(generated),has_EOS=eos,
                early_EOS=eos and len(generated)<expected,capacity_stop=not eos,
                length_deficit=max(0,expected-len(generated)),
                adjacent_repeat=sum(a==b for a,b in zip(generated,generated[1:]))/pairs,
                unique_fraction=len(set(generated))/max(1,len(generated)))
            groups[key].append(row)
            if j==0: first.append(row)
    summary={}
    for key,values in groups.items():
        summary[key]=dict(slots=len(values),early_EOS=sum(v['early_EOS'] for v in values),
            EOS_at_expected_length=sum(v['has_EOS'] and v['generated']==v['expected'] for v in values),
            capacity_stop=sum(v['capacity_stop'] for v in values),
            length_deficit=sum(v['length_deficit'] for v in values if v['early_EOS']),
            early_EOS_fraction=sum(v['early_EOS'] for v in values)/max(1,len(values)))
    return dict(slot_groups=summary,first_segment_adjacent_repeat=statistics.mean(v['adjacent_repeat'] for v in first),
        first_segment_unique_fraction=statistics.mean(v['unique_fraction'] for v in first),
        source_length_is_evaluation_only=True,capacity_stop_is_NOT_repetition=True,
        EOS_is_NOT_automatically_premature=True)


def summary(rows):
    x=[r['normalized_wer'] for r in rows]
    return dict(n=len(x),mean_WER=statistics.mean(x),SD_WER=statistics.stdev(x) if len(x)>1 else None,
        corpus_WER=sum(r['normalized_word_errors'] for r in rows)/max(1,sum(r['normalized_source_words'] for r in rows)),
        exact=sum(r['text_exact'] for r in rows),natural_EOS=sum(not r['stopped_by_capacity'] for r in rows),
        **diagnostic(rows))
