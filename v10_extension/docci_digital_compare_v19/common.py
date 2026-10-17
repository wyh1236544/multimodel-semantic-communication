"""No neural imports. Immutable inputs, bounded work, atomic new outputs."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np

FOLDER=Path(__file__).resolve().parent
PROJECT=FOLDER.parents[1]
CHECKPOINTS={
    'frozen':'3e6743eb1bcdbe4bed5db14ba9c9d5cb6d30cb2676066fc41c63fbb09ee3b000',
    'clean_continue':'4a0a97034e7b7c2ff48303bfbeadd31677542cccf95ee9ad5e9ba8b02e83f6b1',
    'noise_replay':'96b27347b62c918a931ba221640139ce3e162b8ff8fbcaebde89f6b13cc5312b'}

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p):
    with Path(p).open(encoding='utf-8') as f: return [json.loads(s) for s in f if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def safe(p):
    p=Path(p);p=p if p.is_absolute() else PROJECT/p
    if '..' in p.parts:raise ValueError('Traversal rejected')
    rel=p.absolute().relative_to(PROJECT.resolve())
    if not rel.parts:raise ValueError('Project root is not an output')
    at=PROJECT
    for x in rel.parts:
        at=at/x
        if at.is_symlink():raise ValueError('Symlink output rejected')
    return p
def atomic(p,value):
    p=safe(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError('Existing artifact preserved: '+str(p))
    tmp=p.with_name(p.name+'.part-'+str(os.getpid()))
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(str(tmp),str(p))
def progress(p,value):
    # Explicitly mutable live status only. Historical results are not rewritten.
    p=safe(p);tmp=p.with_name(p.name+'.part-'+str(os.getpid()))
    with tmp.open('x',encoding='utf-8') as f:json.dump(value,f,allow_nan=False)
    os.replace(str(tmp),str(p))
def load():
    c=read(FOLDER/'config.json')
    if c['schema']!='v19-text-digital-v1' or c['symbols']!=9216 or c['repeats']!=3:raise ValueError('Protocol identity differs')
    return c
def trials(c,smoke=False,negative=False):
    if negative:
        snrs=[-5] if smoke else c['negative_snr_db']
    else:
        yield None,0
        snrs=[0,-5] if smoke else c['snr_db']
    for snr in snrs:
        for repeat in range(1 if smoke else c['repeats']):yield snr,repeat
def check_source():
    m=read(FOLDER/'MANIFEST.json')
    for base,files in ((FOLDER,m['files']),(PROJECT,m['dependencies'])):
        for n,h in files.items():
            p=Path(n)
            if p.is_absolute() or '..' in p.parts or (base/p).is_symlink() or sha(base/p)!=h:raise ValueError('Source differs: '+n)
    return m
def snapshot(root):
    root=safe(root);m=check_source();root.mkdir(exist_ok=False)
    for n in list(m['files'])+['MANIFEST.json']:
        q=root/'source_snapshot'/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(FOLDER/n,q)
    for n in m['dependencies']:
        q=root/'dependency_snapshot'/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT/n,q)
def seal(root):
    root=safe(root)
    files={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='SEALED.json'}
    atomic(root/'SEALED.json',files)
def unseal(root):
    root=Path(root)
    for n,h in read(root/'SEALED.json').items():
        p=Path(n)
        if p.is_absolute() or '..' in p.parts or not p.parts or (root/p).is_symlink() or sha(root/p)!=h:raise ValueError('Sealed artifact changed: '+n)
class Budget:
    def __init__(self,hours,device=None):
        if not math.isfinite(hours) or not 0<hours<=8:raise ValueError('Finite 0..8h limit')
        self.deadline=time.monotonic()+hours*3600;self.device=device
    def __call__(self):
        if time.monotonic()>self.deadline:raise TimeoutError('Bounded stage time exhausted')
        if shutil.disk_usage(PROJECT).free<2*2**30:raise RuntimeError('Disk reserve crossed')
        if self.device is not None:
            import torch
            if torch.cuda.mem_get_info(self.device)[0]<4*2**30:raise RuntimeError('GPU reserve crossed; no auto retry')
def score(source,text):
    from docci_text_continue_v11.metrics import distance,words
    a,b=words(source),words(text);e=distance(a,b)
    return dict(normalized_word_errors=e,normalized_source_words=len(a),normalized_wer=e/max(1,len(a)),
                text_exact=source==text,adjacent_word_repeat=sum(x==y for x,y in zip(b,b[1:]))/max(1,len(b)-1))
def ci(values,draws=10000):
    x=np.asarray(values,dtype=float)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Nonempty finite observations required')
    ix=np.random.default_rng(42).integers(0,len(x),(draws,len(x)))
    return dict(mean=float(x.mean()),CI95=np.quantile(x[ix].mean(1),[.025,.975]).tolist())
