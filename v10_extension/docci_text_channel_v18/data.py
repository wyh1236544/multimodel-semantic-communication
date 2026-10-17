"""Read existing V14 source-feature caches, never old frozen TX wires."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np
from docci_text_channel_v18.core import read, sha, atomic, lines

def array_hash(x):
    return hashlib.sha256(x.tobytes()).hexdigest()

def entries(root, smoke=False):
    root = Path(root).resolve(strict=True)
    expected = read(Path(__file__).with_name('inputs.json'))
    for name,h in expected.items():
        if sha(root/name)!=h: raise ValueError('Approved V14 feature identity differs: '+name)
    result = {}
    for split,folder,n in [('train','cache_train',8 if smoke else 2048),('qual_dev','cache_validation',2 if smoke else 100)]:
        rows=read(root/(folder+'_identity.json'))['records']
        if len(rows)!=(2048 if split=='train' else 100): raise ValueError('Incomplete pinned cohort')
        result[split]=[]
        for r in rows[:n]:
            if r['official_split']!=split or not re.fullmatch(r'(train|qual_dev)_\d{5}',r['example_id']): raise ValueError('Wrong source split/ID')
            if hashlib.sha256(r['text'].encode()).hexdigest()!=r['text_sha256']: raise ValueError('Text identity changed')
            if not 2 <= r['source_tokens'] <= 512: raise ValueError('Full text capacity exceeded')
            result[split].append(dict(r,cache_file=str(root/folder/(r['example_id']+'.npz'))))
    a={r['text_sha256'] for r in result['train']}; b={r['text_sha256'] for r in result['qual_dev']}
    if a & b: raise ValueError('Train/validation text duplicate')
    return result

class Items:
    def __init__(self,records,lru=32): self.entries,self.lru,self.values=records,lru,OrderedDict()
    def __len__(self): return len(self.entries)
    def __getitem__(self,i):
        if i not in self.values:
            r=self.entries[i]; p=Path(r['cache_file'])
            if sha(p)!=r['cache_sha256']: raise ValueError('Source feature cache changed: '+r['example_id'])
            with np.load(p,allow_pickle=False) as z:
                if set(z.files)!={'meta','chunk_ids','chunk_hidden'}: raise ValueError('Cache schema differs')
                meta=json.loads(str(z['meta'])); ids=z['chunk_ids'].copy(); hidden=z['chunk_hidden'].copy()
            # V14 array_hash includes dtype and shape; import only the stdlib/NumPy core.
            from docci_baseline_v14.core import array_hash as old_hash
            if meta['identity']!=r['identity'] or meta['array_hashes']!={'chunk_ids':old_hash(ids),'chunk_hidden':old_hash(hidden)}: raise ValueError('Cache metadata/array hash differs')
            if ids.shape!=(9,64) or ids.dtype.kind not in 'iu' or hidden.shape!=(9,64,768) or hidden.dtype!=np.float32 or not np.isfinite(hidden).all(): raise ValueError('Source feature geometry differs')
            count=0
            for slot in ids:
                if slot[0]!=0 or list(slot).count(2)!=1 or (slot<0).any() or (slot>=50265).any(): raise ValueError('Source frame malformed')
                end=list(slot).index(2)
                if (slot[1:end]<=2).any() or not (slot[end+1:]==1).all(): raise ValueError('PAD/EOS framing differs')
                count+=end-1
            if count+2!=r['source_tokens']: raise ValueError('No-truncation source token coverage differs')
            self.values[i]=dict(r,ids=ids,hidden=hidden)
            while len(self.values)>self.lru: self.values.popitem(last=False)
        self.values.move_to_end(i)
        return self.values[i]

def audit(features,root,smoke,budget):
    records=entries(features,smoke)
    root.mkdir(exist_ok=False)
    for name in read(Path(__file__).with_name('inputs.json')):
        out=root/'pinned'/name;out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(features)/name,out)
    for split,rs in records.items():
        items=Items(rs,1)
        for i in range(len(items)):
            budget(); items[i]
            if i==0 or (i+1)%100==0 or i+1==len(items): print('READ-ONLY input audit {} {}/{}'.format(split,i+1,len(items)),flush=True)
        atomic(root/(split+'.json'),rs)
    from docci_text_channel_v18.core import FOLDER
    atomic(root/'INPUTS.json',dict(features=str(Path(features).resolve()),smoke=smoke,
        pinned_metadata=read(FOLDER/'inputs.json'),arrays_verified=True,
        no_cache_regeneration=True,qual_dev_reused=True,independent_final_test=False))
    from docci_text_channel_v18.core import seal
    seal(root)
    return records

def prepared(root,c):
    return {s:Items(read(Path(root)/(s+'.json')),c['lru']) for s in ('train','qual_dev')}
