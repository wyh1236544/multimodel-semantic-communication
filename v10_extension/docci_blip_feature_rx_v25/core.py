"""Fixed RX-only experiment contract and bounded, immutable artifact I/O."""
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
import numpy as np
FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parents[1]
SCHEMA = 'docci-blip-feature-rx-v25'
ARMS = ('self', 'visual')
CHANNELS = (None, 0, -5)
REGIMES = {'clean':(None,None), 'both0':(0,0), 'text0':(None,0),
           'text_m5':(None,-5), 'both_m5':(-5,-5)}
PRIMARY = 'both0'
TRAIN_REGIMES = ('clean', 'both0', 'text0')
CONTROLS = {'base':('matched',), 'caption':('matched',), 'self':('matched',),
            'visual':('matched','shuffled','gray','zero','disabled')}

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',',':')).encode()).hexdigest()
def seed(*parts): return int(digest(parts)[:16],16) % (2**31-1)
def array_sha(v):
    return hashlib.sha256(str((v.shape,str(v.dtype))).encode()+v.tobytes()).hexdigest()
def local(root,name):
    p = Path(name)
    if p.is_absolute() or not p.parts or '..' in p.parts: raise ValueError('Unsafe relative path')
    q = Path(root)/p
    if any(x.is_symlink() for x in [q]+list(q.parents)): raise ValueError('Linked path forbidden')
    if Path(root).resolve() not in q.resolve().parents: raise ValueError('Escaped path')
    return q
def safe(path):
    p = Path(path).absolute()
    if '..' in p.parts or PROJECT.resolve() not in p.resolve().parents:
        raise ValueError('Output must be inside current project, not its root')
    if any(q.is_symlink() for q in [p]+list(p.parents)): raise ValueError('Linked output forbidden')
    return p
def atomic(path,value):
    p = Path(path)
    # Reject unsupported/nonfinite metadata BEFORE touching output files.
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if p.exists(): raise FileExistsError('Existing output preserved '+str(p))
    tmp = p.with_name(p.name+'.writing')
    with tmp.open('x',encoding='utf-8') as f: f.write(text)
    tmp.rename(p)
def line(f,value):
    f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n'); f.flush()
def config(profile):
    c = read(FOLDER/'config.json')
    if profile not in c['profiles'] or c['schema'] != SCHEMA: raise ValueError('Unknown fixed profile')
    return dict(c,profile=profile,**c['profiles'][profile])
def regimes(c): return tuple(REGIMES) if c['profile']=='pilot' else TRAIN_REGIMES
def donor_map(ids):
    if len(ids)<2 or len(set(ids))!=len(ids): raise ValueError('Need distinct within-cohort donors')
    return {eid:ids[(i+1)%len(ids)] for i,eid in enumerate(ids)}
def choose(index,c):
    result = {}
    for cohort,n,want in [('train',c['train'],8192),('dev',c['dev'],100)]:
        rs = [r for r in index['records'] if r['cohort']==cohort]
        if len(rs)!=want or any(r['official_split']!='train' or r['input_truncated'] for r in rs):
            raise ValueError('Need approved complete V20 TRAIN/dev cohorts')
        rs.sort(key=lambda r:seed(42,'V25-fixed-'+cohort,r['example_id']))
        result[cohort] = rs[:n]
    for key in ('example_id','text_sha256','image_sha256'):
        if {r[key] for r in result['train']} & {r[key] for r in result['dev']}:
            raise ValueError('TRAIN/dev leakage '+key)
    return result
def stream(c):
    step = 0
    for epoch in range(c['epochs']):
        order = list(range(c['train'])); random.Random(seed(42,'V25-order',epoch)).shuffle(order)
        for start in range(0,len(order),c['accumulation']):
            if step>=c['updates']: return
            ix = order[start:start+c['accumulation']]
            # Each arm consumes the SAME observed memories/visual features.
            rs = [TRAIN_REGIMES[seed(42,'V25-regime',epoch,i)%len(TRAIN_REGIMES)] for i in ix]
            yield step,epoch,ix,rs
            step += 1
def expected_keys(c,entries):
    return {(r['example_id'],regime,arm,ctrl) for r in entries for regime in regimes(c)
            for arm,controls in CONTROLS.items() for ctrl in controls}
def key(r): return r['example_id'],r['regime'],r['arm'],r['control']
def source_check():
    m = read(FOLDER/'MANIFEST.json')
    for base,files in ((FOLDER,m['files']),(PROJECT,m['dependencies'])):
        for n,h in files.items():
            if sha(local(base,n))!=h: raise ValueError('Pinned source differs '+n)
    return m
def snapshot(root):
    m = source_check(); p = Path(root)/'source_snapshot'; p.mkdir()
    for n in list(m['files'])+['MANIFEST.json']: shutil.copyfile(FOLDER/n,p/n)
def seal(root,stage):
    root = Path(root)
    if (root/'FAILED.json').exists(): raise ValueError('Cannot complete a failed stage')
    files = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink(): raise ValueError('No linked artifacts')
        if p.is_file(): files[p.relative_to(root).as_posix()] = dict(sha256=sha(p),bytes=p.stat().st_size,
            heavy=p.suffix in ('.npz','.pth'))
    atomic(root/'COMPLETE.json',dict(schema=SCHEMA,stage=stage,files=files,
        source_manifest_sha256=sha(FOLDER/'MANIFEST.json'),facts='PENDING',semantic_success=False))
def verify_files(root,stage,arrays=True):
    root = Path(root).resolve(strict=True); v = read(root/'COMPLETE.json')
    if (v['schema']!=SCHEMA or v['stage']!=stage or v['source_manifest_sha256']!=sha(FOLDER/'MANIFEST.json')
            or v['facts']!='PENDING' or v['semantic_success'] is not False): raise ValueError('Stage identity differs')
    wanted = {n:s for n,s in v['files'].items() if arrays or not s['heavy']}
    found = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and
        p.relative_to(root).as_posix() not in ('COMPLETE.json','VERIFIED.json') and
        (arrays or p.suffix not in ('.npz','.pth'))}
    if found!=set(wanted): raise ValueError('Missing or unexpected stage files')
    for n,s in wanted.items():
        p = local(root,n)
        if sha(p)!=s['sha256'] or p.stat().st_size!=s['bytes']: raise ValueError('Artifact differs '+n)
    return v
def save_npz(path,**arrays):
    if any(not np.isfinite(x).all() for x in arrays.values()): raise ValueError('Nonfinite array')
    with Path(path).open('xb') as f: np.savez_compressed(f,**arrays)
    return sha(path)
class Budget:
    def __init__(self,hours,gpu=False):
        if not math.isfinite(hours) or not 0<hours<=12: raise ValueError('Hours must be in (0,12]')
        self.deadline = time.monotonic()+hours*3600; self.gpu = gpu
    def __call__(self):
        if time.monotonic()>=self.deadline: raise TimeoutError('Stage budget exhausted; no retry')
        if shutil.disk_usage(PROJECT).free<4*2**30: raise RuntimeError('Project disk reserve crossed')
        if self.gpu:
            import torch
            if torch.cuda.mem_get_info()[0]<4*2**30: raise RuntimeError('GPU reserve crossed; no automatic changes')
