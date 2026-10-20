"""Read-only V25 checkpoint diagnosis. No training or model-quality promotion."""
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from docci_blip_feature_rx_v25.core import (sha,read,rows,atomic,line,array_sha,seed,
    safe,local,REGIMES,CHANNELS,donor_map,digest)
FOLDER=Path(__file__).resolve().parent
PROJECT=FOLDER.parents[1]
SCHEMA='docci-rx-diagnose-v26'
CONTROLS=(('base','matched'),('self','matched'),('visual','matched'),
          ('visual','shuffled'),('visual','gray'),('visual','zero'),('visual','disabled'))
PINS={
    'PIPELINE.json':'1192e82e091dd348e9aed17cd4bb75385179a9722b360cab7c5f208c9d38974c',
    'cache/index.json':'3e06549dd80f3b89e93e352d0141ec730f581dcc49b2dfd1d874fd8e3affe8dd',
    'cache/COMPLETE.json':'cd1221e9e98d26fe03b16fff40fcb2ba97e9c90854f25d771783058ee3599d7a',
    'train_self/COMPLETE.json':'7e54aa3aa02ee376961c98913ab535183afa1b154225136210cb1e438fd70d41',
    'train_visual/COMPLETE.json':'2605f00a8aa2c7fb764fecc8d34d65c7c4e2e7dfa96de0e1e8193217cea66d3f',
    'evaluation/COMPLETE.json':'8ecefef4c5f083a35166eba004ff22cec7d3824c3c9ae2b8c34f6be8a52e3c8f',
    'evaluation/rows.jsonl':'4822764b41c16c98f9ab6b6aac6d7b05a21df65f8164fbd5585487dce29ae41d',
    'train_self/last.pth':'81dac36e10bf00a619ccbb8a01d0b169985d3cf52bdaf3d26ce5f17fdd78fe04',
    'train_visual/last.pth':'8890fe8525469c483e8ff26a34c9fbaade0956959348c76c58a00dbdaaf214f0'
}

def config(profile):
    c=read(FOLDER/'config.json')
    if c['schema']!=SCHEMA or profile not in c['profiles']:raise ValueError('Unknown fixed profile')
    return dict(c,profile=profile,**c['profiles'][profile])

def selected(index,profile):
    """Fixed selections independent of V25 outcomes; DEV donor assignment is unchanged."""
    cfg=config(profile);entries=index['records']
    train=[r for r in entries if r['cohort']=='train'];dev=[r for r in entries if r['cohort']=='dev']
    if len(train)!=512 or len(dev)!=32:raise ValueError('Pinned complete V25 pilot cohorts required')
    if any(r['official_split']!='train' for r in entries):raise ValueError('Only prior TRAIN-source cohorts')
    train.sort(key=lambda r:seed(42,'V26-fixed-TRAIN',r['example_id']))
    full_train=train[:16]
    donors=dict(donor_map([r['example_id'] for r in full_train]),
                **donor_map([r['example_id'] for r in dev]))
    return dict(train=full_train[:cfg['train']],dev=dev[:cfg['dev']]),donors

def matrix(refs,c):
    return {(r['example_id'],regime,arm,control) for r in refs
            for regime in c['regimes'] for arm,control in CONTROLS}

def key(row):return row['example_id'],row['regime'],row['arm'],row['control']

def reference_counts(ids):
    a=np.asarray(ids)
    if a.shape!=(9,64) or a.dtype.kind not in 'iu':raise ValueError('Expected fixed 9x64 gold labels')
    content=[]
    for s in a:
        if s[0]!=0 or list(s).count(2)!=1:raise ValueError('Invalid gold framing')
        end=list(s).index(2)
        if (s[1:end]<=2).any() or (s[end+1:]!=1).any():raise ValueError('Invalid content/PAD')
        content.append(end-1)
    return [dict(all=n+2,content=n,eos=1,bos=1) for n in content]

def close(a,b):return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-7)

def source_check():
    m=read(FOLDER/'MANIFEST.json')
    for base,files in ((FOLDER,m['files']),(PROJECT,m['dependencies'])):
        for n,h in files.items():
            if sha(local(base,n))!=h:raise ValueError('Fixed source changed '+n)
    return m

def snapshot(root):
    source_check();p=Path(root)/'source_snapshot';p.mkdir()
    for n in list(read(FOLDER/'MANIFEST.json')['files'])+['MANIFEST.json']:shutil.copyfile(FOLDER/n,p/n)

def seal(root):
    root=Path(root)
    if (root/'FAILED.json').exists():raise ValueError('Failed output cannot be completed')
    files={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    atomic(root/'COMPLETE.json',dict(schema=SCHEMA,files=files,training_updates=0,
        facts='PENDING',semantic_success=False,source_manifest_sha256=sha(FOLDER/'MANIFEST.json')))

def verify_files(root):
    root=Path(root);v=read(root/'COMPLETE.json')
    if (v['schema']!=SCHEMA or v['training_updates']!=0 or v['semantic_success'] or v['facts']!='PENDING' or
            v['source_manifest_sha256']!=sha(FOLDER/'MANIFEST.json')):raise ValueError('Completion identity differs')
    if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('No linked results')
    found={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='COMPLETE.json'}
    if found!=set(v['files']):raise ValueError('Missing/unexpected diagnostic files')
    for n,h in v['files'].items():
        if sha(local(root,n))!=h:raise ValueError('Diagnostic artifact differs '+n)
    if sha(root/'source_snapshot/MANIFEST.json')!=sha(FOLDER/'MANIFEST.json'):
        raise ValueError('Executed source manifest differs')
    for n,h in read(FOLDER/'MANIFEST.json')['files'].items():
        if sha(root/'source_snapshot'/n)!=h:raise ValueError('Executed source bytes differ')
    run=read(root/'RUN.json')
    if run['schema']!=SCHEMA or run['training_updates']!=0:raise ValueError('Run identity differs')
    return v

class Budget:
    def __init__(self,hours,gpu=False):
        if not math.isfinite(hours) or not 0<hours<=4:raise ValueError('Hours must be in (0,4]')
        self.deadline=time.monotonic()+hours*3600;self.gpu=gpu
    def __call__(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('Diagnostic budget exhausted')
        if shutil.disk_usage(PROJECT).free<2**30:raise RuntimeError('Project disk reserve crossed')
        if self.gpu:
            import torch
            if torch.cuda.mem_get_info()[0]<2*2**30:raise RuntimeError('GPU reserve crossed; do not stop other jobs')
