"""Verified source-only/results bundle; no model checkpoint or backbone weight."""
import argparse
import gzip
import html
import io
from pathlib import Path
import sys
import tarfile
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from docci_modal_v9.core import safe_write,read,atomic_json
from docci_modal_v9.report import verify


def collect(root,destination):
    root=safe_write(root).resolve(strict=True); destination=safe_write(destination)
    if destination.exists(): raise FileExistsError('Existing record archive preserved')
    launch=read(root/'launch_identity.json'); branches=launch['requested_branches']; statuses={}
    for branch in branches:
        name='v10_smoke_'+branch+'512' if launch['action']=='smoke' else 'v10_'+launch['action']+'_'+branch
        result=verify(root/name); statuses[branch]=dict(status=result['identity']['status'],overfit_gate=result['identity']['overfit_gate'],
            clean_gate=result['identity'].get('clean_gate'),folder=name)
    for folder in root.iterdir():
        if folder.is_dir() and (folder/'run_identity.json').exists(): verify(folder)
    aggregate=dict(schema='docci-modal-v9-records',requested_branches=branches,branches=statuses,
        engineering_verified=True,quality_stop_retained=True,source_only_plus_records=True,checkpoints_included=False,
        scope='independent private branches, not Common/joint gain; TRAIN4 diagnosis and VAL8 exploratory only')
    if (root/'aggregate.json').exists():
        if read(root/'aggregate.json')!=aggregate: raise ValueError('Existing aggregate differs')
    else: atomic_json(root/'aggregate.json',aggregate)
    if not (root/'index.html').exists():
        with (root/'index.html').open('x',encoding='utf-8') as f:
            f.write('<!doctype html><meta charset="utf-8"><h1>DOCCI 独立分支</h1><p>两条分支分别验收，不代表联合增益。</p>')
            for branch,record in statuses.items(): f.write('<p><a href="'+html.escape(record['folder'])+'/actual_recovery.html">'+html.escape(branch+' — '+record['status'])+'</a></p>')
    allowed={'.json','.jsonl','.html','.png','.svg','.npz','.md','.py','.txt','.log'}
    files=[]
    for file in sorted(root.rglob('*')):
        if file.is_symlink(): raise ValueError('Symlink record rejected')
        if file.is_file() and file.suffix in allowed and '.cache' not in file.parts: files.append(file)
    prefix='v10_docci_modal_v9_records'
    with destination.open('xb') as out,gzip.GzipFile(fileobj=out,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for file in files:
            data=file.read_bytes(); entry=tarfile.TarInfo(prefix+'/'+str(file.relative_to(root))); entry.size=len(data); entry.mode=0o644
            tar.addfile(entry,io.BytesIO(data))
    print('Verified {} branch(es); record bundle {} bytes; no weights.'.format(len(branches),destination.stat().st_size))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('destination'); a=p.parse_args(); collect(a.root,a.destination)
