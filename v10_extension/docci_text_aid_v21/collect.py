"""PRIVATE light collection, including honest failed/partial evidence; never weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_text_aid_v21.core import source_check,safe,read,sha

def collect(root,destination):
    source_check();root=safe(root);destination=safe(destination)
    if destination.exists() or destination.with_name(destination.name+'.sha256').exists() or root==destination or root in destination.parents:
        raise FileExistsError('NEW archive outside run directory required')
    state=read(root/'PIPELINE.json')
    if state['status']=='RUNNING':
        raise ValueError('Run still active; do not collect moving outputs')
    if state['status']=='COMPLETE':
        from docci_text_aid_v21.report import verify
        for stage in ('smoke','main'):
            verify(root/stage,False)
    mapping={}
    for p in root.rglob('*'):
        if p.is_symlink():
            raise ValueError('Symlink in run')
        if p.is_file() and p.suffix in ('.json','.jsonl','.log','.md','.py','.html','.png'):
            if p.name=='collection.log':
                continue  # Current collector log is still moving.
            mapping[p.relative_to(root).as_posix()]=p
    if sum(p.stat().st_size for p in mapping.values())>512*2**20:
        raise RuntimeError('Light archive exceeds512MiB; no silent image/sample filtering')
    hashes={n:sha(p) for n,p in mapping.items()}
    receipt=json.dumps(dict(schema='PRIVATE-v21-records',status=state['status'],
        files=hashes,large_arrays_and_weights_included=False),indent=2).encode()+b'\n'
    with destination.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz',format=tarfile.USTAR_FORMAT) as tf:
            import io
            for name,p in mapping.items():
                info=tarfile.TarInfo('v21_records/'+name);info.size=p.stat().st_size;info.mode=0o644
                with p.open('rb') as f:
                    tf.addfile(info,f)
            info=tarfile.TarInfo('v21_records/RECORDS_MANIFEST.json');info.size=len(receipt);info.mode=0o644
            tf.addfile(info,io.BytesIO(receipt))
    with tarfile.open(destination,'r:gz') as tf:
        members=tf.getmembers()
        if len(members)!=len(mapping)+1 or any(not m.isfile() for m in members):
            raise ValueError('Archive matrix differs')
        for m in members:
            name=m.name[len('v21_records/'):]
            expected=hashlib.sha256(receipt).hexdigest() if name=='RECORDS_MANIFEST.json' else hashes[name]
            if hashlib.sha256(tf.extractfile(m).read()).hexdigest()!=expected:
                raise ValueError('Archived bytes differ')
    with destination.with_name(destination.name+'.sha256').open('x',encoding='utf-8') as f:
        f.write(sha(destination)+'  '+destination.name+'\n')
    print('PRIVATE records '+str(destination),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('destination');a=p.parse_args()
    collect(a.root,a.destination)
