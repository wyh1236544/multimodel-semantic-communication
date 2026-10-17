"""Bounded deterministic CPU checks and PRIVATE light archive. No inference/training."""
import argparse
import io
import json
from pathlib import Path
import sys
import tarfile
EXT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_digital_compare_v19.common import safe,sha,read,atomic,Budget
from docci_digital_compare_v19.report import verify

def collect(root,destination,max_minutes=30,replay=False):
    root=safe(root);dest=safe(destination);part=dest.with_name(dest.name+'.part')
    if dest.exists() or part.exists() or dest.with_name(dest.name+'.sha256').exists():raise FileExistsError('Existing archive preserved')
    if not 0<max_minutes<=120:raise ValueError('Bounded 0..120min collection')
    budget=Budget(max_minutes/60);result=verify(root,budget,replay)
    report=root/'REPORT.json'
    if report.exists():
        old=read(report)
        # A previous full replay receipt can be reused for an identical sealed stage.
        if old!=result:raise ValueError('Existing report differs; preserve and inspect')
    else:atomic(report,result)
    if result['comparison'] is not None:
        from docci_digital_compare_v19.display import render
        render(root,result)
    files={}
    for p in sorted(root.rglob('*')):
        budget()
        if p.is_symlink():raise ValueError('Symlink archive rejected')
        if p.is_file():
            if p.suffix in ('.pth','.npz','.bin','.safetensors'):raise ValueError('Large model/data artifact must stay on server')
            files[p.relative_to(root).as_posix()]=sha(p)
    manifest=json.dumps(dict(schema='v19-private-records',files=files,status=result['status']),indent=2).encode()
    dest.parent.mkdir(parents=True,exist_ok=True)
    with part.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz',format=tarfile.PAX_FORMAT) as tf:
            for n in files:
                budget();tf.add(root/n,arcname='v19_records/'+n,recursive=False)
            info=tarfile.TarInfo('v19_records/ARCHIVE_MANIFEST.json');info.size=len(manifest);info.mode=0o600;tf.addfile(info,io.BytesIO(manifest))
    # Independently read every archived regular member, not only the outer gzip hash.
    with tarfile.open(part,'r:gz') as tf:
        members=tf.getmembers();expected={'v19_records/'+n:h for n,h in files.items()}
        if len(members)!=len(files)+1 or any(not m.isfile() for m in members):raise ValueError('Records member matrix differs')
        import hashlib
        for m in members:
            budget()
            if m.name in expected and hashlib.sha256(tf.extractfile(m).read()).hexdigest()!=expected[m.name]:raise ValueError('Archived bytes differ')
    part.rename(dest)
    with dest.with_name(dest.name+'.sha256').open('x',encoding='utf-8') as f:f.write(sha(dest)+'  '+dest.name+'\n')
    print('PRIVATE records: '+str(dest),flush=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('destination');p.add_argument('--max-minutes',type=float,default=30);p.add_argument('--replay-digital',action='store_true');a=p.parse_args()
    collect(a.root,a.destination,a.max_minutes,a.replay_digital)
