"""PRIVATE light records, no arrays/checkpoints; independently verify before copying."""
import argparse
from pathlib import Path
import shutil
import sys
import tarfile
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_evidence_v23.core import FOLDER,read,sha,atomic,safe,source_check,Budget,verify_files
from docci_evidence_v23.report import verify,render

def collect(destination,prepared_root,stages,hours):
    source=source_check();budget=Budget(hours)
    destination=safe(destination);staging=destination.with_name(destination.name+'.contents')
    if destination.exists() or staging.exists():raise FileExistsError('Existing collection preserved; use collect02')
    targets={}
    if prepared_root:targets[sha(Path(prepared_root)/'COMPLETE.json')]=Path(prepared_root)
    targets.update({sha(Path(p)/'COMPLETE.json'):Path(p) for p in stages})
    checks=[]
    for path in ([prepared_root] if prepared_root else [])+stages:
        receipt=read(Path(path)/'COMPLETE.json');target=None
        if receipt['stage']=='score':target=targets.get(read(Path(path)/'identity.json')['target_sha256'])
        checks.append(verify(path,prepared_root,target,budget=budget))
    staging.mkdir(parents=True);mapping=[]
    if prepared_root:mapping.append((Path(prepared_root),'prepared'))
    mapping.extend((Path(p),'stage_{:02d}'.format(i+1)) for i,p in enumerate(stages))
    total=0
    for original,label in mapping:
        out=staging/label;out.mkdir();original_hashes={p.relative_to(original).as_posix():sha(p) for p in original.rglob('*') if p.is_file()}
        for name,h in original_hashes.items():
            budget();p=original/name
            if p.suffix in ('.npz','.pth','.pt','.bin','.safetensors'):continue
            total+=p.stat().st_size
            if total>4*2**30:raise ValueError('Light record cap4GiB exceeded')
            target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
            if sha(target)!=h:raise ValueError('Collection copy differs')
        if any(sha(original/n)!=h for n,h in original_hashes.items()):raise ValueError('Readonly source changed')
    (staging/'source').mkdir()
    for n in list(source['files'])+['MANIFEST.json']:shutil.copyfile(FOLDER/n,staging/'source'/n)
    if prepared_root and all(read(Path(p)/'COMPLETE.json')['stage']!='calibrate' for p in stages):
        render(staging/'prepared',[staging/label for _,label in mapping if label!='prepared'],staging/'display',budget)
    atomic(staging/'PRIVATE.json',dict(raw_DOCCI=True,public_upload_allowed=False,large_arrays_kept_on_server=True,server_verification=checks))
    hashes={p.relative_to(staging).as_posix():sha(p) for p in staging.rglob('*') if p.is_file()}
    atomic(staging/'FILE_HASHES.json',hashes)
    with destination.open('xb') as f,tarfile.open(fileobj=f,mode='w:gz',format=tarfile.USTAR_FORMAT) as tf:
        for p in sorted(staging.rglob('*')):
            budget()
            if p.is_file():tf.add(p,arcname='v23_records/'+p.relative_to(staging).as_posix(),recursive=False)
    with tarfile.open(destination,'r:gz') as tf:
        members=tf.getmembers()
        if len(members)!=len(hashes)+1 or any(not m.isfile() for m in members):raise ValueError('Archive matrix differs')
        import hashlib
        for m in members:
            n=Path(m.name).relative_to('v23_records').as_posix()
            expected=sha(staging/'FILE_HASHES.json') if n=='FILE_HASHES.json' else hashes[n]
            if hashlib.sha256(tf.extractfile(m).read()).hexdigest()!=expected:raise ValueError('Archive bytes differ')
    print(str(destination));print('SHA256 '+sha(destination),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('destination');p.add_argument('--prepared');p.add_argument('--stages',nargs='*',default=[])
    p.add_argument('--max-hours',type=float,default=1.);a=p.parse_args()
    if not a.prepared and not a.stages:p.error('At least one complete stage required')
    collect(a.destination,a.prepared,a.stages,a.max_hours)
