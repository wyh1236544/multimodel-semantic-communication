"""Independent bounded CPU collection. Original experiment is read-only."""
import argparse
import gzip
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_semantic_probe_v12.core import safe,read,digest,check_sources
from docci_semantic_probe_v12.report import verify,render


def collect(root,destination,max_minutes=30):
    root=Path(root).resolve(strict=True); destination=safe(destination)
    if not destination.name.endswith('.tar.gz') or not 0<max_minutes<=120: raise ValueError('Need NEW tar.gz and bounded CPU minutes')
    try: destination.relative_to(root)
    except ValueError: pass
    else: raise ValueError('Archive must be OUTSIDE immutable source experiment')
    staging=destination.with_name(destination.name+'.contents'); checksum=destination.with_name(destination.name+'.sha256')
    if any(p.exists() for p in (destination,staging,checksum)): raise FileExistsError('Existing collection preserved; use collect02 name')
    start=time.monotonic()
    def budget():
        if time.monotonic()-start>max_minutes*60: raise TimeoutError('CPU collection cap reached; no GPU retraining needed')
    check_sources(); result=verify(root); budget()
    staging.mkdir(parents=True,exist_ok=False); files={}
    allowed={'.json','.jsonl','.npz','.png','.py','.md','.log'}
    for original in sorted(root.rglob('*')):
        budget()
        if original.is_symlink(): raise ValueError('Symlink result rejected')
        if not original.is_file() or original.suffix not in allowed: continue
        relative=original.relative_to(root); before=digest(original); target=staging/relative
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(original,target)
        if digest(original)!=before or digest(target)!=before: raise ValueError('Original changed during collection')
        files[str(relative)]=before
    # All original GPU records remain untouched. Only the new staging tree gets derived reports.
    with (staging/'verified_analysis.json').open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    with (staging/'index.html').open('x',encoding='utf-8') as f: f.write(render(staging,result))
    with (staging/'COLLECTION.json').open('x',encoding='utf-8') as f:
        json.dump(dict(original_run=str(root),original_files_sha256=files,source_weights_excluded=True,
            no_torch_no_pickle_no_GPU_no_training=True,original_run_modified=False,
            CPU_wall_seconds=time.monotonic()-start,engineering_verification_NOT_model_success=True),f,indent=2)
    with destination.open('xb') as out,gzip.GzipFile(fileobj=out,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for file in sorted(staging.rglob('*')):
            budget()
            if file.is_file():
                data=file.read_bytes(); member=tarfile.TarInfo('v10_docci_semantic_probe_v12_records/'+str(file.relative_to(staging)))
                member.size=len(data); member.mode=0o644; tar.addfile(member,io.BytesIO(data))
    sha=digest(destination)
    with checksum.open('x') as f: f.write(sha+'  '+destination.name+'\n')
    print('Send '+str(destination),flush=True); print('SHA256 '+sha,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('destination'); p.add_argument('--max-minutes',type=float,default=30)
    a=p.parse_args(); collect(a.root,a.destination,a.max_minutes)
