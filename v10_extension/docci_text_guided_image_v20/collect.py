"""Independent full cache/weight hashes then PRIVATE, independently-read light tar."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT),str(EXT.parent)]
from docci_text_guided_image_v20.core import safe,sha,read,atomic,Budget
from docci_text_guided_image_v20.report import verify

def collect(root,destination,max_minutes=60):
    root,dest = safe(root),safe(destination)
    part = dest.with_name(dest.name+'.part')
    if dest.exists() or part.exists() or dest.with_name(dest.name+'.sha256').exists():
        raise FileExistsError('Existing archive preserved')
    if not 1<=max_minutes<=120:
        raise ValueError('Finite collection budget required')
    budget = Budget(max_minutes/60)
    result = verify(root,full=True,max_hours=max_minutes/60)
    if (root/'REPORT.json').exists():
        if read(root/'REPORT.json')!=result:
            raise ValueError('Existing differing report preserved')
    else:
        atomic(root/'REPORT.json',result)
    from docci_text_guided_image_v20.display import render
    render(root,result)
    files = {}
    for p in sorted(root.rglob('*')):
        budget()
        if p.is_symlink():
            raise ValueError('Symlink in run rejected')
        if p.is_file() and p.suffix not in ('.pth','.npz','.bin','.safetensors') and p.name!='collection.log':
            files[p.relative_to(root).as_posix()] = sha(p)
    manifest = json.dumps(dict(schema='v20-PRIVATE-records',files=files,status=result['status']),indent=2).encode()
    dest.parent.mkdir(parents=True,exist_ok=True)
    with part.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz',format=tarfile.PAX_FORMAT) as tf:
            for n in files:
                budget()
                tf.add(root/n,arcname='v20_records/'+n,recursive=False)
            info = tarfile.TarInfo('v20_records/ARCHIVE_MANIFEST.json')
            info.size,info.mode = len(manifest),0o600
            tf.addfile(info,io.BytesIO(manifest))
    with tarfile.open(part,'r:gz') as tf:
        members = tf.getmembers()
        if len(members)!=len(files)+1 or any(not m.isfile() for m in members):
            raise ValueError('Archive regular member matrix differs')
        for member in members:
            budget()
            name = member.name[len('v20_records/'):]
            if name in files and hashlib.sha256(tf.extractfile(member).read()).hexdigest()!=files[name]:
                raise ValueError('Archived light bytes differ')
    part.rename(dest)
    with dest.with_name(dest.name+'.sha256').open('x',encoding='utf-8') as f:
        f.write(sha(dest)+'  '+dest.name+'\n')
    print('PRIVATE records: '+str(dest),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('destination')
    p.add_argument('--max-minutes',type=float,default=60)
    a=p.parse_args()
    collect(a.root,a.destination,a.max_minutes)
