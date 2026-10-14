"""One complete independent verification, then atomic light archive; no GPU/weights."""
import argparse
import gzip
import io
from pathlib import Path
import sys
import tarfile
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_text_continue_v11.core import read,digest,safe_write,atomic_json
from docci_text_continue_v11.report import verify,finish


def collect(root,destination):
    root=safe_write(root).resolve(strict=True); destination=safe_write(destination)
    partial=destination.with_name(destination.name+'.part')
    if destination.exists() or partial.exists(): raise FileExistsError('Existing archive/partial preserved; new collect02/03 required')
    print('COLLECT full independent verification ONCE; no training',flush=True)
    result=verify(root)
    if (root/'COMPLETE.json').exists():
        done=read(root/'COMPLETE.json')
        if done['schema']!='docci-text-continue-v11' or not done['engineering_completion_NOT_quality'] or done['run_identity_sha256']!=digest(root/'run_identity.json') or done['summary_sha256']!=digest(root/'summary.json'): raise ValueError('Existing completion binding differs')
    else: finish(root,result)
    print('LEGACY QUALITY '+read(root/'summary.json')['legacy_quality_status']+' (rule unchanged)',flush=True)
    files=[]; allowed={'.json','.jsonl','.md','.py','.txt','.html','.npz','.log','.svg','.png'}
    for file in sorted(root.rglob('*')):
        if file.is_symlink(): raise ValueError('Symlink in records')
        if file.is_file() and file.suffix in allowed and '.cache' not in file.parts: files.append(file)
    print('COLLECT packing {} small files; weights/optimizer/RNG binaries stay server'.format(len(files)),flush=True)
    with partial.open('xb') as out,gzip.GzipFile(fileobj=out,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for i,file in enumerate(files):
            data=file.read_bytes(); member=tarfile.TarInfo('v10_docci_text_continue_v11_records/'+str(file.relative_to(root))); member.size=len(data); member.mode=0o644
            tar.addfile(member,io.BytesIO(data))
            if (i+1)%100==0: print('COLLECT files {}/{}'.format(i+1,len(files)),flush=True)
    partial.rename(destination)
    atomic_json(root/'progress.json',dict(stage='COMPLETE',record_archive=str(destination),archive_sha256=digest(destination)))
    print('Verified and packed '+str(destination),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('destination'); a=p.parse_args(); collect(a.root,a.destination)
