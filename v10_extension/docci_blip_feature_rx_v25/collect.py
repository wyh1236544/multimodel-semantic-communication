"""Private, lightweight verified records. Heavy arrays and weights stay on server."""
import argparse
import hashlib
import io
from pathlib import Path
import shutil
import sys
import tarfile
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT),str(EXT.parent)]
from docci_blip_feature_rx_v25.core import safe,sha,atomic,read,rows,source_check
from docci_blip_feature_rx_v25.report import verify_pipeline,render

def collect(root,destination):
    root = Path(root).resolve(strict=True); dest = safe(destination)
    staging = dest.with_name(dest.name+'.contents')
    if any(p.exists() for p in (dest,staging,dest.with_name(dest.name+'.sha256'))):
        raise FileExistsError('Existing collection preserved; use NEW name')
    if root in dest.parents or root in staging.parents: raise ValueError('Collection must be outside run')
    result = verify_pipeline(root,True)
    if any(p.is_symlink() for p in root.rglob('*')): raise ValueError('No linked collection inputs')
    before = {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and
        p.suffix not in ('.npz','.pth') and p.relative_to(root).as_posix()!='collect.log'}
    staging.mkdir(parents=True); data = staging/'data'; data.mkdir()
    for n,h in before.items():
        p = root/n
        if p.is_symlink() or p.suffix not in ('.py','.md','.json','.jsonl','.png','.log') or p.stat().st_size>16*1024**2:
            raise ValueError('Unexpected heavy/executable artifact '+n)
        q = data/n; q.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,q)
        if sha(q)!=h: raise ValueError('Copy changed')
    copied = verify_pipeline(data,False)
    atomic(staging/'SUMMARY.json',dict(result,private=True,
        server_heavy_verified_before_collection=True,downloaded_heavy_arrays=False,
        light_copy_independently_verified=True))
    out = render(data,copied)
    with (staging/'index.html').open('x',encoding='utf-8') as f: f.write(out)
    evaluation = rows(data/'evaluation/rows.jsonl')
    ids = sorted({r['example_id'] for r in evaluation})[:4]
    review = [dict(example_id=r['example_id'],regime=r['regime'],arm=r['arm'],control=r['control'],
        source_text=r['source_text'],output=r['received_caption'],objects=None,attributes=None,
        counts=None,relations=None,hallucinations=None,status='PENDING') for r in evaluation
        if r['example_id'] in ids and r['control']=='matched']
    atomic(staging/'FACT_REVIEW.json',dict(private=True,reviewer='',status='PENDING',
        automatic_approval=False,items=review))
    if any(sha(root/n)!=h for n,h in before.items()): raise ValueError('Original light results changed')
    files = {p.relative_to(staging).as_posix():sha(p) for p in staging.rglob('*') if p.is_file()}
    atomic(staging/'COLLECTION.json',dict(files=files,private=True,heavy_arrays_included=False))
    with dest.open('xb') as f,tarfile.open(fileobj=f,mode='w:gz',format=tarfile.USTAR_FORMAT) as tf:
        for p in sorted(staging.rglob('*')):
            if p.is_file():
                info = tarfile.TarInfo('v25_records/'+p.relative_to(staging).as_posix())
                info.mode=0o644; info.size=p.stat().st_size
                with p.open('rb') as src: tf.addfile(info,src)
    with tarfile.open(dest) as tf:
        for m in tf.getmembers():
            if not m.isfile() or hashlib.sha256(tf.extractfile(m).read()).hexdigest()!=sha(staging/Path(m.name).relative_to('v25_records')):
                raise ValueError('PRIVATE archive differs')
    with dest.with_name(dest.name+'.sha256').open('x') as f: f.write(sha(dest)+'  '+dest.name+'\n')
    print('PRIVATE V25 records: '+str(dest),flush=True)
    return dest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('destination');a=p.parse_args()
    source_check();collect(a.root,a.destination)
