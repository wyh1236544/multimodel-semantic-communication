"""PRIVATE light records; never package model binaries or modify original results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_caption_calibration_v24.core import source_check, safe, sha, atomic, read
from docci_caption_calibration_v24.report import verify, render, review_template

def collect(root, destination):
    root = Path(root).resolve(strict=True); dest = safe(destination)
    stage = dest.with_name(dest.name+'.contents')
    if dest.exists() or stage.exists() or dest.with_name(dest.name+'.sha256').exists():
        raise FileExistsError('Existing collection preserved; choose a NEW collect name')
    if root == dest or root in dest.parents or root in stage.parents:
        raise ValueError('Collection must be outside inference run')
    result = verify(root); before = read(root/'COMPLETE.json')
    stage.mkdir(parents=True, exist_ok=False); data = stage/'data'
    data.mkdir()
    for p in sorted(root.rglob('*')):
        if p.is_symlink(): raise ValueError('No linked artifacts')
        if not p.is_file(): continue
        if p.suffix not in ('.png', '.json', '.jsonl', '.md', '.py'):
            raise ValueError('Unexpected executable/heavy artifact '+str(p))
        if p.stat().st_size > 8*1024**2:
            raise ValueError('Light record size exceeded; never include arrays/weights')
        q = data/p.relative_to(root); q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)
        if sha(q) != sha(p): raise ValueError('Copy changed')
    verify(data)
    logs=stage/'logs'
    for suffix in ('.contracts.log','.hf_contracts.log','.log'):
        path=root.with_name(root.name+suffix)
        if path.is_file():
            logs.mkdir(exist_ok=True)
            shutil.copyfile(path,logs/path.name)
    atomic(stage/'SUMMARY.json', result)
    if result['stage'] == 'caption': atomic(stage/'CAPTION_REVIEW_TEMPLATE.json', review_template(data))
    with (stage/'index.html').open('x', encoding='utf-8') as f:
        f.write(render(data, result))
    files = {p.relative_to(stage).as_posix(): sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    atomic(stage/'COLLECTION.json', dict(private=True, files=files, training_updates=0, semantic_success=False))
    with dest.open('xb') as f, tarfile.open(fileobj=f, mode='w:gz', format=tarfile.USTAR_FORMAT) as tf:
        for p in sorted(stage.rglob('*')):
            if p.is_file():
                info = tarfile.TarInfo('v24_records/'+p.relative_to(stage).as_posix())
                info.size = p.stat().st_size; info.mode = 0o644
                with p.open('rb') as src: tf.addfile(info, src)
    with tarfile.open(dest, 'r:gz') as tf:
        for m in tf.getmembers():
            path = stage/Path(m.name).relative_to('v24_records')
            if not m.isfile() or hashlib.sha256(tf.extractfile(m).read()).hexdigest() != sha(path):
                raise ValueError('Archive verification failed')
    with dest.with_name(dest.name+'.sha256').open('x') as f:
        f.write(sha(dest)+'  '+dest.name+'\n')
    if read(root/'COMPLETE.json') != before: raise ValueError('Original result changed')
    print('PRIVATE records: '+str(dest), flush=True)
    return dest

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('destination'); a = p.parse_args()
    source_check(); collect(a.root, a.destination)
