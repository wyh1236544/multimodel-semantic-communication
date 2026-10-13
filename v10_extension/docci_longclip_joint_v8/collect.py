"""Verified light records only: no weights, native dataset or pretrained cache."""
import argparse
import gzip
import io
from pathlib import Path
import sys
import tarfile
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from docci_longclip_joint_v8.core import PROJECT,safe_write,digest,read
from docci_longclip_joint_v8.report import finish,verify


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('run'); p.add_argument('archive'); a=p.parse_args()
    root=Path(a.run).resolve(strict=True)
    try: root.relative_to(PROJECT)
    except ValueError: raise ValueError('Only this project records')
    finish(root); result,_=verify(root); target=safe_write(a.archive)
    if target.exists() or target.with_name(target.name+'.sha256').exists(): raise FileExistsError('Existing records archive preserved')
    target.parent.mkdir(parents=True,exist_ok=True)
    files=[]
    for f in sorted(root.rglob('*')):
        if f.is_symlink(): raise ValueError('Symlink records rejected')
        if f.is_file() and f.suffix in ('.json','.jsonl','.log','.md','.html','.png','.svg','.npz','.py','.txt'):
            if '.cache' in f.parts or f.name.endswith('.part'): raise ValueError('Unexpected source/cache in result')
            files.append(f)
    for f in sorted(root.parent.glob('v10_smoke*.log')): files.append(f)
    files=sorted(set(files))
    with target.open('xb') as stream,gzip.GzipFile(fileobj=stream,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for f in files:
            try: relative=str(f.relative_to(root))
            except ValueError: relative='launch_logs/'+f.name
            name='v10_docci_longclip_joint_v8_records/'+relative
            data=f.read_bytes(); member=tarfile.TarInfo(name); member.size=len(data); member.mode=0o644
            tar.addfile(member,io.BytesIO(data))
    with target.with_name(target.name+'.sha256').open('x') as f: f.write(digest(target)+'  '+target.name+'\n')
    print('Collected {} files, {:.1f}MiB, scope={}; no weights/native images/cache'.format(len(files),target.stat().st_size/2**20,result['status']))


if __name__=='__main__': main()
