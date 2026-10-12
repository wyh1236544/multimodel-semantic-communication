"""Verified lightweight results archive; exact weights remain server."""
import argparse
import gzip
import io
from pathlib import Path
import sys
import tarfile
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from text_recovery_v3.report import verify
from v10.paths import write_path


def collect(source, output):
    root=Path(source).resolve(strict=True); verify(root)
    target=write_path(output)
    if target.exists(): raise FileExistsError('Existing archive preserved')
    with target.open('xb') as f, gzip.GzipFile(fileobj=f,mode='wb',filename='',mtime=0) as gz, tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for file in sorted(root.rglob('*')):
            if not file.is_file() or file.suffix=='.pth': continue
            if file.is_symlink(): raise ValueError('Symlink artifact')
            data=file.read_bytes(); member=tarfile.TarInfo('v10_text_recovery_v3_records/'+str(file.relative_to(root)))
            member.size=len(data); member.mode=0o644; tar.addfile(member,io.BytesIO(data))
    print('Send '+str(target)+'; all negative quality gates retained, weights excluded.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root'); p.add_argument('output'); a=p.parse_args(); collect(a.root,a.output)
