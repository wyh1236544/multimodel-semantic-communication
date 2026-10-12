"""Verified result-only archive; pretrained/model weights stay on server."""
import argparse
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from text_jscc_v1.report import verify
from text_jscc_v1.core import digest,file_under
from v10.paths import write_path


def collect(source,destination):
    source=Path(source).resolve(strict=True); verify(source)
    status=json.loads((source/'run_identity.json').read_text())['status']
    marker='COMPLETE.json' if status=='complete' else 'DIAGNOSTIC_COMPLETE.json'
    end=json.loads((source/marker).read_text()); names=sorted(set(end['result_sha256'])|set(end['bank_sha256'])|{marker})
    if any(Path(n).suffix=='.pth' for n in names): raise ValueError('Weights cannot enter light records')
    target=write_path(destination)
    if target.exists() or target.with_name(target.name+'.sha256').exists(): raise FileExistsError('Archive/checksum preserved')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',filename='',mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
                for name in names:
                    data=file_under(source,name).read_bytes(); entry=tarfile.TarInfo('v10_text_jscc_v1_records/'+name)
                    entry.size,entry.mode=len(data),0o644; tar.addfile(entry,io.BytesIO(data))
    with tarfile.open(target,'r:gz') as tar:
        if set(tar.getnames())!={'v10_text_jscc_v1_records/'+n for n in names} or any(not m.isfile() for m in tar.getmembers()): raise ValueError('Wrong record archive members')
        for n in names:
            if tar.extractfile('v10_text_jscc_v1_records/'+n).read()!=file_under(source,n).read_bytes(): raise ValueError('Record archive byte mismatch')
    with target.with_name(target.name+'.sha256').open('x') as f: f.write(digest(target)+'  '+target.name+'\n')
    print('Verified {} record archive {}'.format(status,target))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('source'); p.add_argument('destination'); a=p.parse_args(); collect(a.source,a.destination)
