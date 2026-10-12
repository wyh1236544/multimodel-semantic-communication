"""Verified complete OR diagnostic-stop records; weights remain on server."""
import argparse
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from residual_fusion_v2.report import verify
from residual_fusion_v2.core import digest
from fso_jscc_v1.report import safe_file
from v10.paths import write_path


def collect(source,destination):
    source=Path(source).resolve(strict=True); verify(source)
    status=json.loads((source/'run_identity.json').read_text())['status']
    marker='COMPLETE.json' if status=='complete' else 'DIAGNOSTIC_COMPLETE.json'
    end=json.loads((source/marker).read_text()); names=sorted(set(end['result_sha256'])|{marker})
    if any(Path(n).suffix=='.pth' for n in names): raise ValueError('Weights cannot enter light archive')
    target=write_path(destination)
    if target.exists() or target.with_name(target.name+'.sha256').exists(): raise FileExistsError('Archive/checksum preserved')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',filename='',mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
                for n in names:
                    data=safe_file(source,n).read_bytes(); info=tarfile.TarInfo('v10_residual_fusion_v2_records/'+n)
                    info.size,info.mode=len(data),0o644; tar.addfile(info,io.BytesIO(data))
    with tarfile.open(target,'r:gz') as tar:
        if set(tar.getnames())!={'v10_residual_fusion_v2_records/'+n for n in names} or any(not m.isfile() for m in tar.getmembers()): raise ValueError('Wrong collection members')
        for n in names:
            if tar.extractfile('v10_residual_fusion_v2_records/'+n).read()!=safe_file(source,n).read_bytes(): raise ValueError('Collection bytes differ')
    with target.with_name(target.name+'.sha256').open('x') as f: f.write(digest(target)+'  '+target.name+'\n')
    print('Verified {} records archive: {}'.format(status,target))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('source'); p.add_argument('destination'); a=p.parse_args(); collect(a.source,a.destination)
