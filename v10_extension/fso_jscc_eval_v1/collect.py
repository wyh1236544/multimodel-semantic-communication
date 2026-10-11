"""Collect only independently verified records and visuals; never model/data weights."""
import argparse
import gzip
import io
import json
import sys
import tarfile
from pathlib import Path
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from fso_jscc_eval_v1.report import verify
from fso_jscc_eval_v1.core import digest
from v10.paths import write_path


def collect(source,destination):
    source=Path(source).resolve(strict=True); verify(source,complete=True); destination=write_path(destination)
    if destination.exists() or destination.with_name(destination.name+'.sha256').exists(): raise FileExistsError('Existing archive/checksum preserved')
    done=json.loads((source/'COMPLETE.json').read_text()); names=sorted(set(done['result_sha256'])|{'COMPLETE.json'})
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream,mode='wb',filename='',mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
                for name in names:
                    data=(source/name).read_bytes(); member=tarfile.TarInfo('v10_fso_jscc_eval_v1_records/'+name)
                    member.size,member.mode=len(data),0o644; tar.addfile(member,io.BytesIO(data))
    with tarfile.open(destination,'r:gz') as tar:
        if set(tar.getnames())!={'v10_fso_jscc_eval_v1_records/'+n for n in names} or any(not m.isfile() for m in tar.getmembers()):
            raise ValueError('Wrong archive member set')
        for n in names:
            if tar.extractfile('v10_fso_jscc_eval_v1_records/'+n).read()!=(source/n).read_bytes(): raise ValueError('Collected byte mismatch')
    with destination.with_name(destination.name+'.sha256').open('x') as f: f.write(digest(destination)+'  '+destination.name+'\n')
    print('Verified records/visuals only: '+str(destination))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('source'); p.add_argument('destination'); a=p.parse_args(); collect(a.source,a.destination)
