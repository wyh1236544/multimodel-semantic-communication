"""Small verified image/text/evidence archive; omit all large checkpoints."""
import argparse
import gzip
import io
import json
import sys
import tarfile
from pathlib import Path
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from joint_jscc_arch_v1.core import digest
from joint_jscc_arch_v1.report import verify
from fso_jscc_v1.report import safe_file
from v10.paths import write_path


def collect(source,destination):
    source=Path(source).resolve(strict=True); verify(source,complete=True)
    destination=write_path(destination)
    if destination.exists() or destination.with_name(destination.name+'.sha256').exists(): raise FileExistsError('Existing archive/checksum preserved')
    ledger=json.loads((source/'COMPLETE.json').read_text())
    names=sorted(set(ledger['result_sha256'])|{'COMPLETE.json'})
    if any(Path(n).suffix=='.pth' for n in names): raise ValueError('Large weights cannot enter small records archive')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream,mode='wb',filename='',mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
                for name in names:
                    data=safe_file(source,name).read_bytes(); entry=tarfile.TarInfo('v10_joint_jscc_arch_v1_records/'+name)
                    entry.size,entry.mode=len(data),0o644; tar.addfile(entry,io.BytesIO(data))
    with tarfile.open(destination,'r:gz') as tar:
        if set(tar.getnames())!={'v10_joint_jscc_arch_v1_records/'+n for n in names} or any(not m.isfile() for m in tar.getmembers()): raise ValueError('Wrong collection members')
        for name in names:
            if tar.extractfile('v10_joint_jscc_arch_v1_records/'+name).read()!=safe_file(source,name).read_bytes(): raise ValueError('Collected bytes differ')
    destination.with_name(destination.name+'.sha256').write_text(digest(destination)+'  '+destination.name+'\n')
    print('Verified records/images/text bundle: '+str(destination))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('source'); p.add_argument('destination'); a=p.parse_args(); collect(a.source,a.destination)

