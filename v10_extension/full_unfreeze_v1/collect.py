"""Collect verified small evidence/images/text, deliberately omit large weights."""
import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT))
from full_unfreeze_v1.report import verify
from full_unfreeze_v1.core import digest
from v10.paths import write_path


def collect(source, destination):
    source = Path(source).resolve(strict=True)
    verify(source, complete=True)
    destination = write_path(destination)
    if destination.exists() or destination.with_name(destination.name+'.sha256').exists():
        raise FileExistsError('Existing archive/checksum preserved')
    complete = json.loads((source/'COMPLETE.json').read_text(encoding='utf-8'))
    names = sorted(set(complete['result_sha256']) | {'COMPLETE.json'})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w', format=tarfile.USTAR_FORMAT) as tar:
                for name in names:
                    data = (source/name).read_bytes()
                    info = tarfile.TarInfo('v10_full_unfreeze_v1_records/'+name)
                    info.size, info.mode = len(data), 0o644
                    tar.addfile(info, io.BytesIO(data))
    with tarfile.open(destination, 'r:gz') as tar:
        members = tar.getmembers()
        if len(members) != len(names) or any(not m.isfile() for m in members):
            raise ValueError('Wrong archive members')
        for name in names:
            if tar.extractfile('v10_full_unfreeze_v1_records/'+name).read() != (source/name).read_bytes():
                raise ValueError('Collected bytes mismatch: '+name)
    with destination.with_name(destination.name+'.sha256').open('x') as stream:
        stream.write(digest(destination)+'  '+destination.name+'\n')
    print('Verified source/result bundle; large checkpoints retained ONLY on server: '+str(destination))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('source'); p.add_argument('destination')
    args = p.parse_args(); collect(args.source, args.destination)
