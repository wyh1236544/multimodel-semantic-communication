"""Lightweight verified records, no pretrained/model checkpoints or data cache."""
import argparse
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile
EXT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXT))
from bart_coco_modal_v2.core import digest, file_under
from bart_coco_modal_v2.report import verify, MARKERS
from v10.paths import write_path


def collect(source, destination):
    source = Path(source).resolve(strict=True); status = verify(source)
    marker = MARKERS[status]
    end = json.loads((source/marker).read_text()); names = sorted(set(end['result_sha256']) | {marker})
    if any(Path(n).suffix in ('.pth', '.bin', '.npz') for n in names): raise ValueError('Heavy weights/bank forbidden')
    target = write_path(destination)
    if target.exists() or target.with_name(target.name+'.sha256').exists(): raise FileExistsError('Archive/checksum preserved')
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix = 'v10_bart_coco_modal_v2_records/'
    with target.open('xb') as f:
        with gzip.GzipFile(fileobj=f, mode='wb', filename='', mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w', format=tarfile.USTAR_FORMAT) as tar:
                for n in names:
                    data = file_under(source, n).read_bytes(); member = tarfile.TarInfo(prefix+n)
                    member.size, member.mode = len(data), 0o644; tar.addfile(member, io.BytesIO(data))
    with tarfile.open(target, 'r:gz') as tar:
        if set(tar.getnames()) != {prefix+n for n in names} or any(not m.isfile() for m in tar.getmembers()): raise ValueError('Archive members differ')
        for n in names:
            if tar.extractfile(prefix+n).read() != file_under(source, n).read_bytes(): raise ValueError('Archive byte mismatch')
    with target.with_name(target.name+'.sha256').open('x') as f: f.write(digest(target)+'  '+target.name+'\n')
    print('Return '+str(target)+' status='+status)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('source'); p.add_argument('destination'); a = p.parse_args(); collect(a.source, a.destination)

