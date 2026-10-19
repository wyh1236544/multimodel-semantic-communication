"""PRIVATE light bundle: exact completed stage files, never model assets/weights."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from types import SimpleNamespace
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_pretrained_checks_v22.core import (PROJECT, FOLDER, Budget, safe, sha, atomic, read, source_check, verify_files)
from docci_pretrained_checks_v22.report import render, verify

def collect(a):
    source_check()
    budget = Budget(a.max_hours)
    destination = safe(a.destination)
    checksum = destination.with_name(destination.name+'.sha256')
    if destination.exists() or checksum.exists():
        raise FileExistsError('Existing archive/checksum preserved')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(PROJECT/'experiment_results'), prefix='v22_collect_') as temp:
        staging = Path(temp)/'v22_records'
        staging.mkdir()
        mapping = {}
        total = 0
        for i, original in enumerate([a.prepared]+a.stages):
            budget()
            original = Path(original)
            receipt = verify_files(original)
            if receipt['stage'] not in ('prepare', 'digital', 'caption', 'enhance', 'score'):
                raise ValueError('Only completed experiment stages, not asset directories')
            dest = staging/('prepared' if i==0 else 'stage_%02d'%i)
            for name in list(receipt['files'])+['COMPLETE.json']:
                budget()
                src = original/name
                total += src.stat().st_size
                if total > 4*2**30 or src.suffix in ('.pth', '.pt', '.bin', '.safetensors', '.npz'):
                    raise ValueError('Light-record bound/weight exclusion violated')
                target = dest/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, target)
                if sha(src) != sha(target):
                    raise ValueError('Record copy differs')
            mapping[str(original)] = dest
        render(SimpleNamespace(prepared=mapping[str(Path(a.prepared))],
            stages=[mapping[str(Path(p))] for p in a.stages], output=staging/'display'))
        snapshot = staging/'source'
        snapshot.mkdir()
        m = source_check()
        for name in list(m['files'])+['MANIFEST.json']:
            shutil.copyfile(FOLDER/name, snapshot/name)
        atomic(staging/'PRIVATE.json', dict(raw_DOCCI_and_outputs_private=True,
            weights_and_large_cache_excluded=True, training_updates=0, semantic_success=False))
        manifest = {p.relative_to(staging).as_posix():sha(p) for p in staging.rglob('*') if p.is_file()}
        atomic(staging/'FILE_HASHES.json', manifest)
        with destination.open('xb') as f, tarfile.open(fileobj=f, mode='w:gz', format=tarfile.USTAR_FORMAT) as tf:
            for p in sorted(staging.rglob('*')):
                budget()
                if p.is_file():
                    info = tarfile.TarInfo('v22_records/'+p.relative_to(staging).as_posix())
                    info.size = p.stat().st_size; info.mode = 0o644
                    with p.open('rb') as source:
                        tf.addfile(info, source)
        with tarfile.open(destination, 'r:gz') as tf:
            for member in tf.getmembers():
                budget()
                name = member.name.removeprefix('v22_records/') if sys.version_info >= (3,9) else member.name[len('v22_records/'):]
                expected = sha(staging/name)
                if not member.isfile() or hashlib.sha256(tf.extractfile(member).read()).hexdigest()!=expected:
                    raise ValueError('Archive byte verification failed')
        with checksum.open('x') as f:
            f.write(sha(destination)+'  '+destination.name+'\n')
    print('PRIVATE archive verified: '+str(destination), flush=True)

if __name__=='__main__':
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('destination'); p.add_argument('--prepared', required=True)
    p.add_argument('--stages', nargs='*', default=[]); p.add_argument('--max-hours', type=float, default=.5)
    collect(p.parse_args())
