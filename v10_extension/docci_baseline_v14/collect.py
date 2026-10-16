"""Independent CPU collection; original outputs immutable, weights/cache excluded."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_baseline_v14.core import safe, sha, read, atomic
from docci_baseline_v14.report import verify, render, inside


def collect(root, destination, minutes=30):
    root = Path(root).resolve(strict=True)
    out = safe(destination)
    try:
        out.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError('Archive/staging must be outside immutable original run')
    if not 0 < minutes <= 120:
        raise ValueError('Finite collection budget required')
    staging = out.with_name(out.name+'.contents')
    if out.exists() or staging.exists():
        raise FileExistsError('Archive/staging exists; use collect02, never delete old evidence')
    started = time.monotonic()
    def budget():
        if time.monotonic()-started > minutes*60:
            raise TimeoutError('Collection cap exceeded; no training needed')
    data = verify(root, budget)
    staging.mkdir(parents=True, exist_ok=False)
    files = dict(read(root/'LIGHT_MANIFEST.json'))
    files.update({'LIGHT_MANIFEST.json':sha(root/'LIGHT_MANIFEST.json'), 'RUN_FINISHED.json':sha(root/'RUN_FINISHED.json')})
    for name, expected in files.items():
        budget()
        if Path(name).suffix in ('.pth', '.bin') or 'cache_train' in Path(name).parts or 'cache_validation' in Path(name).parts:
            raise ValueError('Raw weights/cache must not enter light archive')
        source = inside(root, name)
        target = staging/'records'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(source) != expected or sha(target) != expected:
            raise ValueError('Original/copy changed during collection')
    atomic(staging/'summary.json', {k:v for k,v in data.items() if k!='evaluations'})
    with (staging/'index.html').open('x', encoding='utf-8') as f:
        f.write(render(data))
    atomic(staging/'COLLECTION.json', dict(original_read_only=True,
           files=len(files), elapsed_seconds=time.monotonic()-started,
           verification='metrics/frames/training schedules/source snapshots',
           raw_descriptions_and_example_images_PRIVATE_not_for_public_GitHub=True))
    with out.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', format=tarfile.USTAR_FORMAT) as tar:
            for p in sorted(staging.rglob('*')):
                budget()
                if p.is_file():
                    tar.add(p, arcname=staging.name+'/'+p.relative_to(staging).as_posix(), recursive=False)
    with out.with_name(out.name+'.sha256').open('x', encoding='utf-8') as f:
        f.write(sha(out)+'  '+out.name+'\n')
    print('Verified PRIVATE records: '+str(out), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('destination')
    p.add_argument('--max-minutes', type=float, default=30)
    a = p.parse_args()
    collect(a.root, a.destination, a.max_minutes)
