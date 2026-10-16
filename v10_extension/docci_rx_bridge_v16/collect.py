"""Separate CPU-only collection; excludes ALL checkpoints and large feature arrays."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_rx_bridge_v16.core import read, sha, safe, atomic
from docci_rx_bridge_v16.report import verify, render, inside


def copy_light(source, target, budget):
    files = dict(read(source/'LIGHT_MANIFEST.json'))
    files.update({n: sha(source/n) for n in ('LIGHT_MANIFEST.json', 'FINISHED.json')})
    for name, expected in files.items():
        budget()
        if Path(name).suffix in ('.pth', '.npz', '.bin', '.pt', '.safetensors'):
            raise ValueError('Raw feature/weight in light manifest')
        origin = inside(source, name)
        dest = target/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, dest)
        if sha(origin) != expected or sha(dest) != expected:
            raise ValueError('Original or copy changed during collection')


def collect(root, destination, max_minutes=90):
    root, out = Path(root).resolve(strict=True), safe(destination)
    if not 0 < max_minutes <= 180:
        raise ValueError('Finite collection budget required')
    if root == out or root in out.parents:
        raise ValueError('Archive/staging must be outside immutable run')
    stage = out.with_name(out.name+'.contents')
    if out.exists() or stage.exists() or out.with_name(out.name+'.sha256').exists():
        raise FileExistsError('Archive/staging preserved; choose collect02, do not delete')
    start = time.monotonic()
    def budget():
        if time.monotonic()-start > max_minutes*60:
            raise TimeoutError('CPU collection budget exhausted; no retraining needed')
    result = verify(root, max_minutes=max_minutes)
    stage.mkdir(parents=True, exist_ok=False)
    copy_light(root, stage/'records', budget)
    identity = read(root/'identity.json')
    if identity['action'] != 'cache':
        copy_light(Path(identity['cache']), stage/'records/cache_reference', budget)
    atomic(stage/'summary.json', result)
    # Derived local HTML only; raw text/image examples NEVER public GitHub.
    with (stage/'records/index.html').open('x', encoding='utf-8') as f:
        f.write(render(stage/'records', result))
    atomic(stage/'COLLECTION.json', dict(original_run_read_only=True, PRIVATE=True,
           array_and_checkpoint_hashes_verified_on_server=True,
           feature_arrays_and_weights_excluded=True,
           neural_inference_replayed_by_collector=False,
           elapsed_seconds=time.monotonic()-start))
    hashes = {p.relative_to(stage).as_posix(): sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    atomic(stage/'ARCHIVE_MANIFEST.json', hashes)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', format=tarfile.USTAR_FORMAT) as tar:
            for name in sorted(hashes)+['ARCHIVE_MANIFEST.json']:
                budget()
                data = (stage/name).read_bytes()
                info = tarfile.TarInfo('v16_records/'+name)
                info.mode, info.mtime, info.size = 0o644, 0, len(data)
                import io
                tar.addfile(info, io.BytesIO(data))
    with out.with_name(out.name+'.sha256').open('x', encoding='utf-8') as f:
        f.write(sha(out)+'  '+out.name+'\n')
    print('PRIVATE records collected: '+str(out), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('destination')
    p.add_argument('--max-minutes', type=float, default=90)
    a = p.parse_args()
    collect(a.root, a.destination, a.max_minutes)
