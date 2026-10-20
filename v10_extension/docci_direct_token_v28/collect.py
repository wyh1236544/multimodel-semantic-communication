"""PRIVATE lightweight archive, no cache/weights/frame arrays; no training."""
import argparse
import html
import io
import json
from pathlib import Path
import sys
import tarfile
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_direct_token_v28.core import safe, atomic, sha, read
from docci_direct_token_v28.report import verify, render


def collect(root, destination, full=False, partial=False, max_minutes=60):
    root = safe(root); dest = safe(destination)
    if dest.exists() or dest.with_name(dest.name+'.sha256').exists(): raise FileExistsError('Archive preserved')
    if partial:
        data = dict(status=read(root/'PIPELINE.json')['status'], tasks={}, completed=False,
                    training=False, diagnostic_only=True, full_semantic_success=False)
    else:
        data = verify(root, full, max_minutes)
        if not data['completed']: raise ValueError('Incomplete run cannot be called verified')
        state = read(root/'PIPELINE.json'); state['status'] = data['status']
        state['stages'] = {k: 'VERIFIED' for k in state['stages']}
        atomic(root/'PIPELINE.json', state)
    files = {p.relative_to(root).as_posix(): p for p in root.rglob('*') if p.is_file()
             and p.suffix in ('.json', '.jsonl', '.py', '.md', '.png', '.log', '.txt')
             and '.part-' not in p.name and p.name != 'collection.log'}
    if any(p.is_symlink() for p in files.values()): raise ValueError('Symlink artifact rejected')
    metadata = dict(summary=data, light_files={n: sha(p) for n, p in files.items()},
        PRIVATE=True, excluded=['weights', 'native feature caches', 'TX frame arrays'],
        server_full_verification=full and not partial, diagnostics_only=partial)
    added = {'SUMMARY.json': json.dumps(metadata, ensure_ascii=False, indent=2).encode(),
             'index.html': render(data).encode(),
             'PRIVATE.txt': b'Private DOCCI descriptions/images. Do not publish this archive or its HTML.\n'}
    if set(files) & set(added): raise ValueError('Reserved archive name collision')
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('xb') as out, tarfile.open(fileobj=out, mode='w:gz', format=tarfile.USTAR_FORMAT) as t:
        for name, path in sorted(files.items()):
            info = tarfile.TarInfo(root.name+'/'+name); info.size = path.stat().st_size; info.mode = 0o600
            with path.open('rb') as f: t.addfile(info, f)
        for name, value in added.items():
            info = tarfile.TarInfo(root.name+'/'+name); info.size = len(value); info.mode = 0o600
            t.addfile(info, io.BytesIO(value))
    checksum = dest.with_name(dest.name+'.sha256')
    with checksum.open('x', encoding='utf-8') as f: f.write(sha(dest)+'  '+dest.name+'\n')
    if not partial:
        atomic(root/'VERIFIED_COMPLETE.json', dict(status=data['status'], independent_verified=True,
            weights_and_arrays_rechecked=full, archive_sha256=sha(dest), archive=str(dest),
            performance_success=False, semantic_success=False))
    print('PRIVATE records: '+str(dest), flush=True)
    return metadata


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('destination')
    p.add_argument('--full', action='store_true'); p.add_argument('--partial', action='store_true')
    p.add_argument('--max-minutes', type=float, default=60); a = p.parse_args()
    collect(a.root, a.destination, a.full, a.partial, a.max_minutes)
