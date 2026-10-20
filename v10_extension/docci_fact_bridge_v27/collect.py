"""PRIVATE lightweight review collection; no models, images or public upload."""
import argparse
from pathlib import Path
import shutil
import tarfile
from core import read, sha, atomic, new_output, source_check, Budget, write_text
from report import verify


def collect(root, destination, budget):
    root = Path(root).resolve(strict=True); dest = Path(destination).absolute()
    if dest.exists() or dest.with_name(dest.name + '.sha256').exists():
        raise FileExistsError('Existing archive/checksum preserved')
    if root == dest or root in dest.parents:
        raise ValueError('Archive must be outside source run')
    staging = new_output(dest.with_name(dest.name + '.contents'), (root,))
    result = verify(root, budget)
    before = {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*') if p.is_file()}
    for n in before:
        p = root / n
        if p.suffix not in ('.json', '.jsonl', '.html', '.py', '.md') or p.stat().st_size > 64 * 1024 ** 2:
            raise ValueError('Unexpected/large raw artifact')
    shutil.copytree(root, staging / 'data')
    verify(staging / 'data', budget)
    atomic(staging / 'SUMMARY.json', result)
    write_text(staging / 'index.html', '<!doctype html><meta charset="utf-8"><p>PRIVATE V27 records. <a href="data/index.html">打开核验页面</a>。无训练。</p>')
    files = {p.relative_to(staging).as_posix(): sha(p) for p in staging.rglob('*') if p.is_file()}
    atomic(staging / 'COLLECTION.json', dict(files=files, private=True, training_updates=0,
        weights_or_images_included=False, automatic_training_approval=False))
    after = {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*') if p.is_file()}
    if after != before:
        raise ValueError('Source run changed during collection')
    with dest.open('xb') as f, tarfile.open(fileobj=f, mode='w:gz', format=tarfile.USTAR_FORMAT) as t:
        for path in sorted(staging.rglob('*')):
            if path.is_file():
                budget(); name = 'v27_records/' + path.relative_to(staging).as_posix()
                item = tarfile.TarInfo(name); item.mode = 0o644; item.size = path.stat().st_size
                with path.open('rb') as src:
                    t.addfile(item, src)
    with tarfile.open(dest) as t:
        import hashlib
        members = t.getmembers()
        expected = {p.relative_to(staging).as_posix(): sha(p) for p in staging.rglob('*') if p.is_file()}
        if len(members) != len(expected) or {m.name for m in members} != {'v27_records/' + n for n in expected}:
            raise ValueError('Actual collected archive member matrix differs')
        for m in members:
            if not m.isfile() or hashlib.sha256(t.extractfile(m).read()).hexdigest() != expected[m.name[len('v27_records/'):]]:
                raise ValueError('Actual collected archive differs')
    write_text(dest.with_name(dest.name + '.sha256'), sha(dest) + '  ' + dest.name + '\n')
    print('PRIVATE V27 records: ' + str(dest), flush=True)
    return dest


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('destination')
    p.add_argument('--max-minutes', type=float, default=10)
    a = p.parse_args(); source_check(); collect(a.root, a.destination, Budget(a.max_minutes))
