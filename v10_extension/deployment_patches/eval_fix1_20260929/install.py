"""Install the V10 evaluation correction without overwriting unknown server edits."""
import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ALLOWED = {
    'run_v10_formal_suite.sh',
    'v10_extension/eval_v10.py',
    'v10_extension/models/joint_image_text_model.py',
    'v10_extension/v10/model.py',
    'v10_extension/v10/shuffle.py',
    'v10_extension/verify_shuffled_v10.py',
    'v10_extension/tests/test_v10_shuffle.py',
    'v10_extension/tests/test_v10_shuffled_integration.py',
    'v10_extension/docs/FORMAL_EXPERIMENTS.md',
    'v10_extension/docs/HANDOFF.md',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_target(root, name):
    if name not in ALLOWED:
        raise ValueError('Unexpected patch path: '+name)
    target = root
    for part in Path(name).parts:
        target = target/part
        if target.is_symlink():
            raise ValueError('Symlink in patch target: '+str(target))
    return target


def install(root, dry_run=False):
    root = Path(root).resolve()
    stage = Path(__file__).resolve().parent
    expected_stage = root/'v10_extension/deployment_patches/eval_fix1_20260929'
    if stage != expected_stage or not (root/'run_v10_formal_suite.sh').is_file():
        raise ValueError('Extract patch into the existing multimodel project root')
    frozen = root/'v10_extension/manifests/formal_paper_v10_02_noamp_FROZEN.json'
    if frozen.exists():
        raise ValueError('Formal protocol is already frozen; stop and review before patching')
    manifest = json.loads((stage/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('release') != 'v10-eval-fix1-20260929':
        raise ValueError('Unexpected patch release')
    entries = manifest['files']
    if len(entries) != len(ALLOWED) or {entry['path'] for entry in entries} != ALLOWED:
        raise ValueError('Patch manifest must cover exactly the allowed files')
    pending = []
    for entry in entries:
        name = entry['path']
        target = safe_target(root, name)
        source = stage/'payload'/name
        if source.is_symlink() or not source.is_file():
            raise ValueError('Missing or linked patch payload: '+name)
        if source.stat().st_size != entry['size'] or digest(source) != entry['new_sha256']:
            raise ValueError('Patch payload checksum mismatch: '+name)
        current = digest(target) if target.is_file() else None
        if current == entry['new_sha256']:
            continue
        if current != entry['old_sha256']:
            raise ValueError('Server file differs from the expected old version: '+name)
        pending.append((name, target, source, current))
    result = {'release': manifest['release'], 'project_root': str(root),
              'pending_files': [item[0] for item in pending], 'dry_run': dry_run}
    if dry_run or not pending:
        return result
    records = root/'v10_extension/deployment_records'
    if records.is_symlink():
        raise ValueError('Deployment records path is a symlink')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    record = records/('eval_fix1_'+stamp+'_'+str(os.getpid()))
    record.mkdir(parents=True, exist_ok=False)
    backups = record/'backup'
    for name, target, _, current in pending:
        if current is not None:
            backup = backups/name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(target), str(backup))
            if digest(backup) != current:
                raise RuntimeError('Backup verification failed: '+name)
    applied = []
    try:
        for name, target, source, current in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='wb', prefix=target.name+'.fix1.',
                                             dir=str(target.parent), delete=False) as stream:
                temporary = Path(stream.name)
                with source.open('rb') as source_stream:
                    shutil.copyfileobj(source_stream, stream)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.chmod(str(temporary), stat.S_IMODE(target.stat().st_mode) if current is not None else 0o644)
                os.replace(str(temporary), str(target))
            finally:
                if temporary.exists():
                    temporary.unlink()
            applied.append(name)
        for entry in entries:
            if digest(safe_target(root, entry['path'])) != entry['new_sha256']:
                raise RuntimeError('Installed file checksum mismatch: '+entry['path'])
        result.update(status='installed', record=str(record), applied_files=applied)
    except BaseException:
        result.update(status='incomplete', record=str(record), applied_files=applied)
        (record/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        raise
    (record/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    print(json.dumps(install(a.project_root, a.dry_run), indent=2))
