"""Hash-locked receipt/report fix, all-conflict preflight and recoverable backups."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

NAMES = ('core.py', 'run.py', 'report.py', 'test_core.py', 'test_report.py', 'MANIFEST.json')
MODULE = 'v10_extension/clip_bart_joint_v6'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe(root, name):
    parts = Path(name).parts
    if Path(name).is_absolute() or not parts or any(p in ('.', '..') for p in parts):
        raise ValueError('Unsafe relative path')
    cursor = root
    for part in parts:
        cursor = cursor/part
        if cursor.is_symlink():
            raise ValueError('Symlink path rejected: '+name)
    return cursor


def install(source, root, dry_run=False):
    source = Path(source).resolve(strict=True)
    root = Path(root).resolve(strict=True)
    patch = json.loads(safe(source, 'PATCH_MANIFEST.json').read_text())
    if patch['schema'] != 'clip-bart-joint-v6-fix2' or set(patch['replacements']) != set(NAMES):
        raise ValueError('Wrong receipt/report patch')
    for name, expected in patch['package_files'].items():
        if sha(safe(source, name).read_bytes()) != expected:
            raise ValueError('Patch payload altered: '+name)
    replacements = {}
    for name in NAMES:
        data = safe(source, 'payload/'+name).read_bytes()
        if sha(data) != patch['replacements'][name]['after']:
            raise ValueError('Replacement hash differs')
        replacements[name] = data
    manifest = json.loads(replacements['MANIFEST.json'])
    if manifest['required_sources'] != patch['required_sources']:
        raise ValueError('Historical prerequisites changed')
    for name, expected in patch['required_sources'].items():
        if sha(safe(root, name).read_bytes()) != expected:
            raise ValueError('Historical prerequisite differs: '+name)
    for name, expected in manifest['files'].items():
        if name in NAMES:
            if sha(replacements[name]) != expected:
                raise ValueError('Replacement does not match module manifest')
        elif sha(safe(root, MODULE+'/'+name).read_bytes()) != expected:
            raise ValueError('Unchanged V6 source differs: '+name)
    current = {name: safe(root, MODULE+'/'+name).read_bytes() for name in NAMES}
    old = all(sha(current[n]) == patch['replacements'][n]['before'] for n in NAMES)
    new = all(sha(current[n]) == patch['replacements'][n]['after'] for n in NAMES)
    if not (old or new):
        raise FileExistsError('Unknown or mixed source version preserved; nothing patched')
    handoff = safe(root, 'v10_extension/docs/HANDOFF.md')
    before_handoff = handoff.read_bytes()
    entry = safe(source, 'SHARED_HANDOFF_ENTRY.md').read_bytes()
    if not entry.endswith(b'\n'):
        raise ValueError('Handoff entry must have final newline')
    if entry.splitlines()[0] in before_handoff.splitlines() and entry not in before_handoff:
        raise FileExistsError('Divergent fix2 handoff preserved')
    append = entry not in before_handoff
    records = safe(root, 'v10_extension/deployment_records')
    if records.exists() and not records.is_dir():
        raise FileExistsError('Deployment records is not a directory')
    if dry_run:
        print('Fix2 preflight passed; exact receipt/run/report/test/manifest targets; no files changed.')
        return
    if new and not append:
        print('Fix2 already installed and verified. No training or result changes.')
        return
    records.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = records/('clip_bart_joint_v6_fix2_'+stamp)
    backup.mkdir(exist_ok=False)
    for name, data in dict(current, **{'HANDOFF.md': before_handoff}).items():
        with (backup/name).open('xb') as stream:
            stream.write(data)
    plan = dict(schema=patch['schema'], before={n: sha(v) for n, v in current.items()},
        after={n: sha(v) for n, v in replacements.items()},
        before_handoff_sha256=sha(before_handoff), old_results_modified=False)
    with (backup/'PATCH_PLAN.json').open('x') as stream:
        json.dump(plan, stream, indent=2)
    if any((backup/n).read_bytes() != v for n, v in current.items()):
        raise ValueError('Backup verification failed before patch')
    if (backup/'HANDOFF.md').read_bytes() != before_handoff:
        raise ValueError('Handoff backup verification failed')
    if any(safe(root, MODULE+'/'+n).read_bytes() != v for n, v in current.items()) or handoff.read_bytes() != before_handoff:
        raise RuntimeError('Source changed during preflight; preserved backup, no patch')
    if old:
        for name, data in replacements.items():
            stage = backup/(name+'.new')
            with stage.open('xb') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            os.replace(str(stage), str(safe(root, MODULE+'/'+name)))
    if append:
        stage = backup/'HANDOFF.new'
        with stage.open('xb') as stream:
            stream.write(before_handoff+b'\n'+entry); stream.flush(); os.fsync(stream.fileno())
        os.replace(str(stage), str(handoff))
    if any(sha(safe(root, MODULE+'/'+n).read_bytes()) != patch['replacements'][n]['after'] for n in NAMES):
        raise ValueError('Post-install report hashes differ; preserve backup')
    if entry not in handoff.read_bytes():
        raise ValueError('Post-install handoff entry absent')
    with (backup/'PATCH_COMPLETE.json').open('x') as stream:
        json.dump(dict(passed=True, after_handoff_sha256=sha(handoff.read_bytes())), stream)
    print('Fix2 installed. Exact original sources/HANDOFF backed up: '+str(backup))
    print('Model/training/channel/weights/data/results unchanged. Use NEW run directory03.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    install(Path(__file__).resolve().parent, args.project_root, args.dry_run)
