"""Add missing approved V15+guard files only. Never writes V14 or any run directory."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def target(root, name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or str(p) != name:
        raise ValueError('Unsafe payload path')
    value = root / name
    for parent in (value,) + tuple(value.parents):
        if parent.is_symlink():
            raise ValueError('Symlink installation forbidden')
    return value


def install(bundle, project):
    bundle, project = Path(bundle).absolute(), Path(project).absolute()
    manifest = json.loads((bundle / 'PAYLOAD.json').read_text())
    files = manifest['files']
    allowed = ('v10_extension/docci_semantic_eval_v15/', 'v10_extension/docci_semantic_eval_v15_guard/')
    planned = []
    for name, expected in files.items():
        if not name.startswith(allowed):
            raise ValueError('Payload would touch a forbidden module/run')
        src, dest = target(bundle / 'payload', name), target(project, name)
        if sha(src) != expected:
            raise ValueError('Bundled bytes changed: ' + name)
        if dest.exists():
            if not dest.is_file() or sha(dest) != expected:
                raise FileExistsError('Unknown existing source preserved: ' + name)
        else:
            planned.append((src, dest, expected))
    for name, expected in manifest['dependencies'].items():
        if sha(target(project, name)) != expected:
            raise ValueError('Existing prerequisite differs: ' + name)
    # ALL conflicts/dependencies are checked before the first write.
    for src, dest, expected in planned:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as f:
            f.write(src.read_bytes())
        if sha(dest) != expected:
            raise ValueError('Installed bytes differ; preserve partial install')
    print('Installed {} missing source files; matching existing files preserved. V14/runs/weights untouched; NO training.'.format(len(planned)), flush=True)
    return len(planned)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', required=True)
    a = p.parse_args()
    install(Path(__file__).resolve().parent, a.project_root)
