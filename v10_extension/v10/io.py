from __future__ import annotations
import json
import importlib.metadata
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from .data import sha256
from . import VERSION
from .paths import write_path

def atomic_json(path, value):
    path = write_path(path)
    temporary = write_path(path.with_name(path.name+'.tmp'))
    with temporary.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(temporary), str(path))

def source_manifest(root):
    root = Path(root)
    selected = []
    for folder in ('v10', 'models', 'utils', 'data', 'losses', 'configs'):
        selected.extend(p for p in (root/folder).rglob('*') if p.suffix in ('.py', '.json', '.yaml'))
    selected.extend(root.glob('*v10*.py'))
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(set(selected))}

def begin_run(path, args, config, root):
    path = write_path(path)
    path.mkdir(parents=True, exist_ok=False)
    import torch
    packages = {}
    for name in ('transformers', 'diffusers', 'numpy', 'Pillow', 'PyYAML'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    record = {'version': VERSION, 'status': 'running', 'pid': os.getpid(),
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.executable, 'python_version': sys.version, 'platform': platform.platform(),
              'torch': str(torch.__version__), 'cuda': torch.version.cuda,
              'packages': packages,
              'cwd': str(Path.cwd()), 'argv': sys.argv, 'args': vars(args), 'config': config,
              'source_sha256': source_manifest(root),
              'split_sha256': sha256(args.split_manifest),
              'checkpoint_sha256': sha256(args.checkpoint)}
    atomic_json(path/'run_identity.json', record)
    return path, record

def atomic_checkpoint(path, payload):
    import torch
    path = write_path(path)
    temp = write_path(path.with_name(path.name+'.tmp'))
    with temp.open('xb') as f:
        torch.save(payload, f)
    os.replace(str(temp), str(path))
