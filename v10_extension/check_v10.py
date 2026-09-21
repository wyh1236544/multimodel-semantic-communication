"""Run CPU physics + synthetic integration tests. No weights/downloads required."""
import argparse
import ast
import io
import json
import platform
import sys
import time
import unittest
from pathlib import Path
from v10.io import atomic_json
from v10.paths import write_path

def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--output', required=True)
    p.add_argument('--suite', choices=('all', 'deployment'), default='all')
    a = p.parse_args()
    output = write_path(a.output)
    if output.exists():
        raise FileExistsError('Use a fresh verification output filename')
    root = Path(__file__).resolve().parent
    parsed = []
    for path in sorted(root.rglob('*.py')):
        if '__pycache__' not in path.parts:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path), feature_version=8)
            parsed.append(path.relative_to(root).as_posix())
    log = io.StringIO()
    start = time.monotonic()
    pattern = 'test_v10_deployment.py' if a.suite == 'deployment' else 'test*.py'
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.discover(str(root/'tests'), pattern=pattern))
    torch_version = None
    if a.suite == 'all':
        try:
            import torch
            torch_version = getattr(torch, '__version__', None)
        except ImportError:
            pass
    report = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors), 'seconds': time.monotonic()-start,
              'python': sys.version, 'python_executable': sys.executable, 'torch': torch_version, 'suite': a.suite,
              'platform': platform.platform(), 'python38_syntax_files': parsed, 'log': log.getvalue(),
              'pretrained_model_tested': False, 'gpu_training_tested': False,
              'note': ('Deployment-only suite: no neural/physics numerical tests executed.' if a.suite == 'deployment'
                       else 'Synthetic backbones test real integration code; not numerical equivalence to real CLIP/VAE.')}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(log.getvalue())
    print(json.dumps({k:v for k,v in report.items() if k not in ('log', 'python38_syntax_files')}, indent=2))
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    raise SystemExit(main())
