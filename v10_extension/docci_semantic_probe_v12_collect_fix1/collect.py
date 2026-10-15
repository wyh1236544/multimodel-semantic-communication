"""Supply report.json at runtime, without altering certified V12 source bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

FOLDER = Path(__file__).resolve().parent
EXT = FOLDER.parent
sys.path.insert(0, str(EXT))
sys.path.insert(0, str(EXT.parent))
from docci_semantic_probe_v12 import core, report
from docci_semantic_probe_v12 import collect as original


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources():
    manifest = json.loads((FOLDER / 'MANIFEST.json').read_text(encoding='utf-8'))
    if manifest['schema'] != 'docci-semantic-probe-v12-collect-fix1-source':
        raise ValueError('Wrong collector repair manifest')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or (FOLDER / name).is_symlink() or digest(FOLDER / name) != expected:
            raise ValueError('Collector repair source drift: ' + name)
    if digest(core.FOLDER / 'MANIFEST.json') != manifest['original_V12_manifest_sha256']:
        raise ValueError('Original V12 manifest differs; do not overwrite snapshots')
    core.check_sources()  # All original V12 and historical dependencies remain pinned.
    return manifest


def collect(root, destination, max_minutes=30):
    if 'torch' in sys.modules:
        raise RuntimeError('CPU-only collector must not import Torch')
    manifest = check_sources()
    missing = object()
    previous_json = getattr(report, 'json', missing)
    if previous_json is not missing and previous_json is not json:
        raise ValueError('Unexpected pre-existing report.json global')
    previous_render = original.render
    if previous_render is not report.render:
        raise ValueError('Unexpected collector render hook')

    def render_with_receipt(staging, result):
        # Same renderer, data, escaping and original verifier. No quality edits.
        page = report.render(staging, result)
        evidence = Path(staging) / 'collector_fix1_source'
        evidence.mkdir(exist_ok=False)
        for name in sorted(manifest['files']) + ['MANIFEST.json']:
            data = (FOLDER / name).read_bytes()
            with (evidence / name).open('xb') as handle:
                handle.write(data)
        receipt = dict(
            repair='docci_semantic_probe_v12_collect_fix1',
            change='Provide Python stdlib json in original renderer globals for this call only',
            original_V12_manifest_sha256=manifest['original_V12_manifest_sha256'],
            repair_manifest_sha256=digest(FOLDER / 'MANIFEST.json'),
            original_verifier_and_quality_gates_unchanged=True,
            original_sources_results_weights_modified=False,
            no_GPU_no_training_no_new_inference=True)
        with (Path(staging) / 'COLLECTOR_FIX1.json').open('x', encoding='utf-8') as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        return page

    report.json = json
    original.render = render_with_receipt
    try:
        # Retains original verification, output conflicts, bounds, hashes and tar.
        return original.collect(root, destination, max_minutes)
    finally:
        original.render = previous_render
        if previous_json is missing:
            del report.json
        else:
            report.json = previous_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('destination')
    parser.add_argument('--max-minutes', type=float, default=30)
    args = parser.parse_args()
    collect(args.root, args.destination, args.max_minutes)
