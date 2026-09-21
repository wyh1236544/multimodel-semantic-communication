"""Independent artifact coverage checks, not a model-quality certificate."""
import argparse
import json
from pathlib import Path
from v10.data import sha256

def verify(directory):
    root = Path(directory)
    identity = json.loads((root/'run_identity.json').read_text(encoding='utf-8'))
    complete = json.loads((root/'COMPLETE.json').read_text(encoding='utf-8'))
    if not identity['status'].startswith('complete'):
        raise ValueError('Run identity is not complete')
    if not identity['version'].startswith('V10-'):
        raise ValueError('Unexpected version')
    args = identity['args']
    if sha256(args['split_manifest']) != identity['split_sha256']:
        raise ValueError('Split manifest changed since execution')
    if 'repeats' in args:
        rows = [json.loads(line) for line in (root/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
        expected = args['samples']*args['repeats']*len(args['profiles'])*len(args['snr_list'])
        keys = {(r['image_id'], r['profile'], r['nominal_snr_db'], r['repeat']) for r in rows}
        if len(rows) != expected or len(keys) != expected or complete['rows'] != expected:
            raise ValueError('Missing/duplicate evaluation rows')
        manifest = json.loads(Path(args['split_manifest']).read_text(encoding='utf-8'))
        intended = set(manifest[args['split']][:args['samples']])
        if {r['image_id'] for r in rows} != intended:
            raise ValueError('Evaluated IDs do not match the declared split')
        for image_id in intended:
            for profile in args['profiles']:
                for snr in args['snr_list']:
                    for repeat in range(args['repeats']):
                        if (image_id, profile, snr, repeat) not in keys:
                            raise ValueError('Incomplete image/channel/SNR/repeat grid')
        return {'verified_rows': len(rows), 'images': len(intended), 'quality_claim': 'not_tested_by_verifier'}
    history = json.loads((root/'history.json').read_text(encoding='utf-8'))
    if len(history) != args['epochs'] or [r['epoch'] for r in history] != list(range(1, args['epochs']+1)):
        raise ValueError('Missing/duplicate training epochs')
    if complete['optimizer_steps'] <= 0 or not (root/'best.pth').is_file() or not (root/'last.pth').is_file():
        raise ValueError('No actual optimizer/checkpoint artifacts')
    return {'epochs': len(history), 'optimizer_steps': complete['optimizer_steps'], 'weights_numerically_verified': False}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir), indent=2))
