"""Reject the historical batch-one no-op shuffled-Common control."""
import json
import math
from pathlib import Path
from v10.data import sha256
from v10.shuffle import cyclic_donor_map


def verify(directory):
    root = Path(directory)
    identity = json.loads((root/'run_identity.json').read_text(encoding='utf-8'))
    args = identity['args']
    if identity['status'] != 'complete' or args['ablation'] != 'common_only_shuffled':
        raise ValueError('Expected a completed shuffled-Common evaluation')
    if args['split'] != 'validation':
        raise ValueError('The correction must first be validated on the validation split')
    code = Path(__file__).resolve().parent
    for name in ('eval_v10.py', 'models/joint_image_text_model.py', 'v10/model.py', 'v10/shuffle.py'):
        if identity['source_sha256'].get(name) != sha256(code/name):
            raise ValueError('Shuffled evaluation source does not match installed code: '+name)
    manifest = json.loads(Path(args['split_manifest']).read_text(encoding='utf-8'))
    donors = cyclic_donor_map(manifest['validation'][:args['samples']])
    count, changed = 0, set()
    with (root/'samples.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            target = row['image_id']
            if row.get('common_payload_source_image_id') != donors[target]:
                raise ValueError('Shuffled Common donor ID is missing or incorrect')
            delta = float(row.get('common_payload_l2_delta', 0.))
            if not math.isfinite(delta) or delta <= 0.:
                raise ValueError('Shuffled Common payload equals target payload or is nonfinite')
            changed.add(target)
            count += 1
    expected = args['samples']*args['repeats']*len(args['profiles'])*len(args['snr_list'])
    if count != expected or changed != set(donors):
        raise ValueError('Shuffled Common sample coverage is incomplete')
    return {'verified_rows': count, 'images_with_distinct_payloads': len(changed),
            'mapping': 'cyclic_next_image_in_split_order'}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory')
    a = p.parse_args()
    print(json.dumps(verify(a.directory), indent=2))
