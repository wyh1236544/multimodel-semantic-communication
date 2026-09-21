import argparse
import json
from v10.data import create_manifest

def main():
    p = argparse.ArgumentParser(description='Freeze image-level Localized Narratives splits', allow_abbrev=False)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=20260920)
    p.add_argument('--val-size', type=int, default=500)
    p.add_argument('--calibration-size', type=int, default=300)
    p.add_argument('--test-size', type=int, default=1000)
    p.add_argument('--extra-excluded-json', help='JSON list of any additionally used validation image IDs')
    a = p.parse_args()
    extra = []
    if a.extra_excluded_json:
        with open(a.extra_excluded_json, encoding='utf-8') as f:
            extra = json.load(f)
        if not isinstance(extra, list):
            raise ValueError('extra-excluded-json must contain an ID list')
    d = create_manifest(a.data_root, a.output, a.seed, a.val_size, a.calibration_size, a.test_size, extra)
    print(json.dumps({k: len(d[k]) for k in ('train', 'validation', 'calibration', 'test', 'excluded_history')}))

if __name__ == '__main__':
    main()
