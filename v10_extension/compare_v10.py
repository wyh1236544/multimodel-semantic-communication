"""Paired image-cluster bootstrap for two completed, matched evaluations."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from verify_v10_run import verify
from v10.io import atomic_json
from v10.paths import write_path

def load_rows(directory):
    verify(directory)
    rows = [json.loads(line) for line in (Path(directory)/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    return {(r['image_id'], r['profile'], r['nominal_snr_db'], r['repeat']): r for r in rows}

def compare(baseline, candidate, draws=2000, seed=42):
    a, b = load_rows(baseline), load_rows(candidate)
    if a.keys() != b.keys():
        raise ValueError('Paired comparison requires exactly the same image/channel/SNR/repeat keys')
    grouped = defaultdict(lambda: defaultdict(list))
    fields = ('psnr', 'ssim', 'bleu2', 'common_cosine', 'common_caption_bleu2')
    for key, aa in a.items():
        bb = b[key]
        if any(aa[k] != bb[k] for k in ('physical_unit', 'forward_uses', 'reverse_rf_real_uses')):
            raise ValueError('Different physical budgets; do not claim a matched-resource gain')
        if abs(aa['tx_average_power']-bb['tx_average_power']) > 1e-4:
            raise ValueError('Average transmit powers differ; e.g. no_common zero-mask is not an equal-energy baseline')
        grouped[(key[1], key[2])][key[0]].append([bb[f]-aa[f] for f in fields])
    rng, result = np.random.default_rng(seed), {}
    for group, images in grouped.items():
        values = np.array([np.mean(repeats, axis=0) for repeats in images.values()])
        samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
        intervals = np.quantile(samples, [.025, .975], axis=0)
        result[group[0]+'/'+str(group[1])] = {'independent_image_units': len(values), 'delta_candidate_minus_baseline': {
            f: {'mean': float(values[:, i].mean()), 'ci95': intervals[:, i].tolist()} for i,f in enumerate(fields)}}
    return {'bootstrap_draws': draws, 'summary': result,
            'limitation': 'Image-cluster uncertainty for this checkpoint pair; not variation across independent model-training seeds.'}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--draws', type=int, default=2000)
    args = p.parse_args()
    output = write_path(args.output)
    if output.exists() or args.draws < 100:
        raise ValueError('Use a new output and at least100 bootstrap draws')
    result = compare(args.baseline, args.candidate, args.draws)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(json.dumps(result, indent=2))
