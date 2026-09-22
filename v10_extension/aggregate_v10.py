"""Hierarchical paired bootstrap across independently trained model seeds."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from compare_v10 import load_rows
from v10.io import atomic_json
from v10.paths import write_path

FIELDS = ('psnr', 'ssim', 'bleu2', 'common_cosine', 'common_caption_bleu2')

def paired_deltas(baseline, candidate):
    a, b = load_rows(baseline), load_rows(candidate)
    if a.keys() != b.keys():
        raise ValueError('Every seed pair must have identical image/channel/SNR/repeat keys')
    grouped = defaultdict(lambda: defaultdict(list))
    for key, left in a.items():
        right = b[key]
        if any(left[k] != right[k] for k in ('physical_unit', 'forward_uses', 'reverse_rf_real_uses')):
            raise ValueError('Physical budgets differ in a seed pair')
        grouped[(key[1], key[2])][key[0]].append([right[f]-left[f] for f in FIELDS])
    return {group: np.asarray([np.mean(v, axis=0) for v in images.values()])
            for group, images in grouped.items()}

def aggregate(baselines, candidates, draws, seed):
    if len(baselines) != len(candidates) or len(baselines) < 3:
        raise ValueError('Provide matched baseline/candidate lists with at least three training seeds')
    pairs = [paired_deltas(a, b) for a, b in zip(baselines, candidates)]
    if any(p.keys() != pairs[0].keys() for p in pairs[1:]):
        raise ValueError('Seed pairs have different evaluation groups')
    rng, output = np.random.default_rng(seed), {}
    for group in pairs[0]:
        arrays = [p[group] for p in pairs]
        if len({a.shape for a in arrays}) != 1:
            raise ValueError('Seed pairs have different independent image counts')
        observed = np.stack([a.mean(0) for a in arrays])
        boot = []
        for _ in range(draws):
            seed_ids = rng.integers(0, len(arrays), size=len(arrays))
            sampled = []
            for seed_id in seed_ids:
                values = arrays[seed_id]
                sampled.append(values[rng.integers(0, len(values), size=len(values))].mean(0))
            boot.append(np.mean(sampled, axis=0))
        boot = np.asarray(boot)
        output[group[0]+'/'+str(group[1])] = {
            'training_seeds': len(arrays), 'images_per_seed': arrays[0].shape[0],
            'delta_candidate_minus_baseline': {
                field: {'mean': float(observed[:, i].mean()),
                        'ci95': np.quantile(boot[:, i], [.025, .975]).tolist(),
                        'positive_seed_fraction': float(np.mean(observed[:, i] > 0))}
                for i, field in enumerate(FIELDS)}}
    return {'method': 'hierarchical paired bootstrap over training seeds and image IDs',
            'draws': draws, 'summary': output}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--baseline', nargs='+', required=True)
    p.add_argument('--candidate', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--draws', type=int, default=5000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    target = write_path(args.output)
    if target.exists() or args.draws < 1000:
        raise ValueError('Use a new output and at least 1000 bootstrap draws')
    result = aggregate(args.baseline, args.candidate, args.draws, args.seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, result)
    print(json.dumps(result, indent=2))
