"""Image-paired quick-screen inference; rejects incomplete/duplicate comparison lattices."""
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from token_guidance.spec import VARIANTS, DIAGNOSTICS, conditions


def paired(values, seed=1741):
    a = np.asarray(values, float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Empty/nonfinite paired data')
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), (4000, len(a)))].mean(1)
    signs = rng.choice([-1., 1.], (4000, len(a)))
    p = (1+np.count_nonzero(np.abs((signs*a).mean(1)) >= abs(a.mean())))/4001
    return dict(images=len(a), mean_delta=float(a.mean()),
                ci95=list(map(float, np.quantile(means, [.025, .975]))),
                signflip_p=float(p), wins=int(np.count_nonzero(a > 0)))


def analyze(rows, design):
    ids = design['validation_ids']
    cases = conditions(design['smoke'])
    expected = {(v, d, i, c[0], c[1], c[2]) for v in VARIANTS
                for d in (('matched',)+DIAGNOSTICS if v == 'D' else ('matched',))
                for i in ids for c in cases}
    lookup = {}
    for r in rows:
        key = tuple(r[k] for k in ('variant', 'diagnostic', 'image_id', 'channel_kind', 'profile', 'nominal_snr_db'))
        if key in lookup or key not in expected:
            raise ValueError('Duplicate/unplanned result: '+str(key))
        for k in ('psnr', 'ssim', 'token_acc', 'bleu2', 'text_cer', 'text_wer'):
            if not math.isfinite(r[k]):
                raise ValueError('Nonfinite '+k)
        if r['semantic_reals'] != 4608 or r['raw_utf8_wire_bytes'] != 0:
            raise ValueError('Token budget/raw-text branch changed')
        lookup[key] = r
    if set(lookup) != expected:
        raise ValueError('Incomplete matrix: {} of {}'.format(len(lookup), len(expected)))
    # Equal field/noise/resource schedules, but different payloads can have
    # different CRC outcomes. Report both unconditional and both-delivered gains.
    for i, c in itertools.product(ids, cases):
        group = [lookup[(v, 'matched', i)+tuple(c)] for v in VARIANTS]
        for k in ('field_identity', 'noise_seed', 'awg_samples', 'semantic_body_bytes', 'feature_object_wire_bytes'):
            if len(set(r[k] for r in group)) != 1:
                raise ValueError('Unpaired resources/noise: '+k)
    groups = defaultdict(list)
    for r in rows:
        groups['{}/{}/{}/{}/{}'.format(r['variant'], r['diagnostic'], r['channel_kind'], r['profile'], r['nominal_snr_db'])].append(r)
    summary = {}
    for key, group in groups.items():
        entry = {k: float(np.mean([r[k] for r in group])) for k in ('psnr', 'ssim', 'token_acc', 'bleu2', 'text_cer', 'text_wer')}
        entry.update(images=len(group), psnr_p10=float(np.quantile([r['psnr'] for r in group], .1)),
                     delivery_rate=float(np.mean([r['feature_object_crc_accepted'] for r in group])))
        for prefix in ('pre', 'post'):
            bits = sum(r[prefix+'_bits'] for r in group)
            entry[prefix+'_fec_ber'] = sum(r[prefix+'_errors'] for r in group)/bits if bits else None
        entry['total_scheduled_seconds'] = (sum(r['scheduled_seconds'] for r in group)
                                            if group[0]['scheduled_seconds'] is not None else None)
        summary[key] = entry
    effects = {}
    for b, a in [('B', 'A'), ('C', 'A'), ('D', 'A'), ('D', 'B'), ('D', 'C'), ('D', 'OLD')]:
        per_case = {}
        for c in cases:
            pairs = [(lookup[(b, 'matched', i)+tuple(c)], lookup[(a, 'matched', i)+tuple(c)]) for i in ids]
            info = paired([x['psnr']-y['psnr'] for x, y in pairs])
            info['ssim_mean_delta'] = float(np.mean([x['ssim']-y['ssim'] for x, y in pairs]))
            info['token_acc_mean_delta'] = float(np.mean([x['token_acc']-y['token_acc'] for x, y in pairs]))
            both = [x['psnr']-y['psnr'] for x, y in pairs if x['feature_object_crc_accepted'] and y['feature_object_crc_accepted']]
            info['both_delivered'] = paired(both) if both else None
            per_case[str(c)] = info
        effects[b+'-'+a] = per_case
    primary_cases = [c for c in cases if c[0] == 'physical' and c[2] == 20.]
    primary = {}
    contrasts = [('B', 'A'), ('C', 'A'), ('D', 'A'), ('D', 'B'), ('D', 'C'), ('D', 'OLD')]
    for b, a in contrasts:
        # Aggregate channel groups within image, never call them independent images.
        primary[b+'-'+a] = paired([np.mean([lookup[(b, 'matched', i)+tuple(c)]['psnr']-
                                                          lookup[(a, 'matched', i)+tuple(c)]['psnr'] for c in primary_cases]) for i in ids])
    ordered = sorted(primary, key=lambda k: primary[k]['signflip_p'])
    adjusted = 0.
    for j, key in enumerate(ordered):
        adjusted = max(adjusted, min(1., (len(ordered)-j)*primary[key]['signflip_p']))
        primary[key]['holm_p'] = adjusted
    diagnostics = {}
    for d in DIAGNOSTICS:
        delta = []
        both = []
        for i in ids:
            pairs = [(lookup[('D', 'matched', i)+tuple(c)], lookup[('D', d, i)+tuple(c)]) for c in primary_cases]
            delta.append(np.mean([x['psnr']-y['psnr'] for x, y in pairs]))
            accepted = [x['psnr']-y['psnr'] for x, y in pairs if x['feature_object_crc_accepted'] and y['feature_object_crc_accepted'] and y.get('donor_crc_accepted', True)]
            if accepted:
                both.append(np.mean(accepted))
        diagnostics[d] = dict(all=paired(delta), both_delivered=paired(both) if both else None)
    evidence = primary['D-A']
    clean_gain = effects['D-A'][str(cases[0])]['mean_delta']
    candidate = (not design['smoke'] and evidence['mean_delta'] >= .2
                 and evidence['ci95'][0] > 0 and evidence['holm_p'] < .05 and clean_gain > 0)
    return dict(expected_rows=len(expected), rows=len(rows), groups=summary,
                effects_by_condition=effects, primary_physical_20db=primary,
                diagnostic_matched_minus_perturbed=diagnostics,
                direction_status=('promising_expand_validation' if candidate else 'no_clear_positive_evidence_yet'),
                caution='Inspect SSIM, text accuracy, D-OLD and mismatched guidance before choosing direction; screening flag alone is insufficient',
                limitation='one short training seed; image CI does NOT cover training uncertainty; no independent semantic/object evaluator',
                image_selection='first fixed validation IDs; all failures retained')


def write_report(folder, summary):
    folder = Path(folder)
    (folder/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['# Joint token guidance: quick direction screen', '',
             'Status: '+summary['direction_status'], '',
             'A/B/C/D are separately fine-tuned. OLD is the unmodified-weight receiver-guidance checkpoint.',
             'Only C/I/T tokens cross the link. No original UTF8 or untransmitted RX caption.',
             'Primary: physical 20 dB, profiles averaged WITHIN each image; paired image bootstrap + sign-flip, six-contrast Holm.',
             '', '| Contrast | PSNR delta dB | Image CI95 | Holm p |', '|---|---:|---|---:|']
    for key, v in summary['primary_physical_20db'].items():
        lines.append('|{}|{:.4f}|[{:.4f}, {:.4f}]|{:.4g}|'.format(key, v['mean_delta'], *v['ci95'], v['holm_p']))
    lines += ['', 'No clear positive evidence is NOT proof the idea is impossible. Check clean ceiling, delivery rates, training curves, semantic/text degradation and wrong-text diagnostics before expanding.', '', summary['limitation'], '', 'All condition/CRC/BER/text metrics and matched-minus-wrong controls are in summary.json and samples.jsonl.']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
