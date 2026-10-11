"""Independent evaluation contracts, parent identities and image-cluster statistics."""
import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from fso_jscc_v1 import ARMS, VERSION as PARENT_VERSION
from fso_jscc_v1.core import digest, seed_for
from fso_jscc_eval_v1 import VERSION

TEMPLATE = Path(__file__).with_name('config.json')


def load(path=TEMPLATE):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    return validate(c)


def validate(c):
    keys = set(json.loads(TEMPLATE.read_text(encoding='utf-8')))
    if set(c) != keys or c['schema'] != 'fso-jscc-expanded-eval-v1':
        raise ValueError('Wrong or unknown evaluation configuration')
    for k, lo, hi in [('samples', 2, 500), ('proxy_draws', 1, 4), ('physical_draws', 1, 4),
                      ('save_images', 0, 16), ('bootstrap_resamples', 100, 10000)]:
        if type(c[k]) is not int or not lo <= c[k] <= hi:
            raise ValueError('Invalid integer: '+k)
    if c['save_images'] > c['samples']:
        raise ValueError('Example budget exceeds images')
    for k in ('physical_conditions', 'primary_physical_conditions'):
        if not c[k] or len(set(c[k])) != len(c[k]) or not all(isinstance(v, str) for v in c[k]):
            raise ValueError('Unique nonempty condition IDs required')
    if not set(c['primary_physical_conditions']).issubset(c['physical_conditions']):
        raise ValueError('Primary conditions must be evaluated')
    snrs = c['proxy_snrs_db']
    if not snrs or len(set(snrs)) != len(snrs) or any(isinstance(v, bool) or not math.isfinite(v) or not -20 <= v <= 40 for v in snrs):
        raise ValueError('Invalid nominal SNR grid')
    if any(isinstance(c[k], bool) or not isinstance(c[k], (int, float)) or not math.isfinite(c[k])
           for k in ('image_psnr_target_db', 'normalized_wer_target')) or not 5 <= c['image_psnr_target_db'] <= 50 or not 0 <= c['normalized_wer_target'] <= 1:
        raise ValueError('Invalid diagnostic delivery thresholds')
    return c


def settings(c, smoke=False):
    c = copy.deepcopy(c)
    if smoke:
        c.update(samples=2, proxy_snrs_db=[-5, 0, 20], proxy_draws=1,
                 physical_draws=1, save_images=2,
                 physical_conditions=['good_p09', 'moderate_p24', 'stress_p09'],
                 primary_physical_conditions=['moderate_p24'])
    return c


def proxy_conditions(c):
    return [dict(id='clean_snr40', profile='clean', snr=40.)]+[
        dict(id='{}_snr{:g}'.format(p, s), profile=p, snr=float(s))
        for p in ('awgn', 'fso_weak', 'fso_moderate', 'fso_strong') for s in c['proxy_snrs_db']]


def select_ids(manifest, parent, c, smoke=False):
    if len(manifest['validation']) < c['samples']:
        raise ValueError('Not enough images in existing validation split')
    ids = list(manifest['validation'][:c['samples']])
    old = list(parent['validation_ids'])
    if old != list(manifest['validation'][:len(old)]):
        raise ValueError('Parent pilot differs from frozen validation prefix')
    if not smoke and len(ids) <= len(old):
        raise ValueError('Expanded validation must include new images, not just the old pilot')
    if len(ids) != len(set(ids)) or set(ids) & set(parent['train_ids']):
        raise ValueError('Duplicate or training-overlap validation image')
    return ids, [i for i in ids if i in set(old)], [i for i in ids if i not in set(old)]


def matrix(d):
    keys = set()
    for a in ARMS:
        for i in d['validation_ids']:
            for c in d['proxy_conditions']:
                for r in range(1 if c['profile'] == 'clean' else d['config']['proxy_draws']):
                    keys.add((a, i, 'proxy', c['id'], r))
            for c in d['physical_conditions']:
                for r in range(d['config']['physical_draws']):
                    keys.add((a, i, 'waveform_v3', c['id'], r))
    return keys


def expected_counts(c):
    return {'proxy': len(ARMS)*c['samples']*(1+4*len(c['proxy_snrs_db'])*c['proxy_draws']),
            'waveform_v3': len(ARMS)*c['samples']*len(c['physical_conditions'])*c['physical_draws']}


def check_delta_metadata(payload, record, source_design, base_hash, a_hash, split_hash):
    if (payload.get('version') != PARENT_VERSION or payload.get('arm') != record['arm'] or
            payload.get('base_checkpoint_sha256') != base_hash or payload.get('a_delta_sha256') != a_hash or
            payload.get('split_sha256') != split_hash or payload.get('config') != source_design['config'] or
            payload.get('steps') != record['steps'] or payload.get('optimizer_steps') != record['optimizer_steps']):
        raise ValueError('Incompatible/stale/other-arm delta; original weights preserved')


def image_stats(rows, metric):
    groups = defaultdict(list)
    for r in rows:
        groups[r['image_id']].append(float(r[metric]))
    x = np.asarray([np.mean(groups[i]) for i in sorted(groups)])
    if not len(x):
        return dict(mean=None, image_sd=None, images=0)
    return dict(mean=float(x.mean()), image_sd=float(x.std(ddof=1)) if len(x)>1 else 0., images=len(x))


def contrast(rows, metric='psnr', reference=ARMS[2], candidate=ARMS[3], resamples=5000):
    values = {(r['arm'], r['image_id'], r['engine'], r['condition'], r['draw']): r for r in rows}
    per_image = defaultdict(list)
    for key, r in values.items():
        if r['arm'] == candidate:
            old = values.get((reference,)+key[1:])
            if old is None:
                raise ValueError('Unpaired contrast')
            per_image[r['image_id']].append(float(r[metric])-float(old[metric]))
    x = np.asarray([np.mean(per_image[i]) for i in sorted(per_image)])
    if not len(x):
        return dict(mean=None, ci95=None, images=0)
    rng = np.random.default_rng(seed_for('expanded-bootstrap', reference, candidate, metric))
    boot = x[rng.integers(0, len(x), size=(resamples, len(x)))].mean(axis=1)
    return dict(mean=float(x.mean()), ci95=[float(v) for v in np.quantile(boot, [.025, .975])],
                images=len(x), confidence_scope='image sampling only; one trained model; pointwise CI, no multiplicity correction')


def replay_check(rows, originals):
    old = {(r['arm'], r['image_id'], r['engine'], r['condition'], r['draw']): r for r in originals}
    maxima = dict(psnr=0., ssim=0., normalized_wer=0.)
    matched, unavailable_changed, caption_changed = 0, 0, 0
    for r in rows:
        key = (r['arm'], r['image_id'], r['engine'], r['condition'], r['draw'])
        if key not in old:
            continue
        before = old[key]
        if any(r[k] != before[k] for k in ('source_sha256', 'field_seed', 'noise_seed')):
            raise ValueError('Changed old source/channel draw identity')
        matched += 1
        for k in maxima:
            maxima[k] = max(maxima[k], abs(float(r[k])-float(before[k])))
        unavailable_changed += r['receiver_available'] != before['receiver_available']
        caption_changed += r['received_caption'] != before['received_caption']
    tolerances = dict(psnr=.01, ssim=.001, normalized_wer=.01)
    if not matched or unavailable_changed or any(maxima[k] > tolerances[k] for k in maxima):
        raise ValueError('Frozen-weight replay mismatch: '+str(dict(matched=matched, maxima=maxima, unavailable_changed=unavailable_changed)))
    return dict(matched_rows=matched, max_absolute_metric_difference=maxima, tolerances=tolerances,
                changed_captions=caption_changed, changed_receiver_availability=unavailable_changed,
                exact_bitwise_replay_not_assumed=True)


def state_fingerprint(model):
    """Fingerprint all parameters/buffers without retaining another full model."""
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
