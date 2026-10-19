"""Frozen V20 interventions and bounded receiver-only pilot contracts (CPU)."""
import hashlib
import math
from pathlib import Path
import random
import numpy as np
from docci_text_guided_image_v20.core import (PROJECT, GEOMETRY, read, rows,
    sha, array_sha, safe, atomic, seed, bootstrap)

FOLDER = Path(__file__).resolve().parent
SCHEMA = 'docci-text-aid-v21'
CACHE_SHA = 'befcdefada53ba7be215a8f42af352a4b521de7cba0a7212d446e5ac7a33e5b5'
PARENTS = {
    'A_self': '4946af3906e94d31e0db04cecda965dd1c7c2171692a08e4007aed0768d9e5b4',
    'B_tx_text': '40041f7dbc5cac6dd41d462ce25b1ed61ae70b35946ae62f35b338ff1f4164e2',
    'C_tx_rx_text': '63ff65264bba0014bab59e0217fe2e4e7e322ecfef4532401bcaf65383b76ba5'}
EVAL_SHA = {
    'A_self':'d9499748e70a4943ae19f8157d7fbfd0d0e826fa31fe8b9745f9c54bc6bad210',
    'B_tx_text':'958dc688afb93d7cdc493949ebbbfd34d73ae84b28353d1eff868c35af9698f8',
    'C_tx_rx_text':'144e642b0a52d977efc1eb3d81cf495b4ae034d61bc58fd874eb80211a8359dc'}
PILOT_ARMS = ('self', 'correct', 'donor')
DIAG = {
    'A_self': ('matched',),
    'B_tx_text': ('matched', 'tx_shuffled', 'tx_zero'),
    'C_tx_rx_text': ('matched', 'tx_shuffled', 'tx_zero', 'rx_shuffled',
                      'rx_zero', 'rx_clean', 'rx_clean_shuffled')}
CONTROLS = ('clean', 'clean_shuffled', 'clean_zero', 'noisy', 'noisy_shuffled', 'noisy_zero')

def config(action, profile='nominal', smoke=False):
    if action not in ('diagnose', 'pilot') or profile not in ('nominal', 'missing25'):
        raise ValueError('Unknown fixed experiment')
    if action == 'diagnose' and profile != 'nominal':
        raise ValueError('Diagnose original V20 without extra impairment')
    return dict(schema=SCHEMA, action=action, profile=profile, seed=42,
        diagnostic_train=2 if smoke else 32, diagnostic_dev=2 if smoke else 32,
        train_samples=8 if smoke else 512, dev_samples=2 if smoke else 100,
        epochs=1 if smoke else 8, accumulation=2 if smoke else 4,
        updates=2 if smoke else 1024, learning_rate=0.0001, weight_decay=.01,
        image_snr_db=[None, 0], text_training='clean received V18 memory; privileged diagnostic',
        mask_blocks=4 if profile == 'missing25' else 0, smoke=smoke,
        lpips_weight=.05, latent_weight=.01, save_examples=2,
        fixed_final=True, bootstrap_draws=10000, minimum_relative_LPIPS_gain=.02, geometry=GEOMETRY)

def source_check():
    from docci_text_guided_image_v20_fix1.compat import source_check as old_check
    old_check()
    m = read(FOLDER/'MANIFEST.json')
    for base, files in ((FOLDER, m['files']), (PROJECT, m['dependencies'])):
        for name, digest in files.items():
            if (base/name).is_symlink() or sha(base/name) != digest:
                raise ValueError('Source changed: '+name)
    return m

def choose(index, c):
    entries = index['records']
    groups = {g: [r for r in entries if r['cohort'] == g] for g in ('train', 'dev')}
    if len(groups['train']) != 8192 or len(groups['dev']) != 100:
        raise ValueError('Completed V20 full cache required, not smoke cache')
    for g in groups:
        if any(r['official_split'] != 'train' or r['input_truncated'] for r in groups[g]):
            raise ValueError('Invalid cohort/truncated input')
        groups[g].sort(key=lambda r: seed(42, 'v21-'+g, r['example_id']))
    n = c['diagnostic_train'] if c['action'] == 'diagnose' else c['train_samples']
    v = c['diagnostic_dev'] if c['action'] == 'diagnose' else c['dev_samples']
    out = dict(train=groups['train'][:n], dev=groups['dev'][:v])
    if len({r['example_id'] for g in out.values() for r in g}) != n+v:
        raise ValueError('Duplicate/crossed cohort')
    return out

def order(c):
    step = 0
    for epoch in range(c['epochs']):
        ids = list(range(c['train_samples']))
        random.Random(seed(42, 'v21-order', epoch)).shuffle(ids)
        # Fixed bijective non-identity TRAIN donor, no validation donor in training.
        shift = 1+seed(42, 'v21-donor', epoch) % (len(ids)-1)
        donor = {ids[i]: ids[(i+shift) % len(ids)] for i in range(len(ids))}
        for start in range(0, len(ids), c['accumulation']):
            if step >= c['updates']:
                return
            yield epoch, [(i, donor[i]) for i in ids[start:start+c['accumulation']]]
            step += 1

def impairment(c, eid):
    """Post-AWGN spatial erasure, not a real FSO model; identical across arms."""
    mask = np.ones((1,8,32,32), np.float32)
    blocks = list(range(16))
    random.Random(seed(42, 'v21-mask', eid)).shuffle(blocks)
    for k in blocks[:c['mask_blocks']]:
        y, x = divmod(k,4)
        mask[:,:,y*8:(y+1)*8,x*8:(x+1)*8] = 0
    return mask.reshape(1,8192)

def eval_controls(arm):
    return ('clean',) if arm == 'self' else CONTROLS

def route(arm, control):
    if arm not in DIAG or control not in DIAG[arm]:
        raise ValueError('Unknown diagnostic intervention')
    if arm == 'A_self':
        return 'self', 'self'
    tx = 'donor' if control == 'tx_shuffled' else 'zero' if control == 'tx_zero' else 'source'
    if arm == 'B_tx_text':
        return tx, 'self'
    rx = {'rx_shuffled':'donor', 'rx_zero':'zero', 'rx_clean':'clean',
          'rx_clean_shuffled':'clean_donor'}.get(control, 'received')
    return tx, rx

def expected_rows(c):
    if c['action'] == 'diagnose':
        return {(arm,g,r['example_id'],s,control)
                for arm in PARENTS for g, group in c['selected'].items()
                for r in group for s in c['image_snr_db'] for control in DIAG[arm]}
    return {(arm,'dev',r['example_id'],s,control)
            for arm in PILOT_ARMS for r in c['selected']['dev']
            for s in c['image_snr_db'] for control in eval_controls(arm)}

def key(r):
    return r['arm'], r['cohort'], r['example_id'], r['snr_db'], r['control']

def audit_matrix(c, data):
    seen = {key(r):r for r in data}
    if len(seen) != len(data) or set(seen) != expected_rows(c):
        raise ValueError('Missing, duplicate or unexpected evaluation row')
    for r in data:
        for k in ('psnr', 'ssim', 'LPIPS', 'MSE'):
            if not math.isfinite(r[k]):
                raise ValueError('Nonfinite metric '+k)
        if r['MSE'] <= 0 or not math.isclose(r['psnr'], -10*math.log10(r['MSE']), abs_tol=.0002):
            raise ValueError('PSNR/MSE differs')
        if r['source_text_at_RX'] or r['post_noise_renormalization']:
            raise ValueError('Receiver source leak/post-noise normalization')
        if abs(r['image_stats']['signal_power']-1) > .0002:
            raise ValueError('TX energy differs')
        if c['action'] == 'diagnose':
            ref = seen[(r['arm'],r['cohort'],r['example_id'],r['snr_db'],'matched')]
            if r['control'].startswith('rx_'):
                for name in ('tx_sha256','observed_sha256'):
                    if r[name] != ref[name]:
                        raise ValueError('RX intervention changed own TX/observation')
        else:
            ref = seen[('self',r['cohort'],r['example_id'],r['snr_db'],'clean')]
            if any(r[n] != ref[n] for n in ('tx_sha256','observed_sha256','mask_sha256')):
                raise ValueError('Pilot changed frozen own image TX/channel/mask')
        if r['image_stats']['noise_sha256'] != ref['image_stats']['noise_sha256']:
            raise ValueError('Intervention changed image noise')
    return seen

def paired(a, b, metric):
    if set(a) != set(b):
        raise ValueError('Unpaired samples')
    sign = -1 if metric == 'LPIPS' else 1
    return bootstrap([sign*(a[k][metric]-b[k][metric]) for k in sorted(a)])

def check_anchor(actual, reference):
    for name,old in (('tx_sha256','tx_image_sha256'),('observed_sha256','observed_image_sha256')):
        if actual[name]!=reference[old]:
            raise ValueError('Actual saved-parent anchor payload/observation differs')
    for name,old in (('psnr','psnr'),('ssim','ssim'),('LPIPS','LPIPS'),('MSE','MSE')):
        if not math.isclose(actual[name],reference[old],rel_tol=1e-6,abs_tol=1e-6):
            raise ValueError('Actual saved-parent anchor metric differs: '+name)

def results(c, data):
    audit_matrix(c,data)
    groups = {}
    for r in data:
        label = '{} / {} / {} / {}'.format(r['arm'],r['cohort'],r['snr_db'],r['control'])
        groups.setdefault(label,[]).append(r)
    means = {k:dict(n=len(v),**{m:float(np.mean([r[m] for r in v]))
             for m in ('psnr','ssim','LPIPS')}) for k,v in groups.items()}
    effects = {}
    if c['action'] == 'pilot':
        def select(arm,control,snr):
            return {r['example_id']:r for r in data if r['arm']==arm and r['control']==control and r['snr_db']==snr}
        for s in c['image_snr_db']:
            for name,a,b in (
                ('correct_over_self',select('correct','clean',s),select('self','clean',s)),
                ('correct_over_trained_donor',select('correct','clean',s),select('donor','clean',s)),
                ('correct_over_RX_shuffle',select('correct','clean',s),select('correct','clean_shuffled',s)),
                ('actual_noisy_over_self',select('correct','noisy',s),select('self','clean',s)),
                ('actual_noisy_over_RX_shuffle',select('correct','noisy',s),select('correct','noisy_shuffled',s))):
                effects[str(s)+' / '+name] = {m:paired(a,b,m) for m in ('psnr','LPIPS')}
        # A numerical screen, not semantic success; small exploratory single-seed dev.
        checks = [effects['0 / '+n]['LPIPS']['ci'][0] > 0 for n in
                  ('correct_over_self','correct_over_trained_donor','correct_over_RX_shuffle')]
        checks.append(effects['0 / correct_over_self']['psnr']['ci'][0] >= -.2)
        baseline = np.mean([r['LPIPS'] for r in data if r['arm']=='self' and r['snr_db']==0])
        checks.append(effects['0 / correct_over_self']['LPIPS']['mean'] >= c['minimum_relative_LPIPS_gain']*baseline)
        checks.append(effects['None / correct_over_self']['psnr']['ci'][0] >= -.2)
        screen = all(checks)
    else:
        screen = None
        for arm,controls in DIAG.items():
            for cohort in ('train','dev'):
                for s in c['image_snr_db']:
                    base = {r['example_id']:r for r in data if r['arm']==arm and
                            r['cohort']==cohort and r['snr_db']==s and r['control']=='matched'}
                    for control in controls[1:]:
                        other = {r['example_id']:r for r in data if r['arm']==arm and
                                 r['cohort']==cohort and r['snr_db']==s and r['control']==control}
                        label = '{} / {} / {} / matched_over_{}'.format(arm,cohort,s,control)
                        effects[label] = {m:paired(base,other,m) for m in ('psnr','LPIPS')}
                        effects[label]['changed_TX_fraction'] = float(np.mean([
                            base[k]['tx_sha256']!=other[k]['tx_sha256'] for k in base]))
                        effects[label]['changed_latent_fraction'] = None if 'received_latent_sha256' not in next(iter(base.values())) else float(np.mean([
                            base[k]['received_latent_sha256']!=other[k]['received_latent_sha256'] for k in base]))
    return dict(schema=SCHEMA,means=means,paired_effects=effects,numeric_screen_passed=screen,
        facts='PENDING',semantic_success=False,FSO=False,digital_comparison=False,
        limits='One seed, reused development, one noise draw, pointwise unadjusted image bootstrap; no universal claim')
