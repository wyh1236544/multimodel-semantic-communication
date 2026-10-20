"""Reuse native V14 source caches, never frozen channel/TX/receiver caches."""
import hashlib
from pathlib import Path
from docci_direct_token_v28.core import PROJECT, PARENTS, read, sha, atomic


def arguments(p):
    for task in ('image', 'text'):
        p.add_argument('--'+task+'-features', default=str(PROJECT/PARENTS[task]['features']))
        p.add_argument('--'+task+'-checkpoint', default=str(PROJECT/PARENTS[task]['checkpoint']))
    p.add_argument('--bart-path', default=str(PROJECT/'pretrained/bart-base'))
    p.add_argument('--vae-path', default=str(PROJECT/'pretrained/sd-vae-ft-mse'))
    p.add_argument('--lpips-assets', help='Optional existing V20 assets directory with alexnet.pth/linear.pth; no downloads')


def features(args, task):
    return Path(getattr(args, task+'_features')).resolve(strict=True)


def parent(args, task):
    path = Path(getattr(args, task+'_checkpoint')).resolve(strict=True)
    if sha(path) != PARENTS[task]['sha256']:
        raise ValueError('Pinned '+task+' checkpoint differs BEFORE deserialization')
    return path


def select(args, c, task):
    base = features(args, task)
    if sha(base/'cache_train_identity.json') != PARENTS[task]['metadata']:
        raise ValueError('Pinned original TRAIN metadata differs')
    for name, key in [('cache_validation_identity.json', 'validation_metadata'), ('backbone.json', 'backbone_metadata')]:
        if sha(base/name) != PARENTS[task][key]: raise ValueError('Pinned original metadata differs: '+name)
    if sha(base/'source_identity.json') != '387ac2e3effcecc1b0ff86a0f88bdbb23691557e3c9fcb9b4577baf8e994ffdf':
        raise ValueError('Pinned original DOCCI identity differs')
    source = read(base/'source_identity.json')
    if source['annotation_sha256'] != 'c9df4819963883af35ddd2cf257949892fd8c6d88b33a012094352df60719800':
        raise ValueError('Wrong official DOCCI annotation')
    result = {}
    for split, folder, count, original in [('train', 'cache_train', c['train_samples'], 2048),
                                          ('qual_dev', 'cache_validation', c['val_samples'], 100)]:
        values = read(base/(folder+'_identity.json'))['records']
        if len(values) != original or len({r['example_id'] for r in values}) != original:
            raise ValueError('Incomplete/duplicate historical cohort')
        if any(r['official_split'] != split or r['identity']['task'] != task for r in values):
            raise ValueError('Wrong official split/task')
        chosen = []
        for row in values[:count]:
            r = dict(row, cache_file=str(base/folder/(row['example_id']+'.npz')))
            if not r['example_id'].startswith(split+'_'):
                raise ValueError('ID/split mismatch')
            if task == 'text':
                if hashlib.sha256(r['text'].encode()).hexdigest() != r['text_sha256'] or not 2 <= r['source_tokens'] <= 512:
                    raise ValueError('Full text identity/length differs')
            chosen.append(r)
        if len(chosen) != count:
            raise ValueError('Selected cohort incomplete')
        result[split] = chosen
    key = 'text_sha256' if task == 'text' else 'image_sha256'
    if {r[key] for r in result['train']} & {r[key] for r in result['qual_dev']}:
        raise ValueError('Exact TRAIN/qual_dev overlap')
    return result


def audit(args, c, task, root, budget):
    from docci_baseline_v14.data import Dataset
    root.mkdir(parents=True, exist_ok=False)
    chosen = select(args, c, task)
    receipts = {}
    for split, values in chosen.items():
        if task == 'text':
            from docci_text_channel_v18.data import Items
            items = Items(values, 2)
        else:
            items = Dataset(values, 2)
        for i in range(len(items)):
            budget(); item = items[i]
            if task == 'image':
                if item['latent'].shape != (4, 64, 64) or item['rgb'].shape != (512, 512, 3):
                    raise ValueError('Native image feature shape differs')
            if i == 0 or (i+1) % 100 == 0 or i+1 == len(items):
                print('V28 AUDIT {} {} {}/{}'.format(task, split, i+1, len(items)), flush=True)
        atomic(root/(split+'.json'), values)
        receipts[split] = sha(root/(split+'.json'))
    base = features(args, task)
    tracked = {str(base/n): sha(base/n) for n in ('backbone.json', 'source_identity.json',
               'cache_train_identity.json', 'cache_validation_identity.json')}
    pp = parent(args, task); tracked[str(pp)] = sha(pp)
    reference = PROJECT/PARENTS[task]['reference']
    if not reference.is_file():
        raise FileNotFoundError('Required historical replay rows: '+str(reference))
    if sha(reference) != PARENTS[task]['reference_sha256']:
        raise ValueError('Pinned actual historical replay rows differ')
    tracked[str(reference)] = sha(reference)
    atomic(root/'INPUTS.json', dict(task=task, selected=receipts, historical_files=tracked,
        reference=str(reference), reference_sha256=sha(reference),
        backbone=read(base/'backbone.json'), parent_sha256=sha(pp), arrays_verified=True,
        source_features_only=True, qual_dev_reused=True, independent_final_test=False,
        train_encoder_outputs_NOT_cached=True, no_other_modality_loaded=True))


def items(root, split, c):
    from docci_baseline_v14.data import Dataset
    return Dataset(read(Path(root)/(split+'.json')), max(4, c['text_accumulation'], c['image_accumulation']))
