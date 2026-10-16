"""Offline clean baselines only. No Common/FSO/diffusion/test access or promotion."""
import argparse
import gc
import os
from pathlib import Path
import shutil
import sys
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_baseline_v14.core import (PROJECT, FOLDER, SCHEMA, load, read, sha, safe,
                                    atomic, source_check, parameter_hash, ARMS)


def approved(kind, override=None):
    specs = read(FOLDER/'parents.json') if kind=='text' else read(EXT/'docci_semantic_probe_v12/parents.json')
    spec = specs[kind]
    root = Path(override or PROJECT/spec['default_run']).resolve(strict=True)
    for name, expected in spec['files'].items():
        if sha(root/name) != expected:
            raise ValueError('Approved historical record differs: '+name)
    path = root/spec['checkpoint']
    if sha(path) != spec['checkpoint_sha256']:
        raise ValueError('Approved historical checkpoint differs; no fallback')
    return dict(path=path, spec=spec, config=read(root/'config.json'))


def subset(dataset, count):
    from docci_baseline_v14.data import Dataset
    return Dataset(dataset.entries[:count], dataset.limit)


def run_task(a, c, task, folder, device, budget):
    import torch
    from docci_modal_v9.backbones import load as load_backbone, signature, historical_backbone
    from docci_baseline_v14.data import view, cache
    from docci_baseline_v14.model import TextModel
    from docci_baseline_v14.train import fit
    from docci_baseline_v14.evaluate import evaluate
    from docci_baseline_v14.perceptual import load as load_lpips
    from docci_reconstruction_v10.model import ImageBranch
    folder.mkdir(exist_ok=False)
    backbone, tok, br = load_backbone(task, a, device)
    historical_backbone(task, br['files'])
    before = signature(backbone)
    train_records, val_records, source = view(a, c, task)
    atomic(folder/'backbone.json', br)
    atomic(folder/'source_identity.json', source)
    val = cache(a, c, val_records, task, backbone, tok, br, folder/'cache_validation', device, budget)
    train = None
    if a.mode != 'calibrate':
        train = cache(a, c, train_records, task, backbone, tok, br, folder/'cache_train', device, budget)
    metric, metric_record = load_lpips(a.lpips_assets, device) if task=='image' else (None, dict(status='NOT_APPLICABLE'))
    atomic(folder/'perceptual.json', metric_record)
    native_folder = folder/'native_validation'
    evaluate(None, val, task, backbone, tok, device, native_folder, budget,
             'validation', c['save_images'], native=True, perceptual=metric)
    outputs = [native_folder.relative_to(folder.parent).as_posix()]
    arms = a.text_arms.split(',') if task=='text' else ['data64', 'expanded']
    info = None
    if task == 'image' or a.mode == 'calibrate':
        info = approved(task, getattr(a, task+'_parent'))
    if a.mode == 'calibrate':
        arms = ['historical_reference']
    for arm in arms:
        budget()
        torch.manual_seed(c['seed'])
        if task == 'text':
            model = TextModel(backbone, 'legacy64' if a.mode=='calibrate' else arm).to(device)
            if info is not None:
                record = torch.load(str(info['path']), map_location='cpu')
                if record.get('schema') != 'v12-checkpoint' or record.get('task') != 'text' or record.get('arm') != 'B1_ce_kd' or record.get('updates') != 4096:
                    raise ValueError('Expected pinned V12 CE+KD production text checkpoint')
                model.restore(record['state'])
                del record
        else:
            model = ImageBranch(dict(image_size=512, image_width=8), 'I2_shifted').to(device)
            record = torch.load(str(info['path']), map_location='cpu')
            if record.get('version') != 'docci-reconstruction-v10' or record.get('arm') != 'I2_shifted' or record.get('updates') != 512:
                raise ValueError('Expected pinned I2 checkpoint')
            model.restore(record['state'])
            del record
        initial = parameter_hash(model)
        arm_folder = folder/arm
        arm_folder.mkdir()
        if a.mode != 'calibrate':
            dataset = subset(train, min(64, len(train))) if task=='image' and arm=='data64' else train
            fit(model, dataset, task, c, arm_folder/'train', device, budget, backbone,
                virtual_count=len(train), smoke=a.mode=='smoke')
            train_folder = arm_folder/'diagnostic_train'
            evaluate(model, subset(train, min(c['train_eval_samples'], len(train))), task, backbone, tok,
                     device, train_folder, budget, 'diagnostic_train', 0, perceptual=metric)
            outputs.append(train_folder.relative_to(folder.parent).as_posix())
        val_folder = arm_folder/'validation'
        evaluate(model, val, task, backbone, tok, device, val_folder, budget,
                 'validation', c['save_images'], perceptual=metric)
        outputs.append(val_folder.relative_to(folder.parent).as_posix())
        atomic(arm_folder/'identity.json', dict(arm=arm, model_arm=model.arm, task=task,
            initial_hashes=initial, final_hashes=parameter_hash(model),
            active_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            image_parent_sha256=info['spec']['checkpoint_sha256'] if info else None,
            capacity_not_parameter_matched=arm=='wide768', source_oracle_at_receiver=False,
            full_official_training_replication=False))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if signature(backbone) != before:
        raise RuntimeError('Frozen backbone changed')
    if any(p.grad is not None for p in backbone.parameters()):
        raise RuntimeError('Frozen backbone received parameter gradients')
    if info is not None and sha(info['path']) != info['spec']['checkpoint_sha256']:
        raise ValueError('Historical checkpoint changed during run')
    if sha(Path(a.data_root)/'docci_descriptions.jsonlines') != source['annotation_sha256']:
        raise ValueError('DOCCI source changed during run')
    atomic(folder/'identity.json', dict(task=task, arms=arms, outputs=outputs,
           backbone_signature_before=before, backbone_signature_after=signature(backbone),
           parent_used=None if info is None else dict(checkpoint_sha256=info['spec']['checkpoint_sha256'],
           historical_TRAIN_samples=64 if task=='image' else 2048),
           no_validation_best_selection=True, common=False, FSO=False, diffusion=False))
    return outputs


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--mode', choices=('smoke', 'train', 'calibrate'), required=True)
    p.add_argument('--task', choices=('text', 'image', 'both'), required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config')
    p.add_argument('--profile', choices=('pilot', 'full'), default='pilot')
    p.add_argument('--text-arms', default='legacy64,released64')
    p.add_argument('--text-parent')
    p.add_argument('--image-parent')
    p.add_argument('--lpips-assets')
    p.add_argument('--max-hours', type=float, default=8)
    p.add_argument('--data-root', default=str(PROJECT))
    p.add_argument('--bart-path', default=str(PROJECT/'pretrained/bart-base'))
    p.add_argument('--vae-path', default=str(PROJECT/'pretrained/sd-vae-ft-mse'))
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or not 0 < a.max_hours <= 48:
        raise ValueError('Physical GPU1 and finite0..48h budget required')
    arms = a.text_arms.split(',')
    if len(set(arms)) != len(arms) or any(x not in ARMS for x in arms):
        raise ValueError('Unknown/duplicate text arms')
    c = load(a.config, 'smoke' if a.mode=='smoke' else a.profile)
    manifest = source_check()
    root = safe(a.output)
    if root.exists():
        raise FileExistsError('Existing output preserved; choose NEW attempt')
    if shutil.disk_usage(PROJECT).free < c['min_disk_gib']*2**30:
        raise RuntimeError('Insufficient project disk; no deletion')
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    atomic(root/'config.json', c)
    atomic(root/'design.json', dict(schema=SCHEMA, mode=a.mode, task=a.task, text_arms=arms,
           profile=a.profile, baseline_experiment_NOT_quality_success=True,
           new_training_initialization='fresh text; pinned I2 weights for both image data arms',
           official_code_vs_paper_discrepancy='released default dim_latent64; Table13 says768',
           no_automatic_FSO_or_joint_promotion=True))
    snapshot = root/'source_snapshot'
    snapshot.mkdir()
    for name in list(manifest['files'])+['MANIFEST.json']:
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FOLDER/name, target)
    for name in manifest['dependencies']:
        target = root/'dependency_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/name, target)
    try:
        import torch
        import numpy as np
        if not torch.cuda.is_available():
            raise RuntimeError('Authentic CUDA smoke required; do not substitute CPU')
        torch.set_num_threads(4)
        device = torch.device('cuda:0')
        free, total = torch.cuda.mem_get_info(device)
        print('GPU1 free={:.2f}GiB total={:.2f}GiB'.format(free/2**30, total/2**30), flush=True)
        if free < c['min_free_gib']*2**30:
            raise RuntimeError('Startup free GPU budget not met; no unrelated process actions')
        def budget():
            if time.monotonic()-started > a.max_hours*3600:
                raise TimeoutError('Owned experiment time budget exhausted')
            if torch.cuda.mem_get_info(device)[0] < c['reserve_free_gib']*2**30:
                raise RuntimeError('Free GPU reserve crossed; no automatic batch change')
            if shutil.disk_usage(PROJECT).free < 4*2**30:
                raise RuntimeError('Free project disk reserve crossed; no deletion')
        outputs = []
        tasks = ['text', 'image'] if a.task=='both' else [a.task]
        for task in tasks:
            torch.manual_seed(c['seed'])
            np.random.seed(c['seed'])
            atomic(root/'progress.json', dict(stage='loading_'+task, elapsed_seconds=time.monotonic()-started))
            outputs.extend(run_task(a, c, task, root/task, device, budget))
            gc.collect()
            torch.cuda.empty_cache()
        source_check()
        atomic(root/'run_identity.json', dict(schema=SCHEMA, mode=a.mode, outputs=outputs,
               task=a.task, gpu_physical=1, python=sys.version, torch=torch.__version__,
               max_hours=a.max_hours, wall_seconds=time.monotonic()-started,
               successful_recovery_claim=False, source_manifest_sha256=sha(FOLDER/'MANIFEST.json')))
        # Seal LIGHT artifacts only. Cache/weights stay private on the server.
        files = {}
        for file in sorted(root.rglob('*')):
            if file.is_file() and 'cache_train' not in file.parts and 'cache_validation' not in file.parts and file.suffix != '.pth':
                files[file.relative_to(root).as_posix()] = sha(file)
        atomic(root/'LIGHT_MANIFEST.json', files)
        atomic(root/'RUN_FINISHED.json', dict(manifest_sha256=sha(root/'LIGHT_MANIFEST.json'),
               independent_verification_pending=True, no_quality_promotion=True))
        print('TRAIN_AND_EVAL_FINISHED; independent CPU verification/collection still required', flush=True)
    except BaseException as exc:
        atomic(root/'FAILED.json', dict(exception=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__':
    main()
