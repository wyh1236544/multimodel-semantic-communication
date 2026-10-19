"""One explicit bounded stage. Does not train, install, download, or auto-expand."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_evidence_v23.core import (PROJECT,PROFILES,PROTOCOL,atomic,read,rows,sha,
    safe,source_check,Budget,complete,prepared,seal_tree)

def parser():
    from docci_text_guided_image_v20.run import arguments
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action',choices=('inventory','import-v22','import-v22-score','prepare','digital','caption','score','calibrate','lock-caption'))
    p.add_argument('output');p.add_argument('--profile',choices=tuple(PROFILES),default='smoke')
    p.add_argument('--max-hours',type=float,default=2.);p.add_argument('--prepared');p.add_argument('--target')
    p.add_argument('--from-v22');p.add_argument('--from-v22-score');p.add_argument('--smoke-root');p.add_argument('--smoke-prepared')
    p.add_argument('--v20-root',default=str(PROJECT/'experiment_results/v20_text_guided_image_fix1_01'))
    p.add_argument('--caption-path',default=str(PROJECT/'pretrained/blip2-opt-2.7b-2023'))
    p.add_argument('--asset-lock');arguments(p)
    return p

def inventory(a,root):
    from PIL import features
    names=('torch','transformers','diffusers','Pillow','numpy','sentencepiece')
    versions={}
    for name in names:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    value=dict(schema=PROTOCOL['schema'],training=False,GPU_allocated=False,packages=versions,
        caption_assets_exist=Path(a.caption_path).is_dir(),caption_path=a.caption_path,
        JPEG_available=features.check('jpg'),Pillow_OpenJPEG_available=features.check('jpg_2000'),
        JPEG2000_implemented=False,protocol=PROTOCOL,
        next='Import completed V22 smoke; CPU JPEG first. Missing BLIP-2 is NOT downloaded automatically.')
    atomic(root/'ASSETS.json',value);complete(root,'inventory',dict(no_GPU=True));print(json.dumps(value,ensure_ascii=False,indent=2))

def import_v22(a,root):
    from docci_pretrained_checks_v22.report import verify
    from docci_pretrained_checks_v22.core import prepared as legacy
    if not a.from_v22:raise ValueError('--from-v22 completed smoke required')
    old=Path(a.from_v22);verify(old);index,data,examples=legacy(old)
    if index['profile']!='smoke' or len(examples)!=2 or index['repeats']!=1:
        raise ValueError('Only actual completed V22 two-image smoke can be imported')
    before={p.relative_to(old).as_posix():sha(p) for p in old.rglob('*') if p.is_file()}
    for directory in ('truth','receiver'):
        (root/directory).mkdir()
        for p in sorted((old/directory).iterdir()):
            if not p.is_file() or p.is_symlink() or p.suffix!='.png': raise ValueError('Unexpected original image artifact')
            shutil.copyfile(p,root/directory/p.name)
    shutil.copyfile(old/'rows.jsonl',root/'rows.jsonl')
    if (old/'ANCHOR_A_self.json').is_file():shutil.copyfile(old/'ANCHOR_A_self.json',root/'ANCHOR_A_self.json')
    atomic(root/'index.json',dict(schema='v23-prepared',profile='smoke',spec=PROFILES['smoke'],
        examples=index['examples'],noise_family='v22',independent_test=False,reused_dev=True))
    atomic(root/'IMPORT.json',dict(source_path=str(old.resolve()),source_complete_sha256=sha(old/'COMPLETE.json'),
        exact_original_file_hashes=before,new_inference=False,new_training=False))
    if before!={p.relative_to(old).as_posix():sha(p) for p in old.rglob('*') if p.is_file()}:
        raise ValueError('Readonly original changed')
    complete(root,'prepare',dict(rows=6,imported=True));print('V23 IMPORTED existing six cases; no model/GPU/inference',flush=True)

def lock_caption(a,root):
    from docci_evidence_v23.download_caption import REPO,REVISION,FILES,verify_file
    folder=Path(a.caption_path).resolve(strict=True)
    receipt=read(folder/'DOWNLOAD_RECEIPT.json')
    if receipt['repo']!=REPO or receipt['revision']!=REVISION or receipt['state']!='COMPLETE':
        raise ValueError('Expected verified pinned download receipt')
    if set(receipt['files'])!=set(FILES):raise ValueError('Incomplete official model file matrix')
    for n,spec in FILES.items():
        if verify_file(folder/n,spec)!=receipt['files'][n]:raise ValueError('Downloaded model bytes differ '+n)
    cfg=read(folder/'config.json')
    if cfg.get('model_type')!='blip-2' or cfg.get('image_token_index') is not None:
        raise ValueError('Incompatible modern model snapshot; do not upgrade server environment')
    atomic(root/'ASSET_LOCK.json',dict(kind='caption',tree=seal_tree(folder),overlap_audit='unknown',
        origin=REPO,revision=REVISION,downloads_executed_by_this_stage=False,remote_custom_code=False))
    complete(root,'assets',dict(kind='caption'));print('Pinned caption bytes sealed; pretraining overlap remains unknown',flush=True)

def import_score(a,root):
    from docci_pretrained_checks_v22.report import verify
    if not a.from_v22 or not a.from_v22_score or a.target!=a.prepared:
        raise ValueError('Need old V22 preparation+base score, new prepared and identical --target')
    old=Path(a.from_v22_score);verify(old,a.from_v22,target_root=a.from_v22)
    _,base,_=prepared(a.prepared);rs=rows(old/'rows.jsonl')
    by={(r['example_id'],r['snr_db'],r['repeat']):r for r in base}
    if len(rs)!=len(by) or { (r['example_id'],r['snr_db'],r['repeat']) for r in rs}!=set(by):
        raise ValueError('Old scored and imported cases differ')
    for r in rs:
        if r['control']!='continuous' or r['image_sha256']!=by[r['example_id'],r['snr_db'],r['repeat']]['received_image_sha256']:
            raise ValueError('Old score does not cover the same PNG')
    shutil.copyfile(old/'rows.jsonl',root/'rows.jsonl')
    atomic(root/'identity.json',dict(prepared_sha256=sha(Path(a.prepared)/'COMPLETE.json'),
        target_sha256=sha(Path(a.target)/'COMPLETE.json'),target_path=str(Path(a.target).resolve()),
        old_score_complete_sha256=sha(old/'COMPLETE.json'),old_prepared_complete_sha256=sha(Path(a.from_v22)/'COMPLETE.json'),
        neural_inference_rerun=False,LPIPS='unchanged V22 frozen Alex v0.1 scores, exact same PNGs'))
    complete(root,'score',dict(rows=len(rs),imported=True));print('V23 imported exact original LPIPS; no GPU/re-scoring',flush=True)

def expansion_gate(a):
    from docci_evidence_v23.report import verify
    if a.action not in ('prepare','digital','caption','calibrate'):return
    profile=a.profile if a.action in ('prepare','calibrate') else prepared(a.prepared)[0]['profile']
    if profile=='smoke':return
    if not a.smoke_root:raise ValueError('Actual V23 --smoke-root required before expansion')
    expected='prepare' if a.action=='prepare' else a.action
    if expected in ('digital','caption') and not a.smoke_prepared:
        raise ValueError('Explicit --smoke-prepared required for independent smoke verification')
    result=verify(a.smoke_root,a.smoke_prepared)
    if result['stage']!=expected:raise ValueError('Wrong smoke stage')
    if expected=='prepare' and prepared(a.smoke_root)[0]['profile']!='smoke':raise ValueError('Not a smoke preparation')
    if expected=='caption' and read(Path(a.smoke_root)/'identity.json')['asset_lock_sha256']!=sha(a.asset_lock):
        raise ValueError('Model asset lock differs from actual smoke')

def main():
    a=parser().parse_args();source_check();budget=Budget(a.max_hours)
    if a.config:raise ValueError('Fixed V23 protocol; historical --config unsupported')
    for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ[name]='1'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    root=safe(a.output)
    if root.exists():raise FileExistsError('Existing output preserved')
    if a.action in ('digital','caption','score','import-v22-score') and not a.prepared:raise ValueError('--prepared required')
    if a.action in ('score','import-v22-score') and not a.target:raise ValueError('--target required')
    if a.action=='caption' and not a.asset_lock:raise ValueError('--asset-lock required')
    expansion_gate(a);root.mkdir(parents=True,exist_ok=False)
    atomic(root/'RUN_IDENTITY.json',dict(action=a.action,arguments=vars(a),interpreter=sys.executable,
        python=sys.version,source_manifest_sha256=sha(Path(__file__).parent/'MANIFEST.json'),
        training_updates=0,protocol=PROTOCOL))
    try:
        if a.action=='inventory':inventory(a,root)
        elif a.action=='import-v22':import_v22(a,root)
        elif a.action=='import-v22-score':import_score(a,root)
        elif a.action=='lock-caption':lock_caption(a,root)
        elif a.action=='digital':
            from docci_evidence_v23.digital import run
            run(a,root,budget)
        else:
            from docci_evidence_v23 import neural
            getattr(neural,a.action)(a,root,budget)
        source_check();print('V23 {} FINISHED; independent verification still required'.format(a.action),flush=True)
    except BaseException as error:
        atomic(root/'FAILED.json',dict(error_type=type(error).__name__,message=str(error),partial_preserved=True,no_retry=True));raise

if __name__=='__main__':main()
