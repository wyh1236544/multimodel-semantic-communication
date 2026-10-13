"""New DOCCI TX/Common/RX training; frozen authentic BART/VAE/Long-CLIP."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent
sys.path.insert(0,str(PROJECT)); sys.path.insert(0,str(EXT))
from docci_longclip_joint_v8.core import load,schedule,records,safe_write,atomic_json,digest,read


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--mode',choices=('smoke','diagnostic','screen','pipeline'),required=True)
    p.add_argument('--output-dir',required=True); p.add_argument('--experiment-config',default=str(Path(__file__).with_name('config.json')))
    p.add_argument('--data-root',default=str(PROJECT/'DOCCI'))
    p.add_argument('--bart-path',default=str(PROJECT/'pretrained/bart-base'))
    p.add_argument('--vae-path',default=str(PROJECT/'pretrained/sd-vae-ft-mse'))
    p.add_argument('--clip-tokenizer-path',default=str(PROJECT/'pretrained/clip-vit-base-patch32'))
    p.add_argument('--longclip-path',default=str(PROJECT/'pretrained/longclip-B.pt'))
    p.add_argument('--text-init',help='Explicit optional trusted own completed P0_ce width64 private-codec checkpoint')
    p.add_argument('--image-size',type=int,choices=(256,512),default=512)
    p.add_argument('--max-text-tokens',type=int,choices=(256,512,1024),default=512)
    p.add_argument('--text-updates',type=int); p.add_argument('--max-hours',type=float,default=4.)
    return p


def main():
    args=parser().parse_args(); c=load(args.experiment_config,args.image_size,args.max_text_tokens)
    if args.text_updates is not None:
        if not 8<=args.text_updates<=10000: raise ValueError('Bound text updates8..10000')
        c['text_updates']=args.text_updates
    design_schedule=schedule(c,args.mode)
    # A diagnostic can explicitly extend text-only learning without changing other steps.
    if args.text_updates is not None and args.mode=='diagnostic': design_schedule['text_steps']=args.text_updates
    if not math.isfinite(args.max_hours) or not 0<args.max_hours<=24: raise ValueError('Finite positive run cap<=24h')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise ValueError('Physical GPU1 required; no unrelated GPU/process mutation')
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'
    import numpy as np
    import torch
    from docci_longclip_joint_v8.backbones import load_models,signature,verify_frozen
    from docci_longclip_joint_v8.data import cache_view,tensors
    from docci_longclip_joint_v8.model import JointModel,warm_text
    from docci_longclip_joint_v8.workflow import fit,text_controls,text_gate
    from docci_longclip_joint_v8.evaluate import evaluate,native_diagnostics
    from docci_longclip_joint_v8.report import finish
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('One visible real GPU required')
    device=torch.device('cuda:0'); free,total=torch.cuda.mem_get_info(device)
    print('GPU free={:.2f}GiB total={:.2f}GiB'.format(free/2**30,total/2**30),flush=True)
    if free/2**30<c['min_free_gib']: raise MemoryError('Insufficient startup free memory; no processes stopped')
    if shutil.disk_usage(PROJECT).free<12*2**30: raise OSError('Need12GiB free disk inside project')
    root=safe_write(args.output_dir)
    if root.exists(): raise FileExistsError('Use NEW output; existing result preserved: '+str(root))
    root.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    folder=Path(__file__).parent; files=sorted(f for f in folder.iterdir() if f.is_file() and f.suffix in ('.py','.json','.md','.txt'))
    (root/'source_snapshot').mkdir()
    for file in files:
        with (root/'source_snapshot'/file.name).open('xb') as f: f.write(file.read_bytes())
    identity=dict(version='docci-longclip-joint-v8',mode=args.mode,config=c,schedule=design_schedule,
        status='running',pid=os.getpid(),python=sys.executable,torch=str(torch.__version__),cuda=torch.version.cuda,
        source_sha256={f.name:digest(f) for f in files},argv=sys.argv,gpu_physical=1,amp=False,optimizer_resumed=False)
    atomic_json(root/'run_identity.json',identity)
    def budget():
        if time.monotonic()-started>args.max_hours*3600: raise TimeoutError('Run cap reached; preserve partial logs')
        free,_=torch.cuda.mem_get_info(device)
        if free/2**30<c['reserve_free_gib']: raise MemoryError('Free GPU reserve breached; other tasks untouched')
    def unchanged(models,signatures,receipt,views):
        verify_frozen(models,signatures)
        if digest(Path(args.data_root)/'docci_descriptions.jsonlines')!=receipt['annotation_sha256']: raise ValueError('Original descriptions changed during run')
        from docci_longclip_joint_v8.core import image_path,image_index
        index=image_index(args.data_root)
        for item in views:
            if digest(image_path(args.data_root,index,item))!=item['image_sha256']: raise ValueError('Original image changed during run')
        if any(digest(f)!=identity['source_sha256'][f.name] for f in files): raise ValueError('Runtime source changed')
        atomic_json(root/'frozen_verification.json',dict(passed=True,pretrained_unchanged=True,data_unchanged=True,source_unchanged=True,
            no_original_masks_lengths_or_oracle_features_to_RX=True,actual_image_size=c['image_size']))
    try:
        torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed']); np.random.seed(c['seed'])
        torch.backends.cudnn.benchmark=False
        train,val,receipt=records(args.data_root,c['seed'])
        if receipt['official_counts']!={'train':9647,'test':5000,'qual_dev':100,'qual_test':100}: raise ValueError('Original complete official DOCCI annotation counts differ; no COCO/subset substitution')
        if len(train)<design_schedule['train_samples'] or len(val)<design_schedule['val_samples']: raise ValueError('Insufficient official DOCCI split; no silent reduction')
        train=train[:design_schedule['train_samples']]; val=val[:design_schedule['val_samples']]
        if {x['example_id'] for x in train}&{x['example_id'] for x in val}: raise ValueError('Train/qual_dev overlap')
        atomic_json(root/'docci_view.json',dict(receipt,train=train,validation=val,old_ln_split_untouched=True))
        models,tok,ctok,mr=load_models(args,device); signatures=[signature(m) for m in models]
        atomic_json(root/'checkpoint_load.json',mr)
        items,trainreceipt=cache_view(args,c,train,models,tok,ctok,mr,device,budget,'train')
        validation,valreceipt=cache_view(args,c,val,models,tok,ctok,mr,device,budget,'qual_dev')
        design=dict(config=c,schedule=design_schedule,train_view=trainreceipt,validation_view=valreceipt,arms=['separate','joint'],
            train_split='official DOCCI train',validation_split='official DOCCI qual_dev',test_used=False,
            no_diffusion=True,backbone_training=False,common_and_source_and_receiver_training=True,
            source_images='original DOCCI full view, letterbox, NOT upscaled historical128 images',
            longclip_scope='native224 image tower, native248 text chunks; ordered chunk features retained',
            long_text_scope='full BART tokens, fixed public64 slots; no free source length or attention mask at RX',
            fso_training_scope='differentiable Gamma-Gamma/pointing/complex noise/pilot symbol proxy; no waveform DSP or correlated weather in training',
            physical_eval_scope='existing assumed V3 coherent optical waveform/DSP and correlated weather, not calibrated OptiSystem or measured link',
            clean_vs_fso_scope='FSO continues clean for extra steps; NOT same-total-update causal noise ablation',
            clean_anchor_fraction=c['clean_anchor_weight'],text_teacher_scope='prefit private memory TRAIN-only .01 anchor',
            checkpoint_selection='fixed last; validation controls are engineering gate, no test/qual_test selection')
        atomic_json(root/'design.json',design)
        model=JointModel(models[0],c).to(device).float().eval()
        if args.text_init:
            init_path=Path(args.text_init).resolve(strict=True)
            done=init_path.with_name('TRAIN_COMPLETE.json')
            if not done.is_file() or read(done).get('checkpoint_sha256')!=digest(init_path): raise ValueError('Own completed text-init identity not verified')
            atomic_json(root/'text_warm_start.json',warm_text(model,init_path))
        counts=dict(total_registered=sum(p.numel() for p in model.parameters()),new_trainable=sum(p.numel() for n,p in model.named_parameters() if not n.startswith('text.bart.')),
            registered_parameters_identical_in_two_arms=True,active_attention_length_compute_not_identical=True)
        atomic_json(root/'parameter_counts.json',counts)
        # Authentic Long-CLIP EOS pooling parity plus a nonempty actual native forward.
        with torch.no_grad():
            sample=torch.tensor([[ctok.bos_token_id,ctok.eos_token_id]+[0]*246],device=device)
            full=models[2].encode_text_full(sample)
            pooled=full[torch.arange(full.shape[0],device=device),sample.argmax(-1)]@models[2].text_projection
            if not torch.allclose(pooled,models[2].encode_text(sample),rtol=2e-5,atol=2e-5): raise ValueError('Official Long-CLIP full/EOS parity failed')
        native_diagnostics(model,validation[:2],models,tok,device,budget,root/'native_diagnostics.jsonl')
        (root/'prefit').mkdir()
        fit(model,items,c,models,device,root/'prefit/text','text',design_schedule['text_steps'],budget)
        controls=text_controls(model,validation,tok,device,budget,root/'prefit/validation_controls.jsonl','official DOCCI qual_dev prefit')
        gate=text_gate(controls,c); atomic_json(root/'prefit/text_gate.json',gate)
        fit(model,items,c,models,device,root/'prefit/image','image',design_schedule['image_steps'],budget)
        if design_schedule['quality_gate_required'] and not gate['passed']:
            unchanged(models,signatures,receipt,items+validation)
            identity.update(status='quality_stop',wall_seconds=time.monotonic()-started)
            atomic_json(root/'run_identity.json',identity); finish(root)
            print('QUALITY_STOP: private long-text gate not passed. Preserve diagnostics; joint not trained. No automatic threshold lowering.',flush=True)
            return
        with torch.no_grad():
            for item in items:
                budget(); s=tensors(item,device)
                item['text_anchor']=model.text.receive_text(model.text.transmit(s['bart_hidden'].float(),s['ids'].ne(tok.pad_token_id)))['memory'][0].cpu().numpy()
        initial=model.saved(); states={}
        atomic_json(root/'initial_state_hashes.json',{n:__import__('hashlib').sha256(v.numpy().tobytes()).hexdigest() for n,v in initial.items()})
        for arm in ('separate','joint'):
            (root/arm).mkdir(); model.configure(arm); model.restore(initial)
            states[(arm,'clean')]=fit(model,items,c,models,device,root/arm/'clean','clean',design_schedule['clean_steps'],budget)
            states[(arm,'fso')]=fit(model,items,c,models,device,root/arm/'fso','fso',design_schedule['fso_steps'],budget)
        evaluate(model,states,validation,c,models,tok,device,budget,root/'evaluation',args.mode=='smoke')
        unchanged(models,signatures,receipt,items+validation)
        identity.update(status='finished_execution',wall_seconds=time.monotonic()-started)
        atomic_json(root/'run_identity.json',identity); finish(root)
        print('V8 done. Execution completion does not imply improvement; inspect paired results.',flush=True)
    except BaseException as exc:
        atomic_json(root/'FAILED.json',dict(type=type(exc).__name__,message=str(exc),wall_seconds=time.monotonic()-started)); raise


if __name__=='__main__': main()
