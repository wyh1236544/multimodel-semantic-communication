"""Separate offline frozen-SD baselines and TRAIN-only semantic adapter diagnostic."""
import argparse
import inspect
import json
from pathlib import Path
import sys
import time
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent; sys.path.insert(0,str(EXT))
from clip_bart_joint_v6.core import digest
from clip_bart_joint_v6.report import verify,read,lines
from v10.paths import write_path


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--screen-run',required=True)
    p.add_argument('--diffusion-model-path',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--phase',choices=('clean','fso'),default='clean')
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--adapter-updates',type=int,default=500)
    p.add_argument('--strength',type=float,default=.25)
    p.add_argument('--max-hours',type=float,default=1.)
    a=p.parse_args()
    if not (1<=a.steps<=50 and 4<=a.adapter_updates<=2000 and 0<a.strength<=.5 and 0<a.max_hours<=4):
        raise ValueError('Bounded generation config')
    root=Path(a.screen_run).resolve(strict=True); proof=verify(root)
    identity=read(root/'run_identity.json')
    if proof['smoke'] or not proof['arm_status']['J1_joint']['trained']:
        raise ValueError('Actual non-smoke V6 joint screen required')
    # Cache uses last trained J1 stage; refuse mismatched clean/fso evaluation.
    expected='fso' if proof['arm_status']['J1_joint']['noise_trained'] else 'clean'
    if a.phase!=expected: raise ValueError('Use phase matching cache checkpoint: '+expected)
    folder=root/'J1_joint'/('eval_'+a.phase)
    model_path=Path(a.diffusion_model_path).resolve(strict=True)
    if not (model_path/'model_index.json').is_file(): raise FileNotFoundError('Complete local SD directory required, not VAE-only weights')
    target=write_path(a.output_dir,PROJECT)
    if target.exists(): raise FileExistsError('New generation output required')
    if target.relative_to(PROJECT).parts[0]!='experiment_results': raise ValueError('Project experiment_results only')
    if identity['smoke']: raise ValueError('No smoke generation')
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='1'; os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    import numpy as np
    from PIL import Image
    from diffusers import StableDiffusionImg2ImgPipeline
    from clip_bart_joint_v6.prompt_adapter import PromptAdapter
    from bart_latent_codec_v4.workflow import signature,frozen_check
    from v10.io import atomic_json,atomic_checkpoint
    from utils.metrics import batch_image_metrics
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    free,_=torch.cuda.mem_get_info(0)
    if free<12*2**30: raise MemoryError('Need12GiB free for SD; do not stop unrelated processes')
    target.mkdir(parents=True,exist_ok=False); start=time.monotonic()
    def budget():
        if time.monotonic()-start>a.max_hours*3600: raise TimeoutError('Generation cap')
        free,_=torch.cuda.mem_get_info(0)
        if free<4*2**30: raise MemoryError('GPU reserve breached')
    try:
        budget()
        pipe=StableDiffusionImg2ImgPipeline.from_pretrained(str(model_path),local_files_only=True,
            torch_dtype=torch.float16).to('cuda:0')
        pipe.enable_attention_slicing()
        if not {'prompt_embeds','negative_prompt_embeds'}.issubset(inspect.signature(pipe.__call__).parameters):
            raise RuntimeError('Existing pipeline does not accept explicit prompt embeddings; no silent fallback/upgrade')
        frozen=[]
        for m in (pipe.text_encoder,pipe.unet,pipe.vae):
            m.eval().requires_grad_(False); frozen.append((m,signature(m)))
        dim=pipe.text_encoder.config.hidden_size
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        adapter=PromptAdapter(dim).to('cuda:0').float()
        initial={n:v.detach().cpu().clone() for n,v in adapter.named_parameters()}
        def embed(texts):
            token=pipe.tokenizer(texts,padding='max_length',max_length=pipe.tokenizer.model_max_length,
                truncation=True,return_tensors='pt')
            with torch.no_grad():
                hidden=pipe.text_encoder(token.input_ids.to('cuda:0'))[0].float()
            return hidden,token.attention_mask.to('cuda:0').float()
        cachefile=root/'J1_joint/generation_train.npz'
        with np.load(cachefile,allow_pickle=False) as cache:
            if set(cache['source_split'].tolist())!={'train'}: raise ValueError('No validation labels in adapter training')
            common=torch.tensor(cache['common'],device='cuda:0')
            train_snr=torch.tensor(cache['pilot_snr_db'],device='cuda:0',dtype=torch.float32)
            received,_=embed(cache['received_caption'].tolist())
            teacher,teacher_mask=embed(cache['source_text_train_teacher'].tolist())
            train_ids=cache['image_id'].tolist()
        opt=torch.optim.AdamW(adapter.parameters(),lr=1e-4); observed=set()
        atomic_json(target/'identity.json',dict(version='V6-generation-semantic-adapter-diagnostic',
            args=vars(a),screen_manifest_sha256=digest(root/'RECORDS_MANIFEST.json'),
            train_cache_sha256=digest(cachefile),train_ids=train_ids,
            original_text_in_receiver=False,SD_frozen=True,ControlNet_DiT_replication=False,
            weight_source='existing complete local SD',selection='none/all_fixed_candidates',
            all_communication_resources_unchanged=True))
        with (target/'adapter_history.jsonl').open('x') as log:
            for update in range(a.adapter_updates):
                budget(); idx=[(update*4+j)%len(common) for j in range(4)]
                prediction=adapter(common[idx],received[idx],train_snr[idx])
                mask=teacher_mask[idx][:,:,None]
                loss=((prediction-teacher[idx]).square()*mask).sum()/(mask.sum()*dim)
                opt.zero_grad(set_to_none=True); loss.backward()
                for n,weight in adapter.named_parameters():
                    if weight.grad is not None:
                        if not torch.isfinite(weight.grad).all(): raise FloatingPointError('Adapter nonfinite gradient')
                        if bool(weight.grad.abs().sum()>0): observed.add(n)
                torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True); opt.step()
                log.write(json.dumps(dict(update=update+1,embedding_mse=float(loss.detach())))+'\n'); log.flush()
                if update%50==0: print('SD semantic adapter {}/{} loss={:.4f}'.format(update+1,a.adapter_updates,float(loss)),flush=True)
        state={n:v.detach().cpu().clone() for n,v in adapter.named_parameters()}
        audit={prefix:dict(changed=any(n.startswith(prefix) and not torch.equal(initial[n],v) for n,v in state.items()),
            nonzero_gradient=any(n.startswith(prefix) for n in observed)) for prefix in ('common.','snr.','attention.','output.','gate')}
        if not all(all(v.values()) for v in audit.values()): raise RuntimeError('Adapter required groups did not learn')
        atomic_checkpoint(target/'last_adapter.pth',dict(weights=state,scope='TRAIN semantic embedding regression, not diffusion denoising training'))
        adapter.load_state_dict(torch.load(str(target/'last_adapter.pth'),map_location='cpu')['weights'],strict=True); adapter.eval()
        if any(not torch.equal(v,dict(adapter.named_parameters())[n].detach().cpu()) for n,v in state.items()):
            raise RuntimeError('Actual adapter checkpoint reload differs')
        atomic_json(target/'learning_audit.json',dict(passed=True,groups=audit,exact_disk_reload=True))
        normal=[r for r in lines(folder/'samples.jsonl') if r['control']=='normal' and r.get('receiver_condition_file')]
        # Received arrays are independently hash checked by screen manifest.
        outputs=[]
        with (target/'samples.jsonl').open('x',encoding='utf-8') as log:
            for i,row in enumerate(normal):
                budget()
                image_path=folder/row['image_file']
                if row['receiver_failure']:
                    outputs.append(dict(image_id=row['image_id'],condition=row['condition'],status='receiver_failure_no_generation'))
                    log.write(json.dumps(outputs[-1])+'\n'); log.flush(); continue
                condpath=folder/row['receiver_condition_file']
                with np.load(condpath,allow_pickle=False) as cond:
                    if not bool(cond['receiver_available']) or str(cond['source_split'])!='validation' or int(cond['image_id'])!=row['image_id'] or str(cond['received_caption'])!=row['received_caption']:
                        raise ValueError('Received condition differs from recorded receiver')
                    cw=torch.tensor(cond['common'][None],device='cuda:0',dtype=torch.float32)
                    pilot_snr=torch.tensor([float(cond['pilot_snr_db'])],device='cuda:0')
                donor=normal[(i+1)%len(normal)]
                with np.load(folder/donor['receiver_condition_file'],allow_pickle=False) as dc:
                    shuffled=torch.tensor(dc['common'][None],device='cuda:0',dtype=torch.float32)
                negative,_=embed([''])
                base,_=embed([row['received_caption']])
                candidates={}
                for control in ('image_only','received_text','received_text_csi_only','received_text_common','shuffled_common'):
                    budget()
                    with torch.no_grad():
                        prompt=negative if control=='image_only' else base
                        if control in ('received_text_csi_only','received_text_common','shuffled_common'):
                            feature=cw if control=='received_text_common' else (shuffled if control=='shuffled_common' else torch.zeros_like(cw))
                            prompt=adapter(feature,base,pilot_snr)
                        # Whitelist receiver image/conditions only, never row source_text/image.
                        image=Image.open(image_path).convert('RGB').resize((512,512))
                        output=pipe(image=image,prompt_embeds=prompt.to(dtype=pipe.text_encoder.dtype),
                            negative_prompt_embeds=negative.to(dtype=pipe.text_encoder.dtype),strength=a.strength,
                            num_inference_steps=a.steps,guidance_scale=6.,
                            generator=torch.Generator(device='cuda:0').manual_seed(42000+i)).images[0]
                    name='{}__{}__{}.png'.format(row['condition'],row['image_id'],control)
                    output.resize((128,128)).save(target/name); candidates[control]=name
                # Original pixels are loaded only AFTER all inference, for evaluation.
                original=torch.tensor(np.asarray(Image.open(folder/row['source_image_file']).convert('RGB')).copy()/255.,
                    device='cuda:0',dtype=torch.float32).permute(2,0,1)[None]
                direct='{}__{}__direct_reconstruction.png'.format(row['condition'],row['image_id'])
                with (target/direct).open('xb') as f: f.write(image_path.read_bytes())
                source_name='{}__{}__source.png'.format(row['condition'],row['image_id'])
                with (target/source_name).open('xb') as f: f.write((folder/row['source_image_file']).read_bytes())
                candidates['direct_reconstruction']=direct
                for control,name in candidates.items():
                    path=target/name
                    pixels=torch.tensor(np.asarray(Image.open(path).convert('RGB')).copy()/255.,
                        device='cuda:0',dtype=torch.float32).permute(2,0,1)[None]
                    met={k:float(v[0]) for k,v in batch_image_metrics(pixels,original).items()}
                    record=dict(image_id=row['image_id'],condition=row['condition'],control=control,file=name,
                        received_caption=row['received_caption'],resources=row['geometry'],metrics=met,
                        communication_not_repeated=True,LPIPS=None,FID=None,
                        source_image_file=source_name,perception_metrics_not_installed_or_validated=True,selection='none')
                    outputs.append(record); log.write(json.dumps(record,ensure_ascii=False)+'\n'); log.flush()
        for m,before in frozen: frozen_check(m,before)
        atomic_json(target/'COMPLETE.json',dict(rows=len(outputs),wall_seconds=time.monotonic()-start,
            not_a_quality_success_marker=True,embedding_adapter_not_ControlNet_training=True))
        import html
        page=['<!doctype html><meta charset="utf-8"><h1>V6实际接收条件生成对照</h1>',
            '<p>固定首4个验证ID、全部链路条件。失锁不生成；全部候选展示，不挑图。适配器仅做TRAIN文本嵌入回归，不是扩散去噪训练。LPIPS/FID未测；图片更好看不代表原图细节更准确。</p>',
            '<table border="1"><tr><th>条件/ID/控制</th><th>实际接收文字</th><th>原图（评价）</th><th>恢复</th><th>PSNR/SSIM</th></tr>']
        for r in outputs:
            if r.get('status'):
                page.append('<tr><td>{}/{}</td><td colspan="4">失锁，未生成</td></tr>'.format(html.escape(r['condition']),r['image_id']))
            else:
                page.append('<tr><td>{}/{}/{}</td><td>{}</td><td><img width="128" src="{}"></td><td><img width="128" src="{}"></td><td>{:.2f}/{:.3f}</td></tr>'.format(
                    html.escape(r['condition']),r['image_id'],html.escape(r['control']),html.escape(r['received_caption']),
                    html.escape(r['source_image_file'],quote=True),html.escape(r['file'],quote=True),r['metrics']['psnr'],r['metrics']['ssim']))
        page.append('</table>')
        with (target/'actual_generation.html').open('x',encoding='utf-8') as f: f.write('\n'.join(page))
        import gzip,io,tarfile
        archive=target/'v10_clip_bart_joint_v6_generation_records.tar.gz'
        with archive.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
            for f in sorted(target.iterdir()):
                if not f.is_file() or f==archive or f.suffix=='.pth': continue
                data=f.read_bytes(); member=tarfile.TarInfo('v6_generation/'+f.name); member.size=len(data); tar.addfile(member,__import__('io').BytesIO(data))
        print('Send '+str(archive),flush=True)
    except BaseException as exc:
        atomic_json(target/'FAILED.json',dict(type=type(exc).__name__,message=str(exc))); raise


if __name__=='__main__': main()
