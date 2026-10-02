"""Real COCO + frozen A + sampled SI optics; validation-only final direction screen."""
import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT))
from final_compare_v1 import VERSION, METHODS, REFERENCES
from final_compare_v1.spec import PLAN, load_plan, effective_conditions, resources
from final_compare_v1.codec import choose_source, decode_image, decode_text
from final_compare_v1.metrics import caption_metrics, task_metrics
from final_compare_v1.transport import BudgetTransport
from final_compare_v1.report import write_report, verify, digest
from optical_link_v1.config import load as load_optical, TEMPLATE


def main():
    import torch
    import v10.model as module
    from v10.runtime import add_shared_args, preflight, load_config
    from v10.io import begin_run, atomic_json
    from v10.data import build_loader, sha256
    from v10.model import build_model
    from utils.checkpoint import load_checkpoint
    from utils.metrics import batch_image_metrics, token_accuracy, sentence_bleu2
    from business_link.engine import identity_seed
    from token_guidance.model import Guidance
    from token_guidance.wire import quantized
    from final_compare_v1.model import ReplayFrame
    from optical_link_v1.wire import send_tokens
    from optical_link_v1.plots import constellation_svg
    from eval_v10 import save_image

    p = add_shared_args(argparse.ArgumentParser(description=__doc__, allow_abbrev=False))
    p.add_argument('--a-delta', required=True)
    p.add_argument('--optical-config', default=str(TEMPLATE))
    p.add_argument('--plan', default=str(PLAN))
    p.add_argument('--val-samples', type=int, default=32)
    p.add_argument('--draws', type=int, default=3)
    p.add_argument('--save-images', type=int, default=8)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if not 1 <= args.val_samples <= 500 or not 1 <= args.draws <= 8 or not 0 <= args.save_images <= args.val_samples:
        p.error('1..500 validation images, 1..8 channel draws, bounded examples')
    if not math.isfinite(args.max_hours) or not 0 < args.max_hours <= 6:
        p.error('Finite overall budget >0 and <=6h')
    args.batch_size, args.num_workers, args.no_amp, args.fixed_mode, args.csi, args.blocks = 1, 0, True, 0, 'pilot', 1
    plan, plan_hash = load_plan(args.plan)
    specification, cases, config_hash = load_optical(args.optical_config)
    conditions = effective_conditions(cases, plan, args.smoke)
    for optical in cases.values(): optical['waveform']['policy'] = plan['digital_policy']
    reference_resource = resources(cases[conditions[0]['case']], plan)
    # Baseline source quality must not change between fading/power sweep points.
    for c in conditions:
        if resources(cases[c['case']], plan) != reference_resource:
            raise ValueError('One shared waveform/resource contract required for source codec selection')
    device = preflight(args)
    config = load_config(args)
    base, delta = load_checkpoint(args.checkpoint), load_checkpoint(args.a_delta)
    split_hash, base_hash, delta_hash = sha256(args.split_manifest), sha256(args.checkpoint), sha256(args.a_delta)
    if (not str(base.get('version','')).startswith('V10-') or base.get('split_sha256') != split_hash
            or delta.get('version') != 'V10-token-guidance-quick-v1' or delta.get('variant') != 'A'
            or delta.get('base_checkpoint_sha256') != base_hash or delta.get('split_sha256') != split_hash):
        raise ValueError('Require exact A delta, original V10 base and matching frozen split')
    config['v10'] = copy.deepcopy(base['config']['v10'])
    config['v10'].update(blocks=1, csi='pilot', fixed_mode=0)
    del base
    directory, identity = begin_run(args.output_dir, args, config, EXT)
    started, rows = time.monotonic(), []
    identity.update(version=VERSION, a_delta_sha256=delta_hash, optical_config_sha256=config_hash,
        plan_sha256=plan_hash, scope='frozen_A_validation_final_direction_screen_no_training',
        extension_source_sha256={str(f.relative_to(EXT)):sha256(f) for folder in ('final_compare_v1','optical_link_v1','token_guidance','business_link','physical_link') for f in sorted((EXT/folder).iterdir()) if f.suffix in ('.py','.json','.md')})
    atomic_json(directory/'run_identity.json', identity)
    def budget():
        if time.monotonic()-started >= args.max_hours*3600:
            raise TimeoutError('Wall-time cap reached; preserve partial records, never write COMPLETE')
    try:
        print('Loading frozen A; NO training; common slot/launch-power ceilings, NO preset SNR.', flush=True)
        model, report = build_model(config, device, args.checkpoint)
        if report != {'v10_exact':True}: raise ValueError('Exact base loading required')
        guide = Guidance(model)
        if set(delta['delta']) != {name for name,_ in guide.trainable('A')}:
            raise ValueError('A delta has missing/unexpected keys')
        combined = dict(model.state_dict(), **delta['delta'])
        model.load_state_dict(combined, strict=True)
        del combined, delta
        model.requires_grad_(False); model.eval(); guide.set('A')
        atomic_json(directory/'checkpoint_load.json', dict(base_exact=True, a_delta_exact=True,
            base_checkpoint_sha256=base_hash, a_delta_sha256=delta_hash, trained=False))
        loader = build_loader(config,args.split_manifest,'validation',args.val_samples,args.seed)
        loader.dataset.caption_mode = 'first'
        (directory/'images').mkdir(); (directory/'constellations').mkdir()
        expected = args.val_samples*(len(REFERENCES)+len(conditions)*args.draws*len(METHODS))
        design = dict(version=VERSION, split='validation', variant='A', trained=False,
            expected_rows=expected, val_samples=args.val_samples, draws=args.draws,
            conditions=conditions, methods=list(METHODS), references=list(REFERENCES),
            optical_config=specification, effective_cases=cases, plan=plan, resource_contract=reference_resource,
            save_images=args.save_images, smoke=args.smoke,
            repeated_channel_draws_are_not_training_seeds=True,
            field_is_paired_across_methods_and_power_sweep=True,
            payload_noise_arrays_are_not_identical_across_different_waveform_lengths=True,
            pilot_estimate_shared_not_offline_physical_snr=True,
            receiver_no_source_caption_branch=True,
            analog_is_frozen_digital_A_compatibility_not_optimized_DeepJSCC=True)
        atomic_json(directory/'design.json',design)
        with (directory/'samples.jsonl').open('x',encoding='utf-8') as stream, (directory/'packets.jsonl').open('x',encoding='utf-8') as packets:
            seen_ids = []
            with torch.no_grad():
                for index,batch in enumerate(loader):
                    budget()
                    image_id,caption = int(batch['image_ids'][0]),batch['texts'][0]
                    seen_ids.append(image_id)
                    images = batch['images'].to(device)
                    source = batch['images'][0].permute(1,2,0).numpy()
                    latent = model.base.vae_wrapper.encode_images(images)
                    def neural(context):
                        if device.type=='cuda': torch.cuda.synchronize(device)
                        with patch.object(module,'LinkFrame',lambda *a,**kw: ReplayFrame(*a,context=context,**kw)):
                            output = model(images,[caption],'awgn',0.,vae_latents_override=latent)
                        if device.type=='cuda': torch.cuda.synchronize(device)
                        return output
                    floating = neural(dict(capture=True,probe_snr_db=40.))
                    values = model.last_frame.captured
                    _,q8,clipped = quantized(values)
                    source_hash = hashlib.sha256(source.astype('<f4').tobytes()).hexdigest()
                    payload_hash = hashlib.sha256(values.astype('<f4').tobytes()).hexdigest()
                    codec = choose_source(source,caption,cases[conditions[0]['case']],plan,reference_resource['slot_samples'])
                    clean_image, clean_image_ok = decode_image(codec['offered_image'])
                    clean_text, clean_text_ok = decode_text(codec['text'])
                    if not clean_image_ok or not clean_text_ok: raise RuntimeError('Local source codec roundtrip failed')
                    if index < args.save_images:
                        source_path = directory/'images'/('{}__source.png'.format(image_id))
                        save_image(batch['images'][0],source_path)
                    def emit(method,condition,draw,predicted,text,stats,output=None,transport=None,wall=0.):
                        prediction = torch.as_tensor(predicted,device=device,dtype=images.dtype).permute(2,0,1)[None] if isinstance(predicted,np.ndarray) else predicted
                        m = batch_image_metrics(prediction,images)
                        row = dict(stats, version=VERSION, method=method, condition=condition, draw=draw, image_id=image_id,
                            reference=method in REFERENCES, psnr=float(m['psnr'][0]),ssim=float(m['ssim'][0]),
                            source_caption_evaluation_only=caption,received_caption=text,
                            source_caption_sha256=hashlib.sha256(caption.encode()).hexdigest(),
                            source_rgb_sha256=source_hash,normalized_payload_sha256=payload_hash,
                            source_rgb8_bytes=128*128*3,source_utf8_bytes=len(caption.encode()),
                            neural_and_cpu_transport_wall_seconds=wall,nominal_snr_db=None,
                            jpeg_quality=codec['jpeg_quality'],traditional_codec_budget_fits=codec['budget_fits'],
                            source_clip_fraction=clipped if method in ('semantic_q8_clean','semantic_digital') else 0.,
                            token_acc=None,token_correct=None,token_count=None,bleu2=sentence_bleu2(caption,text),
                            **caption_metrics(caption,text))
                        if output is not None:
                            correct,count=token_accuracy(output['text_logits'],output['text_input_ids'],output['text_attention_mask'])
                            row.update(token_acc=correct/max(count,1),token_correct=correct,token_count=count,
                                neural_text_input_capacity=int(output['text_input_ids'].shape[1]),
                                source_text_may_exceed_neural_token_capacity=True)
                        row.update(task_metrics(row,plan))
                        if transport is not None:
                            row['event_count']=len(transport.events)
                            row.update(field_seed=transport.field_seed,noise_seed=transport.seed,
                                routing_probe_snr_db=transport.events[0]['estimated_snr_db'],
                                routing_snr_clamped_db=max(-40.,min(40.,transport.events[0]['estimated_snr_db'])),
                                task_qualified_source_equivalent_bps=(row['source_rgb8_bytes']+row['source_utf8_bytes'])*8*int(row['joint_task_success'])/row['scheduled_seconds'],
                                task_throughput_not_lossless_byte_goodput=True)
                            if method=='traditional_digital':
                                row.update(lossless_text_goodput_bps=len(caption.encode())*8*int(row['text_exact'] and row['text_object_crc_accepted'])/row['scheduled_seconds'],
                                    source_codec_bytes=row['compressed_text_wire_bytes']+row['image_wire_bytes'],
                                    rgb8_plus_utf8_to_codec_ratio=(row['source_rgb8_bytes']+row['source_utf8_bytes'])/(row['compressed_text_wire_bytes']+row['image_wire_bytes']),
                                    compression_ratio_note='lossy RGB8 JPEG plus lossless compressed UTF8, including object headers/CRC; excludes wire overhead')
                            elif method=='semantic_digital':
                                row.update(source_codec_bytes=row['feature_object_wire_bytes'],
                                    rgb8_plus_utf8_to_codec_ratio=(row['source_rgb8_bytes']+row['source_utf8_bytes'])/row['feature_object_wire_bytes'],
                                    compression_ratio_note='lossy joint semantic representation vs RGB8+UTF8; NOT lossless text compression')
                            else:
                                row.update(source_codec_bytes=None,rgb8_plus_utf8_to_codec_ratio=None,
                                    compression_ratio_note='continuous real/complex symbols have no binary wire byte count')
                            for event in transport.events:
                                packets.write(json.dumps(dict(event,image_id=image_id,condition=condition,draw=draw,method=method),allow_nan=False)+'\n')
                            packets.flush()
                        if index < args.save_images:
                            suffix='{}__{}__{}__draw{}'.format(image_id,condition,method,draw)
                            row.update(source_image_file='images/'+source_path.name,image_file='images/'+suffix+'.png')
                            save_image(prediction[0],directory/row['image_file'])
                            if transport is not None and transport.constellations:
                                row['constellation_file']='constellations/'+suffix+'.svg'
                                constellation_svg(transport.constellations,directory/row['constellation_file'])
                            for name in ('source_image_file','image_file','constellation_file'):
                                if name in row: row[name+'_sha256']=digest(directory/row[name])
                        if not all(math.isfinite(row[k]) for k in ('psnr','ssim','normalized_wer','bleu2')):
                            raise FloatingPointError('Nonfinite task metrics')
                        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()
                        rows.append(row)
                        print('EVAL {}/{} id={} {} {} PSNR={:.2f} WER={:.3f} exact={} elapsed={:.1f}s'.format(
                            len(rows),expected,image_id,condition,method,row['psnr'],row['normalized_wer'],row['text_exact'],time.monotonic()-started),flush=True)
                    emit('semantic_float_clean','clean',0,floating['image'],model.base.decode_text_logits(floating['text_logits'])[0],dict(source_quantization_mse=0.),output=floating)
                    del floating
                    clean = neural(dict(received=q8,probe_snr_db=40.,available=True,reliability=1.))
                    emit('semantic_q8_clean','clean',0,clean['image'],model.base.decode_text_logits(clean['text_logits'])[0],dict(source_quantization_mse=float(np.mean((q8-values)**2))),output=clean)
                    del clean
                    emit('traditional_clean','clean',0,clean_image,clean_text,dict(image_object_crc_accepted=True,text_object_crc_accepted=True,feature_object_crc_accepted=None))
                    for condition in conditions:
                        optical = cases[condition['case']]
                        resource = resources(optical,plan)
                        for draw in range(args.draws):
                            # Equal source/field/probe; not falsely identical data noise arrays.
                            field_seed = identity_seed(args.seed,image_id,draw,condition['case'],'final_field')
                            noise_seed = identity_seed(args.seed,image_id,draw,condition['case'],'final_noise')
                            for method in METHODS:
                                budget(); before=time.monotonic()
                                transport=BudgetTransport(optical,field_seed,noise_seed,condition['launch_signal_dbm'],resource['slot_samples'],plan['laser_max_dbm'])
                                estimated=transport.probe()
                                if method=='traditional_digital':
                                    if codec['budget_fits']:
                                        text_wire,text_packets_ok,text_record=transport.transfer(codec['text'],'text',plan['digital_policy'])
                                        image_wire,image_packets_ok,image_record=transport.transfer(codec['image'],'features',plan['digital_policy'])
                                        text,text_ok=decode_text(text_wire if text_packets_ok else b'')
                                        prediction,image_ok=decode_image(image_wire if image_packets_ok else b'')
                                    else:
                                        text,text_ok='',False; prediction,image_ok=decode_image(b'')
                                        text_record=image_record=None
                                    stats=dict(transport.totals(),text_object_crc_accepted=text_ok,image_object_crc_accepted=image_ok,
                                        feature_object_crc_accepted=None,text_packets=text_record,image_packets=image_record,
                                        compressed_text_wire_bytes=len(codec['text']),image_wire_bytes=len(codec['offered_image']),
                                        policy=plan['digital_policy'],source_quantization_mse=None,
                                        resource_refused_before_payload=not codec['budget_fits'],
                                        text_receiver='zlib+UTF8+object_CRC; no neural text decoder',
                                        raw_utf8_wire_bytes=0,original_utf8_source_bytes=len(caption.encode()),
                                        traditional_image_wire_sha256=hashlib.sha256(codec['offered_image']).hexdigest(),
                                        traditional_text_wire_sha256=hashlib.sha256(codec['text']).hexdigest())
                                    emit(method,condition['id'],draw,prediction,text,stats,transport=transport,wall=time.monotonic()-before)
                                else:
                                    if method=='semantic_digital':
                                        received,stats=send_tokens(values,transport)
                                        available=stats['feature_object_crc_accepted']; reliability=float(available)
                                    else:
                                        received,reliability,stats=transport.continuous(values,resource['analog_repeats'],plan['analog_equalizer'])
                                        available=True  # not CRC-certified: task success evaluated separately.
                                    output=neural(dict(received=received,probe_snr_db=estimated,available=available,reliability=reliability,stats=stats))
                                    emit(method,condition['id'],draw,output['image'],model.base.decode_text_logits(output['text_logits'])[0],stats,
                                        output=output,transport=transport,wall=time.monotonic()-before)
                                    del output
                    del latent, images
                    atomic_json(directory/'progress.json',dict(rows=len(rows),expected_rows=expected,completed_images=index+1,
                        elapsed_seconds=time.monotonic()-started,projected_total_seconds=(time.monotonic()-started)*args.val_samples/(index+1)))
        atomic_json(directory/'validation_ids.json',seen_ids)
        budget()
        write_report(directory)
        verify(directory)
        result_files=['design.json','validation_ids.json','checkpoint_load.json','samples.jsonl','packets.jsonl','summary.json','REPORT.md','report.html']
        result_files += [f.name for f in sorted(directory.glob('*.svg'))]
        identity['status']='complete'; atomic_json(directory/'run_identity.json',identity)
        atomic_json(directory/'COMPLETE.json',dict(version=VERSION,rows=len(rows),expected_rows=expected,
            wall_seconds=time.monotonic()-started,trained=False,hardware_calibrated=False,
            result_sha256={name:digest(directory/name) for name in result_files}))
        verify(directory,require_complete=True)
        print('COMPLETE: '+str(directory/'report.html'),flush=True)
    except BaseException as exc:
        identity.update(status='failed_or_interrupted',error=str(exc),exception=type(exc).__name__)
        atomic_json(directory/'run_identity.json',identity)
        raise


if __name__=='__main__': main()
