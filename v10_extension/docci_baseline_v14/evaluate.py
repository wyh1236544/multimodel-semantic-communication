"""Identical clean frames; separate greedy/deployment/oracle evaluation tracks."""
import json
import numpy as np
import torch
from PIL import Image
from docci_baseline_v14.core import atomic, sha, array_hash, geometry, DECODE
from docci_baseline_v14.data import tensors
from docci_baseline_v14.model import decode
from docci_text_continue_v11.metrics import metrics
from utils.metrics import batch_image_metrics


@torch.no_grad()
def evaluate(model, items, task, backbone, tok, device, folder, budget,
             cohort, save_images=4, native=False, perceptual=None):
    folder.mkdir(exist_ok=False)
    (folder/'frames').mkdir()
    (folder/'images').mkdir()
    selected = items.entries
    if len(selected) < 2:
        raise ValueError('Real different-source control requires >=2 samples')
    controls = ['native'] if native else ['matched', 'shuffled', 'zero']
    strategies = list(DECODE) if task == 'text' else ['image']
    matrix = dict(task=task, cohort=cohort, arm='native' if native else model.arm,
                  example_ids=[e['example_id'] for e in selected], controls=controls,
                  strategies=strategies, geometry=None if native else geometry(task),
                  native_privileged_NOT_transmission=native,
                  source_masks_at_RX=native and task=='text',
                  zero_scope='post-reconstruction memory' if task=='text' else 'received wire',
                  LPIPS_status='NOT_MEASURED' if perceptual is None else 'MEASURED_OFFLINE', facts='PENDING_V13_HUMAN_AUDIT',
                  description_level_paired_analysis=True, evaluation_split='qual_dev' if cohort=='validation' else 'train')
    atomic(folder/'matrix.json', matrix)
    wires = []
    if not native:
        for i in range(len(items)):
            budget()
            src = tensors(items, [i], task, device)
            wire = model.encode(src['chunk_ids'], src['chunk_hidden']) if task=='text' else model.encode(src)
            value = wire[0].float().cpu().numpy()
            if value.shape != (geometry(task)['payload_reals'],) or not np.isfinite(value).all():
                raise ValueError('Invalid transmitted frame')
            if abs(float(np.mean(value**2))-1.) > .0002:
                raise ValueError('Transmitted power differs from unit contract')
            path = folder/'frames'/(selected[i]['example_id']+'.npz')
            with path.open('xb') as f:
                np.savez_compressed(f, observed=value)
            wires.append(dict(value=value, file=path.relative_to(folder).as_posix(),
                              sha256=sha(path), array_sha256=array_hash(value)))
    count = 0
    with (folder/'rows.jsonl').open('x', encoding='utf-8') as log:
        for i in range(len(items)):
            item = items[i]
            donor = (i+1) % len(items)
            for control in controls:
                budget()
                meta = dict(example_id=item['example_id'], official_split=item['official_split'],
                            cohort=cohort, task=task, control=control,
                            condition='ideal_clean', snr_db=None, native=native,
                            frame_file=None if native else wires[i]['file'],
                            frame_sha256=None if native else wires[i]['sha256'],
                            tx_array_sha256=None if native else wires[i]['array_sha256'],
                            donor_example_id=selected[donor]['example_id'] if control=='shuffled' else None,
                            received_array_sha256=None if native else wires[donor if control=='shuffled' else i]['array_sha256'])
                if task == 'text':
                    if native:
                        rx = dict(memory=torch.as_tensor(item['chunk_hidden'], device=device)[None].float(),
                                  receiver_mask=torch.as_tensor(item['chunk_ids'], device=device)[None].ne(tok.pad_token_id))
                    else:
                        wire = torch.as_tensor(wires[donor if control=='shuffled' else i]['value'], device=device)[None]
                        rx = model.receive(wire)
                        if control == 'zero':
                            rx['memory'] = torch.zeros_like(rx['memory'])
                    for strategy in strategies:
                        budget()
                        result = decode(backbone, rx, tok, strategy)
                        row = dict(meta, strategy=strategy, source_text=item['text'],
                                   source_text_sha256=item['text_sha256'], source_tokens=item['source_tokens'],
                                   input_truncated=False, **result,
                                   **metrics(item['text'], result['received_caption']))
                        log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                        count += 1
                else:
                    target = torch.as_tensor(item['rgb'].copy(), device=device).permute(2, 0, 1)[None].float()/255.
                    if native:
                        latent = torch.as_tensor(item['latent'], device=device)[None].float()
                    else:
                        wire = torch.as_tensor(wires[donor if control=='shuffled' else i]['value'], device=device)[None]
                        if control == 'zero':
                            wire = torch.zeros_like(wire)
                        latent = model.receive(wire, torch.ones(1, device=device, dtype=torch.bool),
                                               wire.new_full((1,), 40.), wire.new_ones(1))['latent']
                    rgb = ((backbone.decode(latent/float(backbone.config.scaling_factor)).sample+1)/2).clamp(0, 1)
                    l, t, r, b = map(int, item['rectangle'])
                    score = batch_image_metrics(rgb[:, :, t:b, l:r], target[:, :, t:b, l:r])
                    # Explicit native full-image reference, evaluated on SAME rectangle.
                    row = dict(meta, strategy='image', psnr=float(score['psnr'][0]),
                               ssim=float(score['ssim'][0]), evaluation_rectangle=[l, t, r, b],
                               source_image_sha256=item['image_sha256'], LPIPS=None,
                               restored_image_file=None, source_view_file=None)
                    if perceptual is not None:
                        row['LPIPS'] = float(perceptual(rgb[:, :, t:b, l:r]*2-1,
                                                       target[:, :, t:b, l:r]*2-1).mean())
                    if i < save_images:
                        name = 'images/{}__{}.png'.format(item['example_id'], control)
                        image = (rgb[0].permute(1, 2, 0).cpu().numpy()*255).round().clip(0, 255).astype(np.uint8)
                        Image.fromarray(image).save(folder/name)
                        source = 'images/{}__source.png'.format(item['example_id'])
                        if not (folder/source).exists():
                            Image.fromarray(item['rgb']).save(folder/source)
                        row.update(restored_image_file=name, source_view_file=source,
                                   restored_image_sha256=sha(folder/name), source_view_sha256=sha(folder/source))
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    count += 1
                log.flush()
            if i == 0 or (i+1) % 10 == 0 or i+1 == len(items):
                print('EVAL {} {} {}/{} all controls/strategies'.format(matrix['arm'], cohort, i+1, len(items)), flush=True)
    atomic(folder/'EVAL_COMPLETE.json', dict(rows=count, rows_sha256=sha(folder/'rows.jsonl'),
           matrix_sha256=sha(folder/'matrix.json'), native=native))
