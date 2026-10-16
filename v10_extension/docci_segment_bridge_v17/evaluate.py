"""Five fixed real donors; own received text is fixed across auxiliary controls."""
import json
import torch
from docci_segment_bridge_v17.core import atomic, controls, eval_conditions, donor_index, kind, geometry
from docci_segment_bridge_v17.model import canonical, generate
from docci_text_continue_v11.metrics import metrics, body


@torch.no_grad()
def evaluate(model, bridge, tokenizer, prepared, arm, root, device, budget, smoke=False):
    entries = prepared.val.entries
    atomic(root/'matrix.json', dict(ids=[r['example_id'] for r in entries], conditions=list(eval_conditions(smoke)),
           controls=list(controls(smoke)), geometry=geometry(arm), source=kind(arm),
           unrestricted_greedy=True, source_length_only_for_metrics=True, facts='PENDING',
           off_reused_from_independent_frozen_baseline=True, original_image_branch_not_trained=True))
    count = 0
    with (root/'rows.jsonl').open('x', encoding='utf-8') as log:
        for channel in eval_conditions(smoke):
            for i, entry in enumerate(entries):
                memory, gate = prepared.observation(i, channel)
                source = prepared.observations[entry['example_id'], channel]
                for control in controls(smoke):
                    budget()
                    j = donor_index(i, len(entries), control) if control.startswith('donor') else i
                    other, _ = prepared.observation(j, channel)
                    visual = torch.as_tensor(prepared.val[j]['oracle_clip'][None], device=device)
                    aux = canonical(other if control.startswith('donor') else memory, visual, kind(arm))
                    if control == 'zero':
                        aux = torch.zeros_like(aux)
                    if control == 'off':
                        unchanged, _ = bridge.repair(memory, aux, gate, enabled=False)
                        if not torch.equal(memory, unchanged):
                            raise ValueError('Disabled bridge is not exact identity')
                        baseline = prepared.baseline[entry['example_id'], channel]
                        result = {k: baseline[k] for k in ('received_caption', 'chunk_generated_token_ids', 'generated_token_ids', 'stopped_by_capacity')}
                    else:
                        result = generate(model.bart, bridge, tokenizer, memory, aux, gate)
                    # Source layout is for reporting, never an RX argument.
                    empty = list(range((entry['source_tokens']-2+61)//62, 9))
                    row = dict(example_id=entry['example_id'], official_split='qual_dev', arm=arm, control=control,
                               condition=channel, source_text=entry['text'], source_text_sha256=entry['text_sha256'],
                               source_tokens=entry['source_tokens'], auxiliary_donor_id=entries[j]['example_id'] if control.startswith('donor') else None,
                               own_observation_sha256=source['sha256'], own_stats=source['stats'],
                               auxiliary_cache_sha256=entries[j][('text' if kind(arm) == 'self' else 'image')+'_file_sha256'],
                               first_free_content_probability=gate[0].tolist(), input_truncated=False,
                               empty_slots=empty, empty_false_activation=sum(bool(body(result['chunk_generated_token_ids'][k])[0]) for k in empty),
                               off_reused=control == 'off', oracle=kind(arm) == 'oracle_clip', **result,
                               **metrics(entry['text'], result['received_caption']))
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); log.flush(); count += 1
                if i == 0 or (i+1) % 10 == 0 or i+1 == len(entries):
                    print('EVAL {} {} {}/{} all{} controls'.format(arm, channel, i+1, len(entries), len(controls(smoke))), flush=True)
    atomic(root/'EVAL_COMPLETE.json', dict(rows=count, facts='PENDING', semantic_success=False))
