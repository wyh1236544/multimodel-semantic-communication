"""Own T0 fixed across real-donor/zero/off auxiliary interventions."""
import json
import torch
from docci_rx_bridge_v16.core import atomic, CONTROLS, geometry, lines
from docci_rx_bridge_v16.cache import batch
from docci_rx_bridge_v16.model import generate
from docci_text_continue_v11.metrics import metrics, body


@torch.no_grad()
def evaluate(bart, bridge, tok, items, arm, folder, device, budget, baseline):
    folder.mkdir(exist_ok=False)
    reference = {r['example_id']: r for r in baseline}
    atomic(folder/'matrix.json', dict(arm=arm, controls=list(CONTROLS), strategies=['greedy'],
           ids=[r['example_id'] for r in items.entries], geometry=geometry(arm),
           source_features_allowed_at_RX=arm == 'oracle_clip',
           auxiliary_source_matches_training=True, own_memory_unchanged=True,
           shuffled_is_real_different_pair=True, zero_keeps_all_attention_slots=True,
           facts='PENDING', BERTScore='NOT_MEASURED', LPIPS='NOT_APPLICABLE',
           primary_decode='unrestricted greedy; no repetition penalty/forbidden ngrams'))
    count = 0
    with (folder/'rows.jsonl').open('x', encoding='utf-8') as log:
        for i, entry in enumerate(items.entries):
            donor = (i+1) % len(items)
            memory, matched_aux, gate, ids = batch(items, [i], arm, device)
            for control in CONTROLS:
                budget()
                aux = matched_aux
                if control == 'shuffled':
                    _, aux, _, _ = batch(items, [donor], arm, device)
                elif control == 'zero':
                    aux = torch.zeros_like(aux)
                result = generate(bart, bridge, memory, aux, gate, tok, 'greedy', control != 'off')
                if control == 'off' and result['chunk_generated_token_ids'] != reference[entry['example_id']]['chunk_generated_token_ids']:
                    raise RuntimeError('Gate-off failed actual V14 baseline token reproduction')
                generated = result['chunk_generated_token_ids']
                # Source chunk types used for evaluation ONLY, never for bridge gating.
                empty = [j for j, row in enumerate(ids[0].tolist()) if row[1] == tok.eos_token_id]
                nonempty = [j for j in range(9) if j not in empty]
                activated = sum(bool(body(generated[j])[0]) for j in empty)
                suppressed = sum(not bool(body(generated[j])[0]) for j in nonempty)
                row = dict(example_id=entry['example_id'], official_split=entry['official_split'],
                           arm=arm, control=control, source_text=entry['text'],
                           source_text_sha256=entry['text_sha256'], source_tokens=entry['source_tokens'],
                           auxiliary_donor_id=items.entries[donor]['example_id'] if control == 'shuffled' else None,
                           own_text_cache_sha256=entry['text_file_sha256'],
                           auxiliary_cache_sha256=items.entries[donor if control == 'shuffled' else i][('text' if arm == 'self' else 'image')+'_file_sha256'],
                           snr_db=None, condition='ideal_clean', oracle=arm == 'oracle_clip',
                           first_free_content_probability=gate[0].tolist(),
                           empty_slots=empty, nonempty_slots=nonempty,
                           empty_false_activation=activated, nonempty_suppressed=suppressed,
                           input_truncated=False, **result,
                           **metrics(entry['text'], result['received_caption']))
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                log.flush()
                count += 1
            if i == 0 or (i+1) % 10 == 0 or i+1 == len(items):
                print('EVAL {} {} {}/{} (matched/donor/zero/off)'.format(arm, entry['official_split'], i+1, len(items)), flush=True)
    atomic(folder/'EVAL_COMPLETE.json', dict(rows=count, gate_off_token_equality=True,
                                           facts='PENDING', no_semantic_success_claim=True))
