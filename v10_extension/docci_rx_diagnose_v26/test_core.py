"""Non-neural CPU contracts. Synthetic fixtures are explicitly TEST_ONLY."""
import ast
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_rx_diagnose_v26 import core as c,report as r
from docci_text_continue_v11.metrics import metrics
from docci_caption_calibration_v24.core import text_audit

def labels():
    ids=np.ones((9,64),np.int64);ids[:,0]=0;ids[:,1]=2;ids[0,:3]=[0,4,2]
    return ids.tolist()

def reference(i):
    text='TEST_ONLY_object'+str(i)
    return dict(TEST_ONLY=True,example_id='TEST_ONLY_'+str(i),cohort='train' if i<512 else 'dev',
        official_split='train',text=text,text_sha256=c.digest(text),source_tokens=3,
        memory_slice_sha256=[c.digest(('TEST_ONLY_memory',i,s)) for s in c.CHANNELS],
        visual_slice_sha256=[c.digest(('TEST_ONLY_visual',i,s)) for s in c.CHANNELS])

def row(origin,regime,arm,ctrl,all_refs,donors,gray):
    it,tt=[c.CHANNELS.index(s) for s in c.REGIMES[regime]]
    donor=donors[origin['example_id']] if ctrl=='shuffled' else None
    aux=None if arm=='base' else c.digest(('TEST_ONLY_self',origin['example_id'])) if arm=='self' else (
        all_refs[donor]['visual_slice_sha256'][it] if donor else gray if ctrl=='gray' else
        c.array_sha(np.zeros((1,32,768),np.float32)) if ctrl=='zero' else origin['visual_slice_sha256'][it])
    own=dict(TEST_ONLY=True,example_id=origin['example_id'],cohort=origin['cohort'],official_split='train',
        source_text=origin['text'],source_text_sha256=origin['text_sha256'],source_tokens=3,
        regime=regime,arm=arm,control=ctrl,donor_id=donor,memory_sha256=origin['memory_slice_sha256'][tt],
        auxiliary_sha256=aux,gate_sha256=c.array_sha(np.ones((1,9),np.float32)),
        image_snr_db=c.REGIMES[regime][0],text_snr_db=c.REGIMES[regime][1],training_updates=0)
    slots=[dict(slot=j,counts=counts,nll={k:float(v) for k,v in counts.items()},content_correct=counts['content'])
        for j,counts in enumerate(c.reference_counts(labels()))]
    counts={k:sum(s['counts'][k] for s in slots) for k in slots[0]['counts']}
    enabled=arm!='base' and ctrl!='disabled';layers=[]
    if enabled:
        for i in range(2):
            scopes={}
            for scope in ('all','content'):
                scopes[scope]=[dict(positions=s['counts'][scope],hidden_l2_sq=4. if s['counts'][scope] else 0.,
                    delta_l2_sq=.04 if s['counts'][scope] else 0.,effective_ratio=.1 if s['counts'][scope] else None) for s in slots]
            layers.append(dict(layer=i,alpha=.1,tanh_alpha=math.tanh(.1),received_content_gates=[1.]*9,scopes=scopes))
    tf=dict(own,slots=slots,counts=counts,nll={k:float(v) for k,v in counts.items()},TF_all_CE=1.,TF_content_CE=1.,TF_content_accuracy=1.,
        trace=dict(enabled=enabled,layers=layers),teacher_forced_gold_prefix=True,gold_masks_used_for_measurement_only=True,source_mask_at_receiver=False)
    gen=dict(own,received_caption=origin['text'],generated_token_ids=[4],chunk_generated_token_ids=[[2,0,4,2]]+[[2,0,2]]*8,
        stopped_by_capacity=False,**metrics(origin['text'],origin['text']),output_audit=text_audit(origin['text']),
        generation_provenance='V25_VERIFIED_DEV_REUSE' if origin['cohort']=='dev' else 'V26_NEW_TRAIN_GREEDY',
        source_text_at_RX=False,source_mask_at_RX=False,source_image_at_RX=False)
    return tf,gen

def fixture(base):
    """All receipts/TF values are TEST_ONLY, NOT actual saved neural checkpoints."""
    root=base/'pipeline';root.mkdir();inputs=root/'inputs';inputs.mkdir()
    all_entries=[reference(i) for i in range(544)];index=dict(TEST_ONLY=True,records=all_entries)
    c.atomic(inputs/'PARENT_INDEX.json',index)
    refs={r['example_id']:r for r in all_entries};gray='TEST_ONLY_GRAY'
    old=[]
    selected,donors=c.selected(index,'diagnose')
    for o in selected['dev']:
        for regime in c.config('diagnose')['regimes']:
            for arm,ctrl in c.CONTROLS:old.append(row(o,regime,arm,ctrl,refs,donors,gray)[1])
    with (inputs/'V25_EVALUATION_ROWS.jsonl').open('x') as f:
        for x in old:c.line(f,x)
    c.PINS['cache/index.json']=c.sha(inputs/'PARENT_INDEX.json')
    c.PINS['evaluation/rows.jsonl']=c.sha(inputs/'V25_EVALUATION_ROWS.jsonl')
    c.snapshot(inputs);c.atomic(inputs/'RUN.json',dict(TEST_ONLY=True,schema=c.SCHEMA,stage='preflight',training_updates=0))
    c.atomic(inputs/'PARENTS.json',dict(TEST_ONLY=True,pins=c.PINS,heavy_arrays_and_weights_verified=True,
        launch_checked=True,historical_WER_and_counts_exact=True,old_code_and_results_modified=False))
    c.atomic(inputs/'PARENT_LAUNCH.json',dict(TEST_ONLY=True,status='VERIFIED'))
    with (inputs/'CAPTION_ONLY.jsonl').open('x') as f:
        for o in selected['dev']:c.line(f,dict(TEST_ONLY=True,example_id=o['example_id'],image_snr_db=0,output_text=o['text']))
    c.atomic(inputs/'PARENT_GRAY.json',dict(TEST_ONLY=True,visual_slice_sha256=gray))
    c.atomic(inputs/'V25_CACHE_COMPLETE.json',dict(TEST_ONLY=True,files={
        'caption_only.jsonl':dict(sha256=c.sha(inputs/'CAPTION_ONLY.jsonl')),
        'gray.json':dict(sha256=c.sha(inputs/'PARENT_GRAY.json'))}))
    c.PINS['cache/COMPLETE.json']=c.sha(inputs/'V25_CACHE_COMPLETE.json')
    # Refresh this TEST_ONLY proof after its synthetic receipt has been finalized.
    proof=inputs/'PARENTS.json';proof.unlink()
    c.atomic(proof,dict(TEST_ONLY=True,pins=c.PINS,heavy_arrays_and_weights_verified=True,
        launch_checked=True,historical_WER_and_counts_exact=True,old_code_and_results_modified=False))
    c.seal(inputs)
    for profile in ('smoke','diagnose'):
        path=root/profile;path.mkdir();cfg=c.config(profile);choices,ds=c.selected(index,profile)
        chosen=[o for cohort in choices.values() for o in cohort];records=[]
        c.snapshot(path);c.atomic(path/'RUN.json',dict(TEST_ONLY=True,schema=c.SCHEMA,stage=profile,training_updates=0))
        c.atomic(path/'config.json',cfg)
        for o in chosen:records.append(dict(o,gold_ids=labels(),received_gates=[[1.]*9]*3))
        c.atomic(path/'REFERENCES.json',dict(TEST_ONLY=True,records=records,donors=ds,parent_index_sha256=c.PINS['cache/index.json'],gray_visual_slice_sha256=gray))
        c.atomic(path/'CHECKPOINTS.json',dict(TEST_ONLY=True,parent_pins=c.PINS,trainable_parameters=0,
            BLIP_loaded=False,VAE_loaded=False,source_codec_loaded=False,BART_before='TEST_ONLY',
            adapters={arm:dict(parameters=1516290,tensor_sha256='TEST_ONLY_'+arm,layers=[dict(layer=i,alpha=.1,tanh_alpha=math.tanh(.1)) for i in range(2)]) for arm in ('self','visual')}))
        with (path/'tf_rows.jsonl').open('x') as tf,(path/'generations.jsonl').open('x') as gf:
            for o in chosen:
                for regime in cfg['regimes']:
                    for arm,ctrl in c.CONTROLS:
                        a,b=row(o,regime,arm,ctrl,refs,ds,gray);c.line(tf,a);c.line(gf,b)
        anchors=[dict(example_id=o['example_id'],arm=arm,control=ctrl,tokens_equal=True) for o in choices['dev'][:2]
            for arm,ctrl in (('base','matched'),('self','matched'),('visual','matched'),('visual','shuffled'))]
        c.atomic(path/'ANCHORS.json',dict(TEST_ONLY=True,passed=True,actual_anchors=anchors))
        nt=cfg['train']*len(cfg['regimes'])*7;nd=cfg['dev']*len(cfg['regimes'])*7
        c.atomic(path/'RESULT.json',dict(TEST_ONLY=True,TF_calls=nt+nd,new_TRAIN_greedy=nt,reused_DEV_greedy=nd,
            training_updates=0,optimizer_created=False,BART_before='TEST_ONLY',BART_after='TEST_ONLY',adapter_tensors_unchanged=True,
            adapter_before={arm:'TEST_ONLY_'+arm for arm in ('self','visual')},adapter_after={arm:'TEST_ONLY_'+arm for arm in ('self','visual')}))
        c.seal(path)
    c.atomic(root/'PIPELINE.json',dict(TEST_ONLY=True,schema=c.SCHEMA,status='COMPLETE',source_manifest_sha256=c.sha(c.FOLDER/'MANIFEST.json')))
    return root

class Contracts(unittest.TestCase):
    def test_fixed_no_training_profiles_and_complete_matrix(self):
        self.assertEqual(c.config('diagnose')['training_updates'],0)
        self.assertEqual((c.config('diagnose')['train'],c.config('diagnose')['dev']),(16,32))
        self.assertEqual(len(c.matrix([{'example_id':str(i)} for i in range(48)],c.config('diagnose'))),1680)
        with self.assertRaises(ValueError):c.config('train')
    def test_selection_is_outcome_independent_and_DEV_donors_preserved(self):
        index=dict(records=[reference(i) for i in range(544)])
        a,da=c.selected(index,'diagnose');b,db=c.selected(index,'smoke')
        self.assertEqual(a['train'][:2],b['train']);self.assertEqual(a['dev'][:2],b['dev']);self.assertEqual(da,db)
        self.assertTrue(all(k!=v for k,v in da.items()))
        self.assertEqual([r['example_id'] for r in a['dev']],[r['example_id'] for r in index['records'][512:]])
    def test_public_gold_counts_and_invalid_PAD_rejected(self):
        v=c.reference_counts(labels());self.assertEqual(sum(s['all'] for s in v),19);self.assertEqual(v[0]['content'],1)
        bad=labels();bad[0][4]=5
        with self.assertRaises(ValueError):c.reference_counts(bad)
    def test_loss_verifier_rejects_wrong_counts_and_aggregation(self):
        o=reference(0);tf,_=row(o,'both0','base','matched',{o['example_id']:o},{},'gray');o['gold_ids']=labels()
        r.losses(tf,o);tf['TF_content_CE']=.5
        with self.assertRaisesRegex(ValueError,'Content CE'):r.losses(tf,o)
    def test_actual_residual_ratio_gate_and_empty_scope_checks(self):
        o=dict(reference(0),gold_ids=labels(),received_gates=[[1.]*9]*3)
        tf,_=row(o,'both0','visual','matched',{o['example_id']:o},{},'gray')
        ck=dict(adapters={'visual':dict(layers=[dict(alpha=.1,tanh_alpha=math.tanh(.1))]*2)})
        r.trace(tf,o,ck);tf['trace']['layers'][0]['scopes']['content'][0]['effective_ratio']=.2
        with self.assertRaisesRegex(ValueError,'ratio'):r.trace(tf,o,ck)
    def test_zero_and_gray_are_different_and_disabled_has_no_trace(self):
        o=reference(0);ref={o['example_id']:o}
        a,_=row(o,'both0','visual','zero',ref,{},'gray');b,_=row(o,'both0','visual','gray',ref,{},'gray')
        self.assertNotEqual(a['auxiliary_sha256'],b['auxiliary_sha256'])
        d,_=row(o,'both0','visual','disabled',ref,{},'gray');self.assertFalse(d['trace']['enabled'])
    def test_known_BLEU_roundoff_only_not_WER_accepted(self):
        from docci_rx_diagnose_v26.parents import lexical
        x=dict(source_text='a cat sits',received_caption='a cat',**metrics('a cat sits','a cat'),output_audit=text_audit('a cat'))
        x['bleu2']=float(np.nextafter(x['bleu2'],math.inf));self.assertEqual(len(lexical(x)),1)
        x['normalized_wer']+=.01
        with self.assertRaisesRegex(ValueError,'normalized_wer'):lexical(x)
    def test_strict_output_paths_and_no_nonfinite_or_overwrite(self):
        with tempfile.TemporaryDirectory(dir=c.PROJECT/'experiment_results') as t:
            p=Path(t)/'value.json'
            with self.assertRaises(ValueError):c.atomic(p,dict(v=float('nan')))
            self.assertFalse(p.exists());c.atomic(p,dict(v=1))
            with self.assertRaises(FileExistsError):c.atomic(p,{})
            link=Path(t)/'linked';link.symlink_to(Path(t),target_is_directory=True)
            with self.assertRaises(ValueError):c.safe(link/'out')
        with self.assertRaises(ValueError):c.safe(c.PROJECT)
    def test_source_has_no_training_download_or_gold_in_generate(self):
        tree=ast.parse((c.FOLDER/'neural.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,('backward','step','save','from_pretrained'))
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='generate':
                self.assertTrue(all(k.arg not in ('ids','labels','source_mask','text','length') for k in node.keywords))
        source=(c.FOLDER/'test_hf.py').read_text();self.assertNotIn('skipTest',source);self.assertNotIn('skipIf',source)
        c.source_check()
    def test_full_TEST_ONLY_producer_verifier_PRIVATE_collector(self):
        from docci_rx_diagnose_v26.collect import collect
        with tempfile.TemporaryDirectory(prefix='V26_TEST_ONLY_',dir=c.PROJECT/'experiment_results') as t,patch.dict(c.PINS,dict(c.PINS)):
            root=fixture(Path(t));before=c.sha(root/'inputs/PARENT_INDEX.json')
            result=r.pipeline(root);self.assertEqual(result['stages']['diagnose']['rows'],1680)
            # A forged receipt cannot hide a changed visual intervention or reused DEV output.
            def replace_fixture(path,value):
                path.write_text('\n'.join(json.dumps(x) for x in value)+'\n')
                (path.parent/'COMPLETE.json').unlink();c.seal(path.parent)
            path=root/'smoke/tf_rows.jsonl';valid=c.rows(path);bad=copy.deepcopy(valid)
            next(x for x in bad if x['control']=='shuffled')['auxiliary_sha256']='changed'
            replace_fixture(path,bad)
            with self.assertRaisesRegex(ValueError,'Visual perturbation'):r.verify(root/'smoke',root/'inputs')
            replace_fixture(path,valid)
            replace_fixture(path,valid[:-1])
            with self.assertRaisesRegex(ValueError,'matrix'):r.verify(root/'smoke',root/'inputs')
            replace_fixture(path,valid)
            path=root/'smoke/generations.jsonl';valid=c.rows(path);bad=copy.deepcopy(valid)
            next(x for x in bad if x['cohort']=='dev' and x['arm']=='visual')['received_caption']='changed'
            replace_fixture(path,bad)
            with self.assertRaisesRegex(ValueError,'Reused V25'):r.verify(root/'smoke',root/'inputs')
            replace_fixture(path,valid)
            self.assertFalse(result['semantic_success']);dest=Path(t)/'PRIVATE_TEST_ONLY.tar.gz'
            collect(root,dest);self.assertEqual(before,c.sha(root/'inputs/PARENT_INDEX.json'))
            staging=dest.with_name(dest.name+'.contents')
            self.assertFalse(list(staging.rglob('*.pth')));self.assertFalse(list(staging.rglob('*.npz')))
            self.assertEqual(len(c.read(staging/'FACT_KEY_PRIVATE.json')['items']),160)
            self.assertTrue(all(x['facts']==[] for x in c.read(staging/'FACT_REFERENCES.json')['items']))
            self.assertEqual(r.pipeline(staging/'data')['training_updates'],0)
            # Post-completion corruption is rejected, not silently rehashed.
            p=root/'smoke/tf_rows.jsonl'
            with p.open('a') as f:f.write('{}\n')
            with self.assertRaisesRegex(ValueError,'artifact'):r.verify(root/'smoke',root/'inputs')
    def test_gallery_escapes_raw_output_and_never_claims_semantic_success(self):
        result=dict(results=dict(groups=[dict(cohort='<script>',regime='both0',arm='visual',control='matched',
            token_weighted_content_CE=1.,mean_greedy_WER=.5,mean_content_residual_ratio=.1)]),semantic_success=False)
        page=r.render(result);self.assertNotIn('<script>',page);self.assertIn('&lt;script&gt;',page)

if __name__=='__main__':unittest.main(verbosity=2)
