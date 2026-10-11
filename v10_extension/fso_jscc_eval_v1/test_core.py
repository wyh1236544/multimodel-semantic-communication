"""CPU/NumPy contracts with explicitly synthetic provenance, not model evidence."""
import ast
import copy
import json
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent
sys.path.insert(0,str(EXT))
from fso_jscc_v1 import ARMS, VERSION as PARENT_VERSION
from fso_jscc_eval_v1 import VERSION
from fso_jscc_eval_v1.core import load, settings, validate, proxy_conditions, expected_counts, matrix, select_ids, contrast, image_stats, replay_check, digest, check_delta_metadata
from fso_jscc_eval_v1.report import verify, write_report
from fso_jscc_eval_v1.collect import collect


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')


def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows),encoding='utf-8')


def fixture(root):
    """Small synthetic frozen-weight record, never presented as a real run."""
    root.mkdir(); ev=root/'source_evidence'; ev.mkdir()
    split=dict(train=[900],validation=list(range(1,35)),calibration=[901],test=[902],excluded_history=[])
    write_json(root/'split_evidence.json',split)
    c=load(); c.update(samples=34,proxy_snrs_db=[-5,0],proxy_draws=1,physical_draws=1,save_images=0,bootstrap_resamples=100)
    conditions=[dict(id=n,case='synthetic',launch_signal_dbm=-24.) for n in c['physical_conditions']]
    plan=dict(conditions=conditions,analog_repeats=1); cases=dict(synthetic=dict(note='synthetic only'))
    resource=dict(analog_samples=10)
    source=dict(version=PARENT_VERSION,smoke=False,config=dict(val_samples=32,draws=1,physical_val_samples=8,physical_draws=1),train_ids=[900],validation_ids=list(range(1,33)),
                proxy_conditions=proxy_conditions(c),physical_conditions=conditions,
                train_forward_steps=2,optimizer_steps_per_trained_arm=1,physical_plan=plan,physical_cases=cases,physical_resource=resource)
    write_json(ev/'design.json',source)
    identity=dict(status='complete',checkpoint_sha256='base-synthetic',a_delta_sha256='A-synthetic',split_sha256=digest(root/'split_evidence.json'))
    write_json(ev/'run_identity.json',identity); write_json(ev/'checkpoint_load.json',dict(synthetic=True))
    weights=dict(base=identity['checkpoint_sha256'],A=identity['a_delta_sha256'])
    checkpoint_hashes={}
    for a in ARMS:
        r=dict(arm=a,trained=a!=ARMS[0],steps=0 if a==ARMS[0] else 2,optimizer_steps=0 if a==ARMS[0] else 1)
        if a!=ARMS[0]:
            r.update(nonzero_gradient_steps=dict(encoder=1,decoder=2,tx_mapping=2,rx_mapping=2),parameter_update_squared_l2_by_group=dict(encoder=1.,decoder=1.,tx_mapping=1.,rx_mapping=1.),checkpoint_sha256=a+'-synthetic')
            weights[a]=r['checkpoint_sha256']; checkpoint_hashes['train_'+a+'/last_delta.pth']=weights[a]
            write_rows(ev/('train_'+a)/'history.jsonl',[dict(forward_step=s,optimizer_step=1 if s==2 else 0,epoch=1,image_id=900,nominal_snr_db=0.) for s in (1,2)])
        write_json(ev/('train_'+a)/'COMPLETE.json',r)
    done=dict(checkpoint_sha256=checkpoint_hashes,result_sha256={str(f.relative_to(ev)):digest(f) for f in ev.rglob('*') if f.is_file()})
    write_json(ev/'COMPLETE.json',done)
    write_json(root/'input_receipt.json',dict(source_copy_sha256={str(f.relative_to(ev)):digest(f) for f in ev.rglob('*') if f.is_file()},weights_before=weights,weights_after=weights,parent_verified_at_start_and_end=True))
    d=dict(version=VERSION,arms=list(ARMS),smoke=False,split='validation',training_performed=False,config=c,validation_ids=split['validation'],pilot_ids=list(range(1,33)),new_ids=[33,34],
           proxy_conditions=proxy_conditions(c),physical_conditions=conditions,physical_plan=plan,physical_cases=cases,physical_resource=resource,
           validation_source_sha256={str(i):'synthetic-source-'+str(i) for i in split['validation']},expected_counts=expected_counts(c))
    write_json(root/'design.json',d)
    loads=dict(base_exact=True,a_exact=True,base_sha256=weights['base'],a_delta_sha256=weights['A'],arms={a:dict(exact=True,all_parameters_frozen=True,state_before=a+'-state',state_after=a+'-state',checkpoint_sha256=weights.get(a)) for a in ARMS})
    write_json(root/'checkpoint_load.json',loads)
    rows=[]; events=[]
    for a,i,e,n,draw in sorted(matrix(d)):
        unavailable=(i==34 and n=='moderate_p24')
        r=dict(arm=a,image_id=i,engine=e,condition=n,draw=draw,image_subset='pilot' if i<=32 else 'new',source_sha256=d['validation_source_sha256'][str(i)],field_seed=i,noise_seed=i+100,
            psnr=5. if unavailable else 20.+ARMS.index(a)/10,ssim=.5,normalized_wer=1.2 if unavailable else .8,token_acc=.4,bleu2=.1,cer=.7,receiver_available=not unavailable,
            diagnostic_joint_success=False,analog_repeats=1,raw_utf8_wire_bytes=0,pre_fec_ber=None,post_fec_ber=None,received_caption='<received>',source_caption_evaluation_only='<source>')
        if e=='proxy': r.update(payload_complex_symbols=2304,forward_complex_symbols=2560)
        else:
            r.update(event_count=2,used_samples=10,slot_samples=32,launch_signal_energy_j=1.,signal_energy_budget_j=2.,received_optical_signal_dbm=-30.,physical_symbol_snr_db=-7.,estimated_snr_db=-6.)
            events.extend(dict(arm=a,image_id=i,engine=e,condition=n,draw=draw,purpose=p,awg_samples=5,guard_samples=0) for p in ('probe','features'))
        rows.append(r)
    write_rows(root/'samples.jsonl',rows); write_rows(root/'events.jsonl',events)
    originals=[r for r in rows if r['image_id']<=32 and (r['engine']=='proxy' or r['image_id']<=8)]
    write_json(root/'replay.json',replay_check(rows,originals)); write_json(root/'timing.json',dict(synthetic=True))
    return d,rows


def complete(root):
    write_json(root/'run_identity.json',dict(status='complete',synthetic=True))
    write_json(root/'COMPLETE.json',dict(version=VERSION,rows=len((root/'samples.jsonl').read_text().splitlines()),training_performed=False,
        result_sha256={str(f.relative_to(root)):digest(f) for f in root.rglob('*') if f.is_file() and f!=root/'COMPLETE.json'}))


class Contracts(unittest.TestCase):
    def setUp(self):
        scratch=PROJECT/'.tmp'; scratch.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(prefix='expanded_eval_contract_',dir=str(scratch)); self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def test_full_and_smoke_matrix(self):
        self.assertEqual(expected_counts(load()),dict(proxy=25088,waveform_v3=6144))
        self.assertEqual(expected_counts(settings(load(),True)),dict(proxy=104,waveform_v3=24))
        d,rows=fixture(self.root/'run'); verify(self.root/'run')
        self.assertEqual(len(rows),sum(expected_counts(d['config']).values()))
    def test_config_and_validation_prefix(self):
        c=load(); split=dict(validation=list(range(128))); parent=dict(validation_ids=list(range(32)),train_ids=[900])
        ids,old,new=select_ids(split,parent,c)
        self.assertEqual((len(ids),len(old),len(new)),(128,32,96))
        for patch in (dict(samples=32),dict(primary_physical_conditions=['unknown']),dict(proxy_snrs_db=[float('nan')]),dict(normalized_wer_target=True)):
            bad=dict(c,**patch)
            with self.assertRaises(ValueError):
                validate(bad); select_ids(split,parent,bad)
        with self.assertRaises(ValueError): select_ids(split,dict(parent,train_ids=[100]),c)
        with self.assertRaises(ValueError): select_ids(split,dict(parent,validation_ids=[3,2,1]),c)
    def test_image_cluster_not_draw_pseudoreplication(self):
        rows=[]
        for i,g in ((1,1.),(2,3.)):
            for draw in range(4):
                for a,v in ((ARMS[2],10.),(ARMS[3],10.+g)):
                    rows.append(dict(arm=a,image_id=i,engine='proxy',condition='fixed',draw=draw,psnr=v))
        x=contrast(rows,resamples=1000); self.assertEqual(x['images'],2); self.assertEqual(x['mean'],2.)
        self.assertEqual(x,contrast(rows,resamples=1000)); self.assertEqual(image_stats(rows,'psnr')['images'],2)
        with self.assertRaises(ValueError): contrast([r for r in rows if r['arm']==ARMS[3]])
    def test_original_replay_rejects_metric_and_channel_changes(self):
        _,rows=fixture(self.root/'run'); self.assertEqual(replay_check(rows,rows)['matched_rows'],len(rows))
        for k,v in [('psnr',100.),('field_seed',9999),('receiver_available',False)]:
            bad=copy.deepcopy(rows[:1]); bad[0][k]=v
            with self.assertRaises(ValueError): replay_check(bad,rows)
    def test_wrong_delta_metadata(self):
        source=dict(config=dict(a=1)); record=dict(arm=ARMS[3],steps=2,optimizer_steps=1)
        payload=dict(version=PARENT_VERSION,arm=ARMS[3],steps=2,optimizer_steps=1,config=source['config'],base_checkpoint_sha256='b',a_delta_sha256='a',split_sha256='s')
        check_delta_metadata(payload,record,source,'b','a','s')
        with self.assertRaises(ValueError): check_delta_metadata(dict(payload,arm=ARMS[2]),record,source,'b','a','s')
    def test_independent_negative_audits(self):
        good=self.root/'good'; d,rows=fixture(good)
        for name,change in [('missing',lambda r:r[:-1]),('duplicate',lambda r:r+[r[0]]),('repeat',lambda r:[dict(r[0],analog_repeats=2)]+r[1:]),('availability',lambda r:[dict(r[0],diagnostic_joint_success=True)]+r[1:]),('bad_source',lambda r:[dict(r[0],source_sha256='bad')]+r[1:]),('unpaired',lambda r:[dict(r[0],noise_seed=999)]+r[1:]),('wrong_subset',lambda r:[dict(r[0],image_subset='wrong')]+r[1:])]:
            with self.subTest(name=name):
                root=self.root/name; shutil.copytree(good,root); write_rows(root/'samples.jsonl',change(copy.deepcopy(rows)))
                with self.assertRaises(ValueError): verify(root)
        receipt=json.loads((good/'input_receipt.json').read_text()); receipt['weights_after']=dict(receipt['weights_before'],base='changed'); write_json(good/'input_receipt.json',receipt)
        with self.assertRaises(ValueError): verify(good)
    def test_resources_state_replay_and_parent_evidence_rejected(self):
        good=self.root/'good'; _,rows=fixture(good)
        for kind in ('resource','energy','state','short_replay','source_hash','unsafe_path','fictitious_ber'):
            with self.subTest(kind=kind):
                root=self.root/kind; shutil.copytree(good,root)
                if kind in ('resource','energy','fictitious_ber'):
                    bad=copy.deepcopy(rows)
                    if kind=='fictitious_ber': bad[0]['pre_fec_ber']=0.
                    else:
                        r=next(r for r in bad if r['engine']=='waveform_v3')
                        r['used_samples' if kind=='resource' else 'launch_signal_energy_j']=999
                    write_rows(root/'samples.jsonl',bad)
                elif kind=='state':
                    r=json.loads((root/'checkpoint_load.json').read_text()); r['arms'][ARMS[3]]['state_after']='changed'; write_json(root/'checkpoint_load.json',r)
                elif kind=='short_replay':
                    r=json.loads((root/'replay.json').read_text()); r['matched_rows']=1; write_json(root/'replay.json',r)
                elif kind=='source_hash':
                    with (root/'source_evidence/checkpoint_load.json').open('a') as f: f.write('\n')
                else:
                    r=json.loads((root/'input_receipt.json').read_text()); r['source_copy_sha256']['../outside']='unsafe'; write_json(root/'input_receipt.json',r)
                with self.assertRaises(ValueError): verify(root)
    def test_report_archive_and_failure_retention(self):
        root=self.root/'run'; d,rows=fixture(root); summary=write_report(root)
        primary=[r for r in summary['contrasts'] if r['predeclared_primary']]
        self.assertEqual(len(primary),1); self.assertEqual(primary[0]['images'],2)
        entry=next(g for g in summary['groups'] if g['subset']=='new' and g['condition']=='moderate_p24' and g['arm']==ARMS[3])
        self.assertEqual(entry['receiver_unavailable_fraction'],.5)
        self.assertLess(entry['psnr']['mean'],20.); self.assertIn('NOT strict equal actual energy','; '.join(summary['limitations']))
        complete(root); verify(root,complete=True)
        output=self.root/'records.tar.gz'; collect(root,output)
        with tarfile.open(output,'r:gz') as tar:
            self.assertTrue(all(m.isfile() for m in tar.getmembers()))
            self.assertTrue(any(m.name.endswith('source_evidence/COMPLETE.json') for m in tar.getmembers()))
            self.assertFalse(any(m.name.endswith(('.pth','.pt')) for m in tar.getmembers()))
        with self.assertRaises(FileExistsError): collect(root,output)
        with (root/'samples.jsonl').open('a') as f: f.write('\n')
        with self.assertRaises(ValueError): verify(root,complete=True)
    def test_runtime_has_no_training_or_repetition_combining(self):
        text=(Path(__file__).with_name('run.py')).read_text(); tree=ast.parse(text)
        attrs=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)]
        self.assertNotIn('backward',attrs); self.assertNotIn('optim',attrs); self.assertIn('no_grad',attrs)
        self.assertNotIn('train(',text); self.assertIn('state_fingerprint',text)


if __name__=='__main__': unittest.main(verbosity=2)
