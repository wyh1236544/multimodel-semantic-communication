"""Synthetic end-to-end CPU verifier/collector tests; never neural evidence."""
import copy
import json
import math
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
import numpy as np
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_text_aid_v21.core import (PROJECT,PARENTS,EVAL_SHA,CACHE_SHA,FOLDER,config,sha,read,rows,
    order,seed,array_sha,impairment,source_check)
from docci_text_aid_v21.test_core import fixture
from docci_text_aid_v21.report import verify,render
from docci_text_guided_image_v20_fix1.compat import FOLDER as FIX,SCHEMA as FIX_SCHEMA
from docci_text_guided_image_v20.channel import sample

def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf-8')

def write_rows(p,value):
    p.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in value),encoding='utf-8')

def stats(eid,snr,phase,epoch=0,kind='image'):
    value=seed(42,'v20-AWGN',phase,epoch,eid,kind,snr,0)
    n=sample(8192 if kind=='image' else 18432,snr,value)
    p=sample(64,snr,seed(value,'pilot'))
    pn=float(np.mean(n.astype(np.float64)**2));n0=float(np.mean(p.astype(np.float64)**2))
    return dict(noise_seed=value,noise_sha256=array_sha(n),pilot_sha256=array_sha(p),nominal_snr_db=snr,
        noise_power=pn,pilot_noise_variance=n0,signal_power=1.,post_noise_renormalization=False,
        estimated_snr_db=None if snr is None else -10*math.log10(n0),
        realized_snr_db=None if snr is None else -10*math.log10(pn))

def make_fixture(root,action):
    c,data=fixture(action)
    root.mkdir()
    gate=dict(schema=FIX_SCHEMA,repair_manifest_sha256=sha(FIX/'MANIFEST.json'),
        CUDA_gate_passed=True,strict_determinism=True,GPU_tests_run=3,GPU_tests_skipped=0,physical_GPU='1',
        fixture_only=True)
    write(root/'config.json',c)
    write(root/'lineage.json',dict(cache_index_sha256=CACHE_SHA,parents=PARENTS,
        cache_regenerated=False,strict_determinism=True,CUDA_gate=gate))
    write(root/'RUN_FINISHED.json',dict(training_and_evaluation_finished=True))
    for r in data:
        r.update(image_file=None,source_image_file=None,text_stats=None,
            trace={n:dict(alpha=0.,tanh_alpha=0.,feature_RMS=1.,residual_RMS=0.,residual_relative_RMS=0.)
                   for n in ('tx32','tx16','rx32','rx64')})
    for arm in (('self','correct','donor') if action=='pilot' else (None,)):
        folder=root/arm if arm else root
        folder.mkdir(exist_ok=True)
        rr=[r for r in data if r['arm']==arm] if arm else data
        write_rows(folder/'rows.jsonl',rr)
        write(folder/'DONE.json',dict(rows=len(rr),rows_sha256=sha(folder/'rows.jsonl'),frozen_models_unchanged=True))
        for anchor_arm in (PARENTS if action=='diagnose' else ('A_self',)):
            examples=[]
            for entry in c['selected']['dev']:
                a=dict(arm=anchor_arm,example_id=entry['example_id'],psnr=20.,ssim=.5,LPIPS=.3,MSE=.01,
                    tx_sha256='tx',observed_sha256='observed')
                b=dict(a,tx_image_sha256='tx',observed_image_sha256='observed')
                examples.append(dict(example_id=entry['example_id'],actual=a,reference=b))
            write(folder/('ANCHOR_'+anchor_arm+'.json'),dict(reference_rows_sha256=EVAL_SHA[anchor_arm],
                rows=examples,actual_saved_weight_replay=True,exact_TX_and_observation=True))
        if action!='pilot':continue
        initial={'base.image_encoder.input.weight':'frozen','rx32.alpha':'before'}
        final=dict(initial,**{'rx32.alpha':'after'})
        write(folder/'learning_audit.json',dict(initial=initial,final=final,
            frozen_backbones_before=['signature'],frozen_backbones_after=['signature']))
        write(folder/'identity.json',dict(parent_sha256=PARENTS['A_self'],config=c,
            only_RX_fusion_trains=True,trainable_parameters=514018))
        write(folder/'TRAIN_COMPLETE.json',dict(updates=c['updates'],checkpoint_roundtrip=True,
            frozen_image_codec_TX_unchanged=True,frozen_backbones_unchanged=True,
            changed=['rx32.alpha'],nonzero_gradients=['rx32.alpha']))
        history=[]
        for step,(epoch,batch) in enumerate(order(c),1):
            observations=[]
            for own,donor in batch:
                eid=c['selected']['train'][own]['example_id'];did=c['selected']['train'][donor]['example_id']
                snr=None if seed(42,'v21-train-snr',epoch,eid)%4==0 else 0
                observations.append(dict(example_id=eid,donor_id=did,snr_db=snr,
                    image_stats=stats(eid,snr,'v21-pilot-train',epoch),
                    text_stats=stats(did if arm=='donor' else eid,None,'v21-pilot-train',epoch,'text'),
                    tx_sha256='tx'+eid,observed_sha256='rx'+eid,mask_sha256=array_sha(impairment(c,eid)),
                    pixel_MSE=.01,latent_MSE=.01,LPIPS=.3,loss=.0251))
            history.append(dict(update=step,total_updates=c['updates'],epoch=epoch,
                learning_rate=c['learning_rate']*(.1+.9*(1-(step-1)/c['updates'])),
                gradient_norm=.1,loss=.0251,observations=observations))
        write_rows(folder/'history.jsonl',history)
    return c,data

class Reports(unittest.TestCase):
    def test_complete_both_actual_schema_shapes(self):
        with tempfile.TemporaryDirectory() as d:
            for action in ('diagnose','pilot'):
                root=Path(d)/action;make_fixture(root,action)
                out=verify(root)
                self.assertEqual(out['rows'],88 if action=='diagnose' else 52)
                self.assertFalse(out['semantic_success'])
                self.assertIn('V21',render(root,out))
    def test_changed_rows_or_channel_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'diagnose';make_fixture(root,'diagnose')
            r=rows(root/'rows.jsonl');r[0]['image_stats']['noise_sha256']='wrong';write_rows(root/'rows.jsonl',r)
            done=read(root/'DONE.json');done['rows_sha256']=sha(root/'rows.jsonl');write(root/'DONE.json',done)
            with self.assertRaises(ValueError):verify(root)
    def test_freeze_audit_rejects_changed_base(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'pilot';make_fixture(root,'pilot')
            p=root/'correct/learning_audit.json';a=read(p)
            a['final']['base.image_encoder.input.weight']='changed';write(p,a)
            with self.assertRaises(ValueError):verify(root)
    def test_wrong_aggregation_and_learning_rate_rejected(self):
        for field in ('loss','learning_rate','gradient_norm'):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)/'pilot';make_fixture(root,'pilot')
                p=root/'correct/history.jsonl';r=rows(p);r[0][field]=-1 if field!='gradient_norm' else float('inf')
                # Standard JSON can represent external invalid nonfinite input;
                # the independent verifier must reject it, not silently accept it.
                p.write_text(''.join(json.dumps(x)+'\n' for x in r))
                with self.assertRaises(ValueError):verify(root)
    def test_no_false_complete_on_partial(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'pilot';make_fixture(root,'pilot')
            (root/'donor/DONE.json').unlink()
            with self.assertRaises(FileNotFoundError):verify(root)
    def test_actual_parent_anchor_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'diagnose';make_fixture(root,'diagnose')
            p=root/'ANCHOR_A_self.json';a=read(p)
            a['rows'][0]['actual']['LPIPS']+=.001;write(p,a)
            with self.assertRaises(ValueError):verify(root)
    def test_private_collect_excludes_weight_arrays_and_handles_failure(self):
        # Only enabled once the exact new source release has been frozen.
        if not (FOLDER/'MANIFEST.json').exists():self.skipTest('Release manifest not frozen yet')
        from docci_text_aid_v21.collect import collect
        with tempfile.TemporaryDirectory(prefix='v21_collect_test_',dir=str(PROJECT/'experiment_results')) as d:
            root=Path(d)/'run';root.mkdir()
            write(root/'PIPELINE.json',dict(status='FAILED'))
            (root/'last.pth').write_bytes(b'PRIVATE WEIGHT SENTINEL')
            (root/'cache.npz').write_bytes(b'PRIVATE ARRAY SENTINEL')
            dest=Path(d)/'PRIVATE_records.tar.gz';collect(root,dest)
            with tarfile.open(dest) as tf:
                names=tf.getnames()
                self.assertEqual(set(names),{'v21_records/PIPELINE.json','v21_records/RECORDS_MANIFEST.json'})
                self.assertEqual(json.loads(tf.extractfile('v21_records/RECORDS_MANIFEST.json').read())['status'],'FAILED')
            with self.assertRaises(FileExistsError):collect(root,dest)

if __name__=='__main__':unittest.main(verbosity=2)
