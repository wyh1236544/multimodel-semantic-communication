"""Synthetic evidence fixtures are TEST_ONLY, not trained-model results."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from docci_text_guided_image_v20 import core
from docci_text_guided_image_v20 import report
from docci_text_guided_image_v20.channel import sample

def write_rows(p,data):
    with p.open('w',encoding='utf-8') as f:
        for r in data:
            f.write(json.dumps(r)+'\n')

def stats(eid,snr,kind,repeat=0):
    n = 8192 if kind=='image' else 18432
    value = core.seed(42,'v20-AWGN','eval',0,eid,kind,snr,repeat)
    epsilon = sample(n,snr,value)
    pilot = sample(64,snr,core.seed(value,'pilot'))
    pn = float(np.mean(epsilon.astype(np.float64)**2))
    n0 = float(np.mean(pilot.astype(np.float64)**2))
    return dict(nominal_snr_db=snr,noise_seed=value,noise_sha256=core.array_sha(epsilon),
        pilot_sha256=core.array_sha(pilot),pilot_noise_variance=n0,noise_power=pn,
        signal_power=1.,post_noise_renormalization=False,
        estimated_snr_db=None if snr is None else -10*math.log10(n0),
        realized_snr_db=None if snr is None else -10*math.log10(pn))

def fixture(root):
    c = core.load(smoke=True)
    cache = root/'smoke_cache'
    cache.mkdir()
    records = []
    for cohort,count in (('train',8),('dev',2),('qual_dev',2)):
        for i in range(count):
            eid = cohort+'_'+str(i)
            text = 'TEST_ONLY '+eid
            records.append(dict(example_id=eid,cohort=cohort,
                official_split='qual_dev' if cohort=='qual_dev' else 'train',text=text,
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                image_sha256=hashlib.sha256(eid.encode()).hexdigest(),input_truncated=False))
    core.atomic(cache/'index.json',dict(records=records))
    protected = {k:[r['example_id'] for r in records if r['cohort']==k] for k in ('train','dev','qual_dev')}
    protected['heldout'] = protected['dev']
    core.atomic(cache/'protected_split.json',protected)
    core.atomic(cache/'CACHE_COMPLETE.json',dict(index_sha256=core.sha(cache/'index.json'),
        source_sha256=core.DOCCI_SHA,parents=dict(image=dict(sha256=core.IMAGE_SHA),
        text=dict(sha256=core.TEXT_SHA))))
    core.atomic(root/'config.json',core.load())
    stages = {'smoke_cache':'VERIFIED'}
    for arm in core.ARMS:
        folder = root/('smoke_train_'+arm)
        folder.mkdir()
        stages[folder.name] = 'VERIFIED'
        train = [r for r in records if r['cohort']=='train']
        core.atomic(folder/'identity.json',dict(arm=arm,smoke=True,updates=2,geometry=core.GEOMETRY,
            initial_hashes=dict(parameter='TEST_ONLY'),trainable_parameters=4,
            cache_index_sha256=core.sha(cache/'index.json'),
            train_ids=[r['example_id'] for r in train]))
        history = []
        for step,(epoch,indices) in enumerate(core.batches(len(train),c),1):
            if step>2:
                break
            observations = [dict(example_id=train[i]['example_id'],snr_db=core.train_snr(c,epoch,train[i]['example_id']),
                image=dict(signal_power=1),text=dict(signal_power=1),pixel_MSE=.1,latent_MSE=.2,LPIPS=.3) for i in indices]
            history.append(dict(update=step,total_updates=2,epoch=epoch+1,observations=observations,
                loss=.1+c['latent_weight']*.2+c['lpips_weight']*.3,gradient_norm=.2,learning_rate=.00001))
        write_rows(folder/'history.jsonl',history)
        core.atomic(folder/'TRAIN_COMPLETE.json',dict(updates=2,fixed_final=True,actual_disk_roundtrip=True,
            unique_ids_seen=len({r['example_id'] for h in history for r in h['observations']}),
            checkpoint_sha256='TEST_ONLY'))
        core.atomic(folder/'learning_audit.json',dict(passed=True,groups={g:dict(parameters=1,
            nonzero_gradient=True,changed=True) for g in ('image_encoder','image_decoder','tx_fusion','rx_fusion')},
            frozen_before=['TEST_ONLY'],frozen_after=['TEST_ONLY']))
        ev = root/('smoke_eval_'+arm)
        ev.mkdir()
        stages[ev.name] = 'VERIFIED'
        core.atomic(ev/'identity.json',dict(arm=arm,geometry=core.GEOMETRY,checkpoint_sha256='TEST_ONLY',
            cache_index_sha256=core.sha(cache/'index.json')))
        data = []
        text_data = []
        for cohort in ('dev','qual_dev'):
            items = [r for r in records if r['cohort']==cohort]
            for snr,repeat,control in core.trial_matrix(c,arm,True):
                for i,source in enumerate(items):
                    donor = items[(i+1)%len(items)]['example_id'] if control=='rx_shuffled' else None
                    mse = .01
                    data.append(dict(example_id=source['example_id'],cohort=cohort,
                        official_split=source['official_split'],arm=arm,snr_db=snr,repeat=repeat,control=control,
                        donor_id=donor,psnr=20.,ssim=.5,LPIPS=.2,MSE=mse,
                        source_image_sha256=source['image_sha256'],source_text_sha256=source['text_sha256'],
                        source_text_at_RX=control=='rx_oracle',image_file=None,source_image_file=None,
                        tx_image_sha256='TEST_ONLY_FIXED_TX',observed_image_sha256='TEST_ONLY_FIXED_OBS',
                        image_stats=stats(source['example_id'],snr,'image',repeat),
                        text_stats=stats(source['example_id'],snr,'text',repeat),
                        delivered_text_stats=stats(donor or source['example_id'],snr,'text',repeat)))
                    if arm=='C_tx_rx_text' and control=='matched':
                        from docci_reconstruction_v10.core import joined_ids
                        from docci_text_continue_v11.metrics import metrics
                        chunks = [[2,0,2] for _ in range(9)]
                        joined,capacity = joined_ids(chunks,0,2,1)
                        text_data.append(dict(example_id=source['example_id'],cohort=cohort,snr_db=snr,
                            source_text=source['text'],source_text_sha256=source['text_sha256'],
                            image_assisted_text=False,received_caption='',chunk_generated_token_ids=chunks,
                            generated_token_ids=joined,stopped_by_capacity=capacity,**metrics(source['text'],'')))
        write_rows(ev/'rows.jsonl',data)
        write_rows(ev/'text_rows.jsonl',text_data)
        core.atomic(ev/'EVAL_COMPLETE.json',dict(rows=len(data),rows_sha256=core.sha(ev/'rows.jsonl'),
            text_rows=len(text_data),text_rows_sha256=core.sha(ev/'text_rows.jsonl')))
    core.atomic(root/'PIPELINE.json',dict(status='SMOKE_COMPLETE',stages=stages))
    return c

class StorageContracts(unittest.TestCase):
    def setUp(self):
        base = core.PROJECT/'experiment_results'
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='v20_TEST_ONLY_',dir=str(base))
        self.root = Path(self.temp.name)
        fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_and_no_false_quality_claim(self):
        result = report.verify(self.root)
        self.assertEqual(result['status'],'SMOKE_COMPLETE')
        self.assertIsNone(result['comparisons'])
        self.assertFalse(result['full_semantic_success'])

    def alter(self,field,value):
        p = self.root/'smoke_eval_C_tx_rx_text/rows.jsonl'
        data = core.rows(p)
        target = next(r for r in data if r['control']=='rx_shuffled')
        target[field] = value
        write_rows(p,data)
        complete = core.read(p.parent/'EVAL_COMPLETE.json')
        complete['rows_sha256'] = core.sha(p)
        core.atomic(p.parent/'EVAL_COMPLETE.json',complete)

    def test_rx_control_cannot_change_transmitted_image(self):
        self.alter('tx_image_sha256','BAD')
        with self.assertRaisesRegex(ValueError,'RX control changed'):
            report.verify(self.root)

    def test_wrong_source_and_fake_donor_fail(self):
        self.alter('donor_id','not_in_cohort')
        with self.assertRaises(ValueError):
            report.verify(self.root)

    def test_missing_case_even_with_updated_seal_fails(self):
        p = self.root/'smoke_eval_A_self/rows.jsonl'
        data = core.rows(p)[:-1]
        write_rows(p,data)
        v = core.read(p.parent/'EVAL_COMPLETE.json')
        v.update(rows=len(data),rows_sha256=core.sha(p))
        core.atomic(p.parent/'EVAL_COMPLETE.json',v)
        with self.assertRaisesRegex(ValueError,'Missing evaluation'):
            report.verify(self.root)

    def test_private_render_escapes_source_and_preserves_conflicting_report(self):
        from docci_text_guided_image_v20.display import render
        render(self.root,report.verify(self.root))
        self.assertIn('PRIVATE', (self.root/'index.html').read_text())
        with self.assertRaises(ValueError):
            render(self.root,dict(different=True))

    def test_forged_snr_or_psnr_rejected(self):
        self.alter('psnr',30.)
        with self.assertRaisesRegex(ValueError,'PSNR'):
            report.verify(self.root)

    def test_archive_readback_private_manifest_and_no_overwrite(self):
        import tarfile
        from docci_text_guided_image_v20.collect import collect
        dest = self.root/'fixture_records.tar.gz'
        value = report.verify(self.root)
        with patch('docci_text_guided_image_v20.collect.verify',return_value=value):
            collect(self.root,dest,1)
            with self.assertRaises(FileExistsError):
                collect(self.root,dest,1)
        with tarfile.open(dest) as tf:
            manifest = json.load(tf.extractfile('v20_records/ARCHIVE_MANIFEST.json'))
            for name,h in manifest['files'].items():
                self.assertEqual(hashlib.sha256(tf.extractfile('v20_records/'+name).read()).hexdigest(),h)

    def test_large_weights_are_not_exported(self):
        import tarfile
        from docci_text_guided_image_v20.collect import collect
        value = report.verify(self.root)
        (self.root/'TEST_ONLY.pth').write_bytes(b'not a model')
        with patch('docci_text_guided_image_v20.collect.verify',return_value=value):
            collect(self.root,self.root/'fixture_records.tar.gz',1)
        with tarfile.open(self.root/'fixture_records.tar.gz') as tf:
            self.assertFalse(any(m.name.endswith('.pth') for m in tf.getmembers()))

    def test_stale_or_false_completion_is_rejected(self):
        p = self.root/'PIPELINE.json'
        state = core.read(p)
        state['stages']['not_really_run'] = 'VERIFIED'
        core.atomic(p,state)
        with self.assertRaisesRegex(ValueError,'falsely'):
            report.verify(self.root)

    def test_owned_child_guard_does_not_signal_unrelated_process_group(self):
        from docci_text_guided_image_v20.launch import stop_owned
        class Child:
            pid = 98765
            def poll(self):
                return None
        with patch('docci_text_guided_image_v20.launch.os.getpgid',return_value=123),\
                patch('docci_text_guided_image_v20.launch.os.killpg') as kill:
            with self.assertRaises(RuntimeError):
                stop_owned(Child())
            kill.assert_not_called()

if __name__=='__main__':
    unittest.main(verbosity=2)
