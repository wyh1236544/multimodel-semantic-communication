"""CPU contracts and falsification tests. Synthetic records are not ML results."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_text_channel_v18 import core
from docci_text_channel_v18.report import stats_check,verify_train

class Contracts(unittest.TestCase):
    def test_native_feature_cache_producer_schema_and_PAD_validation(self):
        from docci_text_channel_v18.data import Items
        from docci_baseline_v14.core import array_hash
        with tempfile.TemporaryDirectory(dir=str(core.PROJECT/'experiment_results'),prefix='v18_native_schema_') as d:
            ids=np.tile(np.array([0,4,2]+[1]*61,dtype=np.int64),(9,1));hidden=np.zeros((9,64,768),np.float32)
            identity=dict(schema='v14-feature',example_id='train_00000',task='text')
            meta=dict(identity=identity,array_hashes=dict(chunk_ids=array_hash(ids),chunk_hidden=array_hash(hidden)))
            p=Path(d)/'data.npz';np.savez_compressed(p,meta=np.asarray(json.dumps(meta)),chunk_ids=ids,chunk_hidden=hidden)
            row=dict(example_id='train_00000',official_split='train',source_tokens=11,cache_file=str(p),cache_sha256=core.sha(p),identity=identity)
            self.assertEqual(Items([row])[0]['ids'].shape,(9,64))
            wrong=dict(row,source_tokens=10)
            with self.assertRaises(ValueError):Items([wrong])[0]
    def test_evaluation_producer_schema_and_incomplete_matrix_rejected(self):
        from docci_text_channel_v18.report import verify_eval
        from docci_text_continue_v11.metrics import metrics
        with tempfile.TemporaryDirectory(dir=str(core.PROJECT/'experiment_results'),prefix='v18_eval_schema_') as d:
            root=Path(d)/'smoke_eval_frozen';root.mkdir();c=core.load(smoke=True)
            rs=[dict(example_id='qual_dev_0000'+str(i),text='dog',text_sha256=hashlib.sha256(b'dog').hexdigest(),source_tokens=3) for i in range(2)]
            chunks=[[2,0,4,2]]+[[2,0,2] for _ in range(8)]
            old=[dict(example_id=r['example_id'],strategy='greedy',control='matched',chunk_generated_token_ids=chunks) for r in rs]
            p=Path(d)/'smoke_inputs/pinned/released64/validation/rows.jsonl';p.parent.mkdir(parents=True)
            p.write_text(''.join(json.dumps(x)+'\n' for x in old))
            rows=[]
            def stats(eid,snr,repeat):
                value=core.trial_seed(c,'validation',0,eid,snr,repeat);epsilon=core.noise(18432,snr,value);pn=float(np.mean(epsilon.astype(np.float64)**2));wire=np.ones(18432,np.float32)
                return dict(nominal_snr_db=snr,noise_seed=value,epsilon_sha256=hashlib.sha256(epsilon.tobytes()).hexdigest(),post_noise_normalization=False,signal_power=1.,noise_power=pn,realized_snr_db=None if snr is None else float(10*np.log10(1/pn)),tx_sha256=hashlib.sha256(wire.tobytes()).hexdigest(),observed_sha256=hashlib.sha256((wire+epsilon).tobytes()).hexdigest())
            for snr,repeat,control in core.trials(c,True):
                for i,r in enumerate(rs):
                    own=stats(r['example_id'],snr,repeat);donor=rs[(i+1)%2]['example_id'] if control=='shuffled' else None
                    delivered=stats(donor,snr,repeat) if donor else own
                    rows.append(dict(example_id=r['example_id'],official_split='qual_dev',arm='frozen',source_text=r['text'],source_text_sha256=r['text_sha256'],source_tokens=3,input_truncated=False,snr_db=snr,repeat=repeat,control=control,donor_id=donor,own_stats=own,delivered_stats=delivered,received_payload_sha256=hashlib.sha256(np.zeros(18432,np.float32).tobytes()).hexdigest() if control=='zero' else delivered['observed_sha256'],received_caption='dog',chunk_generated_token_ids=chunks,generated_token_ids=[4],stopped_by_capacity=False,**metrics('dog','dog')))
            core.atomic(root/'identity.json',dict(arm='frozen',config=c,ids=[r['example_id'] for r in rs],trials=[list(x) for x in core.trials(c,True)],parent_sha256=core.PARENT_SHA,checkpoint_sha256=core.PARENT_SHA,smoke=True))
            (root/'rows.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));core.atomic(root/'EVAL_COMPLETE.json',dict(rows=len(rows)));core.seal(root)
            self.assertEqual(verify_eval(root,rs,c,True,lambda:None)['rows'],12)
            (root/'rows.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows[:-1]));core.seal(root)
            with self.assertRaises(ValueError):verify_eval(root,rs,c,True,lambda:None)
    def test_config_finite_and_protocol_bounds(self):
        c=core.load();self.assertEqual((c['train_samples'],c['validation_samples']),(2048,100))
        for key,value in [('learning_rate',float('nan')),('reserve_free_gib',1),('clean_weight',.5),('epochs',0),('evaluation_repeats',1)]:
            x=copy.deepcopy(c);x[key]=value
            with patch.object(core,'read',side_effect=[x,c]):
                with self.assertRaises(ValueError):core.load('TEST_ONLY')
    def test_smoke_shortens_explicitly_and_no_inplace_config_mutation(self):
        a,b=core.load(),core.load(smoke=True)
        self.assertEqual(a['train_samples'],2048);self.assertEqual(b['train_samples'],8)
        self.assertEqual(b['evaluation_repeats'],1)
    def test_order_coverage_and_pairing(self):
        a=list(core.batches(17,8,3,42));b=list(core.batches(17,8,3,42))
        self.assertEqual(a,b)
        for e in range(3):self.assertEqual(sorted(i for epoch,ix in a if epoch==e for i in ix),list(range(17)))
        self.assertNotEqual(a[0][1],a[3][1])
    def test_symbol_noise_power_scaling_common_and_no_arm_key(self):
        a=core.noise(18432,0,123);b=core.noise(18432,10,123)
        self.assertTrue(np.array_equal(a,core.noise(18432,0,123)))
        self.assertTrue(np.allclose(b,a*np.float32(10**(-.5)),rtol=0,atol=0))
        self.assertLess(abs(float(np.mean(a*a))-1),.05)
        self.assertTrue(np.array_equal(core.noise(18432,None,123),np.zeros(18432,np.float32)))
    def test_complete_evaluation_and_negative_controls_not_every_noise_draw(self):
        c=core.load();ts=list(core.trials(c));self.assertEqual(len(ts),20)
        self.assertEqual(sum(k=='matched' for s,r,k in ts),16)
        self.assertEqual(set((s,r) for s,r,k in ts if k!='matched'),{(None,0),(0,0)})
    def test_stats_recomputed_and_tampering_rejected(self):
        x=core.noise(18432,10,42);pn=float(np.mean(x.astype(np.float64)**2))
        s=dict(nominal_snr_db=10,noise_seed=42,epsilon_sha256=hashlib.sha256(x.tobytes()).hexdigest(),post_noise_normalization=False,
               signal_power=1.,noise_power=pn,realized_snr_db=10*np.log10(1/pn))
        stats_check(s,10,42)
        for key,value in [('noise_power',pn*2),('post_noise_normalization',True),('noise_seed',43),('nominal_snr_db',0)]:
            with self.assertRaises(ValueError):stats_check(dict(s,**{key:value}),10,42)
    def test_safe_output_rejects_root_escape_and_symlink(self):
        with self.assertRaises(ValueError):core.safe(core.PROJECT)
        with self.assertRaises(ValueError):core.safe('../outside')
        with tempfile.TemporaryDirectory(dir=str(core.PROJECT/'experiment_results'),prefix='v18_path_test_') as d:
            p=Path(d)/'link';p.symlink_to('/tmp',target_is_directory=True)
            with self.assertRaises(ValueError):core.safe(p/'output')
    def test_private_render_incomplete_inputs_without_false_result(self):
        from docci_text_channel_v18.collect import render
        with tempfile.TemporaryDirectory(dir=str(core.PROJECT/'experiment_results'),prefix='v18_render_test_') as d:
            text=render(Path(d),dict(status='FAILED',verified_stages={}))
            self.assertIn('暂无完整主评价',text);self.assertNotIn('84.06',text)
    def test_owned_process_stop_never_global_name_match(self):
        from docci_text_channel_v18.launch import stop_owned
        class Done:
            def poll(self):return 0
        with patch('os.killpg') as k:stop_owned(Done());k.assert_not_called()
    def test_collect_new_target_refuses_without_touching_existing(self):
        from docci_text_channel_v18.collect import collect
        with tempfile.TemporaryDirectory(dir=str(core.PROJECT/'experiment_results'),prefix='v18_collect_test_') as d:
            root=Path(d)/'run';root.mkdir();p=Path(d)/'exists.tar.gz';p.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):collect(root,p)
            self.assertEqual(p.read_bytes(),b'keep')
if __name__=='__main__':unittest.main(verbosity=2)
