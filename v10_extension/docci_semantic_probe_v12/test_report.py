import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_semantic_probe_v12.core import PROJECT,digest,load,indices,lr,damage_numpy,seed_for
from docci_semantic_probe_v12.report import matrix,file_at,text_evaluation,render,verify_image,ah
from docci_text_continue_v11.metrics import metrics
TEST_ROOT=PROJECT/'.cache/v10/docci_semantic_probe_v12_tests'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False))


class Verification(unittest.TestCase):
    @classmethod
    def setUpClass(cls): TEST_ROOT.mkdir(parents=True,exist_ok=True)
    def test_complete_and_duplicate_missing_matrix(self):
        rows=[dict(example_id=e,condition='clean',control=c) for e in ('a','b') for c in ('matched','zero','shuffled')]
        matrix(rows,['a','b'],['clean'],['matched','zero','shuffled'])
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError): matrix(bad,['a','b'],['clean'],['matched','zero','shuffled'])

    def fixture(self,root):
        receipts={e:dict(example_id=e,official_split='qual_dev',text_sha256=hashlib.sha256(b'a b').hexdigest(),source_tokens=4) for e in ('a','b')}
        frames={}; rows=[]
        for i,e in enumerate(receipts):
            f=root/'frames'/('clean__'+e+'.npz'); f.parent.mkdir(exist_ok=True)
            x=np.ones(18432,np.float32)*(1 if i==0 else -1)
            np.savez_compressed(f,observed=x,available=np.asarray(True),pilot_snr_db=np.asarray(40.,np.float32),reliability=np.asarray(1.,np.float32))
            frames[e]=(hashlib.sha256(x.tobytes()).hexdigest(),digest(f))
        for e in receipts:
            donor='b' if e=='a' else 'a'
            for ctrl in ('matched','zero','shuffled'):
                caption='' if ctrl=='zero' else 'a b'; seq=([[2,0,10,11,2]]+[[2,0,2]]*8) if caption else [[2,0,2]]*9
                rows.append(dict(example_id=e,condition='clean',control=ctrl,official_split='qual_dev',
                    source_text='a b',source_text_sha256=receipts[e]['text_sha256'],source_tokens=4,
                    received_caption=caption,received_sha256=frames[e][0],tx_wire_sha256=frames[e][0],frame_sha256=frames[e][1],
                    donor_example_id=donor if ctrl=='shuffled' else None,donor_received_sha256=frames[donor][0] if ctrl=='shuffled' else None,
                    chunk_generated_token_ids=seq,generated_token_ids=[10,11] if caption else [],stopped_by_capacity=False,
                    input_truncated=False,receiver_failure=False,stats=dict(payload_complex_symbols=9216,analog_repeats=1,clean_sentinel_not_measured_snr=True),
                    **metrics('a b',caption)))
        m=dict(example_ids=list(receipts),cohort='validation',conditions=['clean'],controls=['matched','zero','shuffled'],
            source_count_or_mask_at_RX=False,Common_used=False,clean_snr40_is_sentinel=True)
        write(root/'matrix.json',m)
        (root/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        write(root/'EVAL_COMPLETE.json',dict(rows=len(rows),rows_sha256=digest(root/'rows.jsonl'),matrix_sha256=digest(root/'matrix.json')))
        return receipts,rows

    def test_text_positive_real_format_and_recomputed_metrics(self):
        with tempfile.TemporaryDirectory(dir=str(TEST_ROOT)) as d:
            receipts,rows=self.fixture(Path(d)); actual=text_evaluation(Path(d),receipts,'validation')
            self.assertEqual(len(actual),6)

    def test_forged_metric_fails_even_if_completion_hash_recomputed(self):
        with tempfile.TemporaryDirectory(dir=str(TEST_ROOT)) as d:
            root=Path(d); receipts,rows=self.fixture(root); rows[0]['normalized_wer']=.9
            (root/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            done=json.loads((root/'EVAL_COMPLETE.json').read_text()); done['rows_sha256']=digest(root/'rows.jsonl'); write(root/'EVAL_COMPLETE.json',done)
            with self.assertRaises(ValueError): text_evaluation(root,receipts,'validation')

    def test_frame_and_donor_corruption_fail(self):
        with tempfile.TemporaryDirectory(dir=str(TEST_ROOT)) as d:
            root=Path(d); receipts,rows=self.fixture(root)
            with (root/'frames/clean__a.npz').open('wb') as f: np.savez(f,observed=np.zeros(18432,np.float32),available=True,pilot_snr_db=40.,reliability=1.)
            with self.assertRaises(ValueError): text_evaluation(root,receipts,'validation')

    def test_html_escapes_text_and_labels_oracle(self):
        page=render(Path('.'),dict(task_results={'smoke':{}}))
        self.assertIn('未经过传输',page); self.assertIn('LPIPS 未测',page)

    def test_paths_escape_rejected(self):
        with self.assertRaises(ValueError): file_at(Path('.'),'../weight.pth')

    def image_fixture(self,root):
        """Synthetic RECORDS, not trained weights or a claimed model result."""
        c=load(); ids=['a','b']; train=[dict(example_id='t'+str(i),official_split='train') for i in range(8)]
        val=[dict(example_id=e,official_split='qual_dev',image_sha256='source'+e,rectangle=[0,0,512,512]) for e in ids]
        semantics={e:ah(np.full(512,i+1,np.float32)) for i,e in enumerate(ids)}
        write(root/'data_identity.json',dict(train=dict(records=train),validation=dict(records=val),
            validation_semantics=[dict(example_id=e,ORACLE_untransmitted=True,input_truncated=False,
                semantic_sha256=semantics[e],text_sha256=hashlib.sha256(e.encode()).hexdigest()) for e in ids],
            validation_wires=[dict(example_id=e,wire_sha256=ah(np.ones(8192,np.float32)*(1 if e=='a' else -1))) for e in ids],
            longclip_signature_before='frozen',longclip_signature_after='frozen'))
        arms={}
        for arm in ('A0_image_only','A1_oracle_text'):
            folder=root/arm/'train'; starts={'weight':'initial'}; state={'weight':'final'}
            write(folder/'trainability.json',dict(arm=arm,updates=2,training_example_ids=[r['example_id'] for r in train],
                backbone_frozen=True,initial_state_hashes=starts,active_parameters=399843))
            history=[]; seen=set()
            for u in range(2):
                picked=[train[i]['example_id'] for i in indices(8,c['image_grad_accum'],u,42)]; seen.update(picked)
                history.append(dict(update=u+1,batch_example_ids=picked,unique_examples_seen=len(seen),lr=lr(c,'image',u,2),
                    loss=1.,grad_norm=1.,elapsed_seconds=float(u),condition=c['image_train_conditions'][u%2],teacher_executed=False))
            (folder/'history.jsonl').write_text('\n'.join(json.dumps(r) for r in history)+'\n')
            write(folder/'learning_audit.json',dict(passed=True,updates=2,groups={'adapter':dict(parameters=399843,changed=True,nonzero_gradient=True)}))
            write(folder/'CHECKPOINT.json',dict(updates=2,exact_disk_roundtrip=True,parent_optimizer_restored=False,state_hashes=state))
            write(folder/'TRAIN_COMPLETE.json',dict(updates=2)); arms[arm]=dict(final_state_hashes=state)
        write(root/'identity.json',dict(arms=arms,frozen_image_codec_before='frozen',frozen_image_codec_after='frozen',
            frozen_VAE_before='frozen',frozen_VAE_after='frozen',FSO=False,Common=False,diffusion=False,
            ORACLE_text_is_privileged_NOT_actual_communication=True))
        frames={}; (root/'frames').mkdir()
        for condition in c['image_eval_conditions']:
            for e in ids:
                x=np.ones(8192,np.float32)*(1 if e=='a' else -1); seed=seed_for(42,'EVAL',condition,e); y=damage_numpy(x,condition,seed)
                f=root/'frames'/(condition+'__'+e+'.npz'); np.savez_compressed(f,observed=y,transmitted=x)
                frames[(condition,e)]=(digest(f),ah(x),ah(y),seed)
        for tag in ('P0_parent','A0_image_only_evaluation','A1_oracle_text_evaluation'):
            folder=root/tag; (folder/'images').mkdir(parents=True); oracle=tag.startswith('A1'); controls=['matched','shuffled','zero'] if oracle else ['none']; rows=[]
            for condition in c['image_eval_conditions']:
                for e in ids:
                    for ctrl in controls:
                        donor='b' if e=='a' else 'a'; h,tx,rx,seed=frames[(condition,e)]
                        f=folder/'images'/(condition+'__'+e+'__'+ctrl+'.png'); src=folder/'images'/('source__'+e+'.png'); pixels=np.zeros((512,512,3),np.uint8)
                        Image.fromarray(pixels).save(f)
                        if not src.exists(): Image.fromarray(pixels).save(src)
                        rows.append(dict(tag=tag,control=ctrl,condition=condition,example_id=e,official_split='qual_dev',
                            frame_sha256=h,transmitted_sha256=tx,observed_sha256=rx,damage_seed=seed,source_image_sha256='source'+e,
                            source_text=e,source_text_sha256=hashlib.sha256(e.encode()).hexdigest(),rectangle=[0,0,512,512],
                            semantic_sha256=(semantics[donor] if ctrl=='shuffled' else ah(np.zeros(512,np.float32)) if ctrl=='zero' else semantics[e]) if oracle else None,
                            semantic_donor_example_id=donor if ctrl=='shuffled' else None,oracle_text_not_transmitted=oracle,
                            payload_complex_symbols=4096,analog_repeats=1,physical_snr_db=None,BER=None,LPIPS=None,psnr=120.,mse=0.,ssim=1.,
                            image_file=str(f.relative_to(root)),image_file_sha256=digest(f),source_view_file=str(src.relative_to(root)),source_view_sha256=digest(src),quantized_image_sha256=ah(pixels)))
            write(folder/'matrix.json',dict(example_ids=ids,conditions=c['image_eval_conditions'],controls=controls,ORACLE_text_untransmitted=oracle,FSO=False,image_complex_symbols=4096,LPIPS=None))
            (folder/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            write(folder/'EVAL_COMPLETE.json',dict(rows=len(rows),rows_sha256=digest(folder/'rows.jsonl'),matrix_sha256=digest(folder/'matrix.json')))
        return c

    def test_image_complete_paired_records_and_PNG_subset(self):
        with tempfile.TemporaryDirectory(dir=str(TEST_ROOT)) as d:
            root=Path(d); c=self.image_fixture(root); result=verify_image(root,c,True)
            self.assertEqual(result['decision'],'SMOKE_ONLY'); self.assertEqual(result['PNG_subset_MSE_rechecks'],20)

    def test_image_oracle_donor_forgery_rejected(self):
        with tempfile.TemporaryDirectory(dir=str(TEST_ROOT)) as d:
            root=Path(d); c=self.image_fixture(root); folder=root/'A1_oracle_text_evaluation'
            rows=[json.loads(s) for s in (folder/'rows.jsonl').read_text().splitlines()]
            rows[1]['semantic_donor_example_id']='a'
            (folder/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            done=json.loads((folder/'EVAL_COMPLETE.json').read_text()); done['rows_sha256']=digest(folder/'rows.jsonl'); write(folder/'EVAL_COMPLETE.json',done)
            with self.assertRaises(ValueError): verify_image(root,c,True)

    def test_image_initialization_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(dir=str(TEST_ROOT)) as d:
            root=Path(d); c=self.image_fixture(root); f=root/'A1_oracle_text/train/trainability.json'; t=json.loads(f.read_text())
            t['initial_state_hashes']={'other':'different'}; write(f,t)
            with self.assertRaises(ValueError): verify_image(root,c,True)


if __name__=='__main__': unittest.main(verbosity=2)
