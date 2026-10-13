"""Synthetic independent report tests; corrupt evidence cannot mark complete."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from docci_longclip_joint_v8.core import PROJECT,load,schedule,geometry,atomic_json,digest
from docci_longclip_joint_v8.metrics import metrics
from docci_longclip_joint_v8.report import verify,finish,under,CONTROLS


def write_lines(file,rows):
    with file.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')


def fixture(root,quality_stop=False):
    c=load(image_size=256,max_text_tokens=256); sch=schedule(c,'smoke'); g=geometry(c)
    source=dict(train=[dict(example_id='train_'+str(i),text='red car',text_sha256=hashlib.sha256(b'red car').hexdigest()) for i in range(8)],
        validation=[dict(example_id='qual_dev_'+str(i),text='red car',text_sha256=hashlib.sha256(b'red car').hexdigest()) for i in range(2)])
    receipts={}
    for key,split in (('train','train'),('validation','qual_dev')):
        receipts[key]=[dict(r,official_split=split,image_sha256='a'*64,source_tokens=4,clip_tokens=2,clip_chunks=1,rectangle=[0,0,256,256],native_size=[1024,1024]) for r in source[key]]
    design=dict(config=c,schedule=sch,train_view=dict(records=receipts['train']),validation_view=dict(records=receipts['validation']),test_used=False)
    identity=dict(version='docci-longclip-joint-v8',config=c,schedule=sch,source_sha256={},status='quality_stop' if quality_stop else 'finished_execution',mode='smoke',wall_seconds=1.)
    for name,obj in (('run_identity.json',identity),('design.json',design),('docci_view.json',source),('frozen_verification.json',dict(passed=True)),
        ('checkpoint_load.json',dict(longclip_sha256='ca2face3716d60473a4e284e5be2cf0a1f3a4ad9954a6a651d878e814cc0c894'))): atomic_json(root/name,obj)
    for arm,stage in [('prefit','text'),('prefit','image')]+[(a,s) for a in ('separate','joint') for s in ('clean','fso')]:
        folder=root/arm/stage; folder.mkdir(parents=True)
        atomic_json(folder/'learning_audit.json',dict(passed=True,groups=dict(codec=dict(changed=True,nonzero_gradient=True,parameters=1))))
        atomic_json(folder/'TRAIN_COMPLETE.json',dict(arm=arm,stage=stage,checkpoint_sha256='b'*64))
    controls=[]
    for item in source['validation']:
        for control in ('matched','zero','shuffled'):
            text=item['text'] if control=='matched' and not quality_stop else ''
            controls.append(dict(example_id=item['example_id'],source_split='qual_dev',control=control,source_text=item['text'],
                covered_text=item['text'],input_truncated=False,received_caption=text,stopped_by_capacity=False,**metrics(item['text'],item['text'],text)))
    write_lines(root/'prefit/validation_controls.jsonl',controls)
    atomic_json(root/'prefit/text_gate.json',dict(passed=not quality_stop,full_wer=1. if quality_stop else 0.))
    if quality_stop: return
    ev=root/'evaluation'; ev.mkdir(); (ev/'frames').mkdir(); (ev/'images').mkdir()
    for item in source['validation']: Image.new('RGB',(256,256),(255,255,255)).save(ev/'images'/('source__'+item['example_id']+'.png'))
    res=dict(g,slot_samples=30000); matrix=dict(arms=['separate','joint'],stages=['clean','fso'],conditions=['clean','moderate_p18'],
        controls=list(CONTROLS),example_ids=[r['example_id'] for r in source['validation']],resources=res)
    atomic_json(ev/'matrix.json',matrix); rows=[]
    for arm in matrix['arms']:
        for stage in matrix['stages']:
            for cond in matrix['conditions']:
                for item in source['validation']:
                    eid=item['example_id']; fn='frames/'+arm+'__'+stage+'__'+cond+'__'+eid+'.npz'
                    x=np.zeros(g['total_reals'],np.float32)
                    with (ev/fn).open('xb') as f: np.savez_compressed(f,observed=x,available=np.asarray(True),pilot_snr_db=np.asarray(40.,np.float32),reliability=np.asarray(1.,np.float32))
                    sha=hashlib.sha256(x.tobytes()).hexdigest(); frame_sha=digest(ev/fn)
                    stats=dict(pilot_snr_db=40.,analog_reliability=1.,receiver_available=True,payload_complex_symbols=g['payload_complex_symbols'],
                        analog_repeats=1,raw_utf8_wire_bytes=0,resources=res,resource_feasible=True,slot_samples=30000,used_samples=30000,
                        launch_signal_energy_j=1.,signal_energy_budget_j=1.)
                    for control in CONTROLS:
                        donor=matrix['example_ids'][1-matrix['example_ids'].index(eid)] if control.startswith('shuffle_') else None
                        image='images/'+arm+'__'+stage+'__'+cond+'__'+eid+'__'+control+'.png'; Image.new('RGB',(256,256)).save(ev/image)
                        rows.append(dict(arm=arm,stage=stage,condition=cond,example_id=eid,control=control,official_split='qual_dev',input_truncated=False,image_size=256,
                            source_text=item['text'],received_caption=item['text'],source_text_sha256=item['text_sha256'],source_image_sha256='a'*64,source_tokens=4,
                            evaluation_rectangle=[0,0,256,256],stopped_by_capacity=False,receiver_failure=False,mse=1.,psnr=0.,ssim=0.,
                            **metrics(item['text'],item['text'],item['text']),frame_file=fn,frame_sha256=frame_sha,received_sha256=sha,stats=stats,noise_seed=7,
                            tx_wire_sha256='c'*64,donor_example_id=donor,donor_received_sha256=sha if donor else None,restored_image_file=image,restored_image_sha256=digest(ev/image)))
    write_lines(ev/'rows.jsonl',rows); atomic_json(ev/'EVAL_COMPLETE.json',dict(rows=len(rows),rows_sha256=digest(ev/'rows.jsonl')))


class Reports(unittest.TestCase):
    def setUp(self):
        release=PROJECT/'releases'; release.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='docci_v8_report_test_',dir=str(release)); self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def test_smoke_never_full_complete_and_full_text_recomputed(self):
        fixture(self.root); result=finish(self.root)
        self.assertEqual(result['rows'],96); self.assertTrue((self.root/'SMOKE_COMPLETE.json').exists()); self.assertFalse((self.root/'COMPLETE.json').exists())
        rows=[json.loads(x) for x in (self.root/'evaluation/rows.jsonl').read_text().splitlines()]
        rows[0]['normalized_wer']=.5; write_lines(self.root/'evaluation/rows.jsonl',rows)
        done=json.loads((self.root/'evaluation/EVAL_COMPLETE.json').read_text()); done['rows_sha256']=digest(self.root/'evaluation/rows.jsonl'); atomic_json(self.root/'evaluation/EVAL_COMPLETE.json',done)
        with self.assertRaisesRegex(ValueError,'metric mismatch'): verify(self.root)
    def test_quality_stop_not_complete(self):
        fixture(self.root,True); finish(self.root)
        self.assertTrue((self.root/'QUALITY_STOP.json').exists()); self.assertFalse((self.root/'COMPLETE.json').exists())
    def test_forbidden_receiver_source_rejected(self):
        fixture(self.root); frame=next((self.root/'evaluation/frames').glob('*.npz'))
        with np.load(frame,allow_pickle=False) as z: data={k:z[k].copy() for k in z.files}
        data['source_text']=np.asarray('forbidden')
        with frame.open('wb') as f: np.savez_compressed(f,**data)
        rows=[json.loads(x) for x in (self.root/'evaluation/rows.jsonl').read_text().splitlines()]
        for r in rows:
            if r['frame_file']=='frames/'+frame.name: r['frame_sha256']=digest(frame)
        write_lines(self.root/'evaluation/rows.jsonl',rows)
        done=json.loads((self.root/'evaluation/EVAL_COMPLETE.json').read_text()); done['rows_sha256']=digest(self.root/'evaluation/rows.jsonl'); atomic_json(self.root/'evaluation/EVAL_COMPLETE.json',done)
        with self.assertRaisesRegex(ValueError,'Forbidden'): verify(self.root)
    def test_bad_resource_or_truncation_cannot_complete(self):
        fixture(self.root); rows=[json.loads(x) for x in (self.root/'evaluation/rows.jsonl').read_text().splitlines()]
        rows[0]['input_truncated']=True; write_lines(self.root/'evaluation/rows.jsonl',rows)
        done=json.loads((self.root/'evaluation/EVAL_COMPLETE.json').read_text()); done['rows_sha256']=digest(self.root/'evaluation/rows.jsonl'); atomic_json(self.root/'evaluation/EVAL_COMPLETE.json',done)
        with self.assertRaises(ValueError): verify(self.root)
    def test_path_escape_and_symlink(self):
        with self.assertRaises(ValueError): under(self.root,'../source')
        (self.root/'link').symlink_to('/private/tmp',target_is_directory=True)
        with self.assertRaises(ValueError): under(self.root,'link/somefile')


if __name__=='__main__': unittest.main(verbosity=2)
