"""CPU contracts, synthetic integration and failure paths; never claim model quality."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
import numpy as np
from PIL import Image
from docci_evidence_v23 import core as c
from docci_evidence_v23 import report

def fixture(root):
    root.mkdir();(root/'truth').mkdir();(root/'receiver').mkdir();examples=[];data=[]
    ids=['train_UNIT_A','train_UNIT_B']
    for i,eid in enumerate(ids):
        im=Image.new('RGB',(512,512),(60+80*i,100,180-50*i));path=root/'truth'/(eid+'.png');im.save(path)
        text='a blue object' if i==0 else 'two red objects'
        examples.append(dict(example_id=eid,cohort='dev',original_split='train',source_image='truth/'+eid+'.png',
            source_image_sha256=c.sha(path),source_text=text,source_text_sha256=c.digest(text.encode()),rectangle=[0,64,512,448]))
        for snr in c.CAPTION_SNR:
            out=root/'receiver'/(c.case_name(eid,snr,0)+'.png');im.save(out)
            data.append(dict(example_id=eid,donor_id=ids[1-i],snr_db=snr,repeat=0,received_image=out.relative_to(root).as_posix(),
                received_image_sha256=c.sha(out),received_caption=text,received_caption_sha256=c.digest(text.encode()),
                image_payload_symbols=4096,text_payload_symbols=9216,total_symbols=13376,source_text_at_RX=False,post_noise_renormalization=False))
    c.atomic(root/'index.json',dict(profile='smoke',spec=c.PROFILES['smoke'],examples=examples,noise_family='v23'))
    with (root/'rows.jsonl').open('x') as f:
        for r in data:c.line(f,r)
    c.complete(root,'prepare',dict(rows=6));return root

def reseal(root):
    p=root/'COMPLETE.json';v=c.read(p)
    v['files']={x.relative_to(root).as_posix():c.sha(x) for x in root.rglob('*') if x.is_file() and x.name not in ('COMPLETE.json','LAUNCH.json')}
    p.write_text(json.dumps(v))

class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='v23_UNIT_',dir=c.PROJECT/'experiment_results')
        self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_fixed_protocol_and_symbol_accounting(self):
        self.assertEqual(c.PROTOCOL['image_frame_symbols']+c.PROTOCOL['text_frame_symbols'],13376)
        self.assertEqual(c.PROFILES['pilot']['images'],32)
        self.assertEqual(c.GRID,[None,-8,-5,-2,0,2,5,10,20])
        self.assertEqual(len(c.CONTROLS),6)
    def test_noise_same_v22_bytes_and_all_new_SNRs(self):
        from docci_pretrained_checks_v22.core import noise
        for s in c.CAPTION_SNR:self.assertTrue(np.array_equal(c.noise('v22','x',s,0,'image',8192),noise('x',s,0,'image',8192)))
        for s in c.GRID:
            a=c.noise('v23','x',s,0,'image',8192);self.assertTrue(np.array_equal(a,c.noise('v23','x',s,0,'image',8192)))
        with self.assertRaises(ValueError):c.noise('v23','x',-7,0,'image',1)
        self.assertFalse(np.array_equal(c.noise('v23','x',0,0,'image',100),c.noise('v23','x',0,0,'text',100)))
    def test_awgn_actual_variance_and_no_post_normalization(self):
        e=c.noise('v23','x',-5,0,'image',100000)
        self.assertAlmostEqual(float(np.mean(e*e)),10**.5,delta=.05)
        y=c.digital_observation(np.ones(4128,np.complex128),{'noise_family':'v23'},dict(example_id='x',snr_db=-5,repeat=0))
        self.assertGreater(float(np.mean(abs(y)**2)),2)
    def test_complete_and_preparation_matrix(self):
        p=fixture(self.root/'p');self.assertEqual(report.verify(p)['rows'],6)
    def test_missing_or_duplicate_preparation_rejected(self):
        p=fixture(self.root/'p');rr=c.rows(p/'rows.jsonl');rr[-1]=rr[0]
        (p/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rr));reseal(p)
        with self.assertRaisesRegex(ValueError,'matrix'):c.prepared(p)
    def test_original_truth_and_receiver_hash_tamper_rejected(self):
        p=fixture(self.root/'p');r=c.rows(p/'rows.jsonl')[0]
        (p/r['received_image']).write_bytes(b'bad');reseal(p)
        with self.assertRaisesRegex(ValueError,'PNG'):c.prepared(p)
    def test_source_at_receiver_or_wrong_budget_rejected(self):
        p=fixture(self.root/'p');rr=c.rows(p/'rows.jsonl');rr[0]['total_symbols']=4096
        (p/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rr));reseal(p)
        with self.assertRaisesRegex(ValueError,'resource'):c.prepared(p)
    def test_atomic_no_overwrite_and_finite_json(self):
        p=self.root/'a.json';c.atomic(p,dict(a=1))
        with self.assertRaises(FileExistsError):c.atomic(p,dict(a=2))
        with self.assertRaises(ValueError):c.atomic(self.root/'nan.json',dict(a=float('nan')))
    def test_budget_constraints(self):
        for x in (0,-1,float('nan'),13):
            with self.assertRaises(ValueError):c.Budget(x)
        b=c.Budget(.01);b.start-=100
        with self.assertRaises(TimeoutError):b()
    def test_safe_paths(self):
        with self.assertRaises(ValueError):c.local(self.root,'../x')
        with self.assertRaises(ValueError):c.safe('/tmp/v23_UNIT_escape')
    def test_latent_statistics_zero_correlated_and_shape(self):
        zero=c.residual_stats(np.zeros((4,64,64),np.float32));self.assertEqual(zero['MSE'],0);self.assertIsNone(zero['skew'])
        a=np.broadcast_to(np.arange(64,dtype=np.float32),(4,64,64)).copy()
        self.assertAlmostEqual(c.residual_stats(a)['adjacent_horizontal_correlation'],1)
        with self.assertRaises(ValueError):c.residual_stats(np.zeros((4,32,32)))
    def test_prompts_only_actual_received_text(self):
        cap,joint=c.prompts('a "red" fish')
        self.assertNotIn('red',cap);self.assertIn('a \\"red\\" fish',joint)
        self.assertTrue(cap.startswith('Question:'))
    def test_no_training_or_silent_config_cli(self):
        from docci_evidence_v23.run import parser
        with self.assertRaises(SystemExit):parser().parse_args(['train','out'])
    def test_no_expansion_without_actual_smoke(self):
        from docci_evidence_v23.run import expansion_gate
        a=SimpleNamespace(action='prepare',profile='quick',smoke_root=None)
        with self.assertRaisesRegex(ValueError,'smoke'):expansion_gate(a)
    def test_real_four_MCS_constellations_roundtrip_energy(self):
        from docci_pretrained_checks_v22.digital import PROFILES,encode,receive,capacity,QAM_POINTS
        self.assertEqual(len(set(QAM_POINTS)),16)
        for p in PROFILES:
            data=bytes(range(32));x,meta,private=encode(data,0,p);rx=receive(x,p)
            self.assertEqual(len(x),4128);self.assertAlmostEqual(float(np.mean(abs(x)**2)),1,places=6)
            self.assertEqual(rx['payload'],data);self.assertTrue(rx['packet_crc']);self.assertGreater(capacity(p),len(data))
    def test_real_synthetic_JPEG_receiver_verifier_and_report(self):
        from docci_evidence_v23.digital import run
        p=fixture(self.root/'p');d=self.root/'digital';d.mkdir()
        run(SimpleNamespace(prepared=p),d,lambda:None)
        self.assertEqual(report.verify(d,p)['rows'],24)
        result=report.render(p,[d],self.root/'display')
        self.assertFalse(result['semantic_success']);self.assertEqual(result['facts'],'PENDING')
        self.assertIn('jpeg_qpsk12',(self.root/'display/index.html').read_text())
    def test_score_verifier_recomputes_png(self):
        p=fixture(self.root/'p');s=self.root/'s';s.mkdir()
        data=[]
        for r in c.rows(p/'rows.jsonl'):
            data.append(dict(example_id=r['example_id'],snr_db=r['snr_db'],repeat=0,control='continuous',
                image_sha256=r['received_image_sha256'],MSE=0.,psnr=None,identical=True,LPIPS=0.))
        with (s/'rows.jsonl').open('x') as f:
            for r in data:c.line(f,r)
        c.atomic(s/'identity.json',dict(prepared_sha256=c.sha(p/'COMPLETE.json'),target_sha256=c.sha(p/'COMPLETE.json')))
        c.complete(s,'score',dict(rows=6));self.assertEqual(report.verify(s,p,p)['rows'],6)
        data[0]['MSE']=.1;(s/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in data));reseal(s)
        with self.assertRaisesRegex(ValueError,'MSE'):report.verify(s,p,p)
    def test_import_old_producer_schema_without_original_changes(self):
        from docci_pretrained_checks_v22.core import complete
        from docci_evidence_v23.run import import_v22
        old=fixture(self.root/'old');idx=c.read(old/'index.json');idx.update(profile='smoke',repeats=1)
        (old/'index.json').write_text(json.dumps(idx));(old/'COMPLETE.json').unlink();complete(old,'prepare',dict(rows=6))
        before={x.name:c.sha(x) for x in old.iterdir() if x.is_file()}
        new=self.root/'new';new.mkdir();import_v22(SimpleNamespace(from_v22=old),new)
        self.assertEqual(report.verify(new)['rows'],6);self.assertEqual(c.prepared(new)[0]['noise_family'],'v22')
        self.assertEqual(before,{x.name:c.sha(x) for x in old.iterdir() if x.is_file()})
    def test_caption_full_six_control_matrix_and_wrong_text_prompt(self):
        p=fixture(self.root/'p');index,rs,ex=c.prepared(p);by={c.case_key(r):r for r in rs};s=self.root/'caption';s.mkdir();data=[]
        for r in rs:
            donor=by[r['donor_id'],r['snr_db'],0];cap,joint=c.prompts(r['received_caption'])
            for control in c.CONTROLS:
                prompt=None if control=='received_text' else cap if control=='caption_only' else c.prompts(donor['received_caption'])[1] if control=='joint_wrong_text' else joint
                im=None
                if control=='joint_blank_image':im=np.full((512,512,3),128,np.uint8)
                elif control!='received_text':im=np.asarray(Image.open(p/(donor if control=='joint_wrong_image' else r)['received_image']).convert('RGB'))
                output=r['received_caption']
                data.append(dict(example_id=r['example_id'],snr_db=r['snr_db'],repeat=0,control=control,output_text=output,
                    output_sha256=c.digest(output.encode()),received_caption_sha256=r['received_caption_sha256'],source_text_at_RX=False,total_symbols=13376,
                    prompt_sha256=None if prompt is None else c.digest(prompt.encode()),conditioning_image_sha256=None if im is None else c.digest(im.tobytes()),
                    telemetry={} if im is None else dict(do_sample=False,maximum_new_tokens=512,capacity_stop=False,natural_EOS=True)))
        with (s/'rows.jsonl').open('x') as f:
            for r in data:c.line(f,r)
        c.atomic(s/'identity.json',dict(prepared_sha256=c.sha(p/'COMPLETE.json')));c.complete(s,'caption',dict(rows=36))
        self.assertEqual(report.verify(s,p)['rows'],36)
        data[-1]['prompt_sha256']=c.digest(joint.encode());(s/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in data));reseal(s)
        with self.assertRaisesRegex(ValueError,'Caption'):report.verify(s,p)
    def test_private_collector_portable_hashes(self):
        from docci_evidence_v23.collect import collect
        p=fixture(self.root/'p');dest=self.root/'records.tar.gz';before=c.sha(p/'COMPLETE.json')
        collect(dest,p,[],.05)
        self.assertEqual(before,c.sha(p/'COMPLETE.json'));self.assertTrue(dest.is_file())
        with self.assertRaises(FileExistsError):collect(dest,p,[],.05)
    def test_mixed_pixel_packet_and_LPIPS_groups_never_collide(self):
        from docci_evidence_v23.digital import run
        p=fixture(self.root/'p');d=self.root/'d';d.mkdir();run(SimpleNamespace(prepared=p),d,lambda:None)
        s=self.root/'s';s.mkdir()
        with (s/'rows.jsonl').open('x') as f:
            for r in c.rows(d/'rows.jsonl'):
                value={k:r[k] for k in ('example_id','snr_db','repeat','control','image_sha256','MSE','psnr','identical')}
                value['LPIPS']=.5;c.line(f,value)
        c.atomic(s/'identity.json',dict(prepared_sha256=c.sha(p/'COMPLETE.json'),target_sha256=c.sha(d/'COMPLETE.json')))
        c.complete(s,'score',dict(rows=24))
        result=report.render(p,[d,s],self.root/'display')
        self.assertIn('LPIPS',result['groups']['score:d/jpeg_qpsk12 / 0']['mean'])
        self.assertIn('packet_success',result['groups']['digital:d/jpeg_qpsk12 / 0']['mean'])
        self.assertTrue((self.root/'display/psnr_curve.svg').is_file())
    def test_source_manifest_identity_tamper_rejected(self):
        p=fixture(self.root/'p');v=c.read(p/'COMPLETE.json');v['source_manifest_sha256']='0'*64
        (p/'COMPLETE.json').write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,'identity'):report.verify(p)
    def test_calibration_arrays_pixels_and_light_readonly_verification(self):
        root=self.root/'calibration';root.mkdir();(root/'arrays').mkdir();(root/'images').mkdir()
        im=root/'images/a.png';Image.new('RGB',(512,512),(50,80,90)).save(im);zero=np.zeros((4,64,64),np.float32)
        image=np.asarray(Image.open(im));metric=c.metric(image,image,[0,0,512,512])
        data=[]
        for i in range(8):
            for snr in c.CAPTION_SNR:
                name='arrays/'+c.case_name('train_'+str(i),snr,0)+'.npz'
                with (root/name).open('xb') as f:np.savez(f,source_latent=zero,received_latent=zero,clean_received_latent=zero,reencoded_latent=zero)
                data.append(dict(example_id='train_'+str(i),snr_db=snr,repeat=0,cohort='train',
                    source_latent_used_only_after_receive_for_diagnosis=True,array_file=name,array_sha256=c.sha(root/name),
                    source_image='images/a.png',source_image_sha256=c.sha(im),rectangle=[0,0,512,512],
                    direct_image='images/a.png',direct_image_sha256=c.sha(im),roundtrip_image='images/a.png',roundtrip_image_sha256=c.sha(im),
                    direct=metric,roundtrip=metric,source_error=c.residual_stats(zero),incremental_channel_error=c.residual_stats(zero),roundtrip_error=c.residual_stats(zero)))
        with (root/'rows.jsonl').open('x') as f:
            for r in data:c.line(f,r)
        c.atomic(root/'identity.json',dict(count=8,timestep_mapping=None));c.complete(root,'calibrate',dict(rows=24))
        self.assertTrue(report.verify(root)['arrays_verified'])
        import shutil
        light=self.root/'light';shutil.copytree(root,light,ignore=shutil.ignore_patterns('*.npz'))
        self.assertFalse(report.verify(light,arrays=False)['arrays_verified'])
        data[0]['direct']['MSE']=.5
        (light/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in data));reseal(light)
        with self.assertRaisesRegex(ValueError,'PNG metrics'):report.verify(light,arrays=False)
    def test_owned_timeout_stops_only_its_child_and_records_PID(self):
        from docci_evidence_v23.launch import invoke
        import os
        owners=[]
        with self.assertRaises(TimeoutError):
            invoke([sys.executable,'-c','import time; time.sleep(20)'],self.root/'owned.log',dict(os.environ),time.monotonic()+.1,owners)
        self.assertEqual(len(owners),1)
        with self.assertRaises(ProcessLookupError):os.kill(owners[0],0)
    def test_downloader_default_plan_has_no_network_or_write(self):
        from docci_evidence_v23 import download_caption as d
        with patch.object(sys,'argv',['download_caption.py','--output',str(self.root/'model')]),patch.object(d.urllib.request,'urlopen',side_effect=AssertionError('network')):
            d.main()
        self.assertFalse((self.root/'model').exists());self.assertGreater(sum(v[0] for v in d.FILES.values()),15e9)
    def test_downloader_git_and_LFS_hashes(self):
        from docci_evidence_v23.download_caption import verify_file
        p=self.root/'blob';p.write_bytes(b'abc');git=hashlib.sha1(b'blob 3\0abc').hexdigest();h=hashlib.sha256(b'abc').hexdigest()
        self.assertEqual(verify_file(p,(3,'git',git)),h);self.assertEqual(verify_file(p,(3,'lfs',h)),h)
        with self.assertRaises(ValueError):verify_file(p,(3,'lfs','0'*64))
    def test_downloader_resumes_only_exact_range_and_empty_partial(self):
        from docci_evidence_v23.download_caption import download_one
        class Response(io.BytesIO):
            status=206;headers={'Content-Range':'bytes 2-4/5'}
        partial=self.root/'file.part';partial.write_bytes(b'ab');h=hashlib.sha256(b'abcde').hexdigest()
        self.assertEqual(download_one(self.root,'file',(5,'lfs',h),time.monotonic()+5,lambda *a,**kw:Response(b'cde')),h)
        self.assertEqual((self.root/'file').read_bytes(),b'abcde')
        (self.root/'empty.part').touch()
        self.assertEqual(download_one(self.root,'empty',(5,'lfs',h),time.monotonic()+5,lambda *a,**kw:Response(b'abcde')),h)
    def test_downloader_rejects_ignored_range(self):
        from docci_evidence_v23.download_caption import download_one
        class Response(io.BytesIO):status=200;headers={}
        (self.root/'file.part').write_bytes(b'ab')
        with self.assertRaisesRegex(ValueError,'resume range'):
            download_one(self.root,'file',(5,'lfs','0'*64),time.monotonic()+5,lambda *a,**kw:Response(b'abcde'))

if __name__=='__main__':unittest.main(verbosity=2)
