"""Synthetic data/geometry/optical reference tests; not real dataset/GPU gates."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from docci_longclip_joint_v8.core import load,geometry,records,image_index,image_path,box,safe_write,schedule
from docci_longclip_joint_v8.data import token_plan,letterbox
from docci_longclip_joint_v8.channel import resource,physical_config,send
from fso_fluctuating_v2.core import paths
from bart_latent_codec_v4.channel import proxy_numpy


class Tokens:
    bos_token_id=0; eos_token_id=2; pad_token_id=1
    def __call__(self,text,**kw):
        if kw.get('truncation') is not False: raise AssertionError('No truncation')
        return dict(input_ids=list(range(3,3+int(text))))


class ClipTokens(Tokens): bos_token_id=49406; eos_token_id=49407


class Contracts(unittest.TestCase):
    def test_python38_and_vendor_exact(self):
        for file in Path(__file__).parent.glob('*.py'): ast.parse(file.read_text(),feature_version=(3,8))
        data=(Path(__file__).parent/'vendor_longclip.py').read_bytes()
        self.assertEqual(hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest(),'9045ac614db11deb69521b67085d39355162d705')

    def test_no_clip_or_bart_silent_cut(self):
        p=token_plan('500',Tokens(),ClipTokens(),512)
        self.assertEqual(p['length'],502); self.assertFalse(p['input_truncated'])
        self.assertEqual([b['content_tokens'] for b in p['clip_blocks']],[246,246,8])
        restored=[x for b in p['clip_blocks'] for x in b['ids'][1:1+b['content_tokens']]]
        self.assertEqual(restored,list(range(3,503)))
        self.assertTrue(all(len(b['ids'])==248 for b in p['clip_blocks']))
        with self.assertRaises(ValueError): token_plan('511',Tokens(),ClipTokens(),512)

    def test_native_full_frame_corners_letterbox(self):
        im=Image.new('RGB',(1024,768),(255,0,0)); out,rect=letterbox(im,512)
        self.assertEqual(rect,[0,64,512,448]); self.assertEqual(out.size,(512,512))
        self.assertEqual(out.getpixel((0,64)),(255,0,0)); self.assertEqual(out.getpixel((511,447)),(255,0,0))
        self.assertEqual(out.getpixel((256,0)),(0,0,0))

    def test_docci_official_splits_and_no_coco_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); images=root/'images_unpacked/docci_images'; images.mkdir(parents=True)
            rows=[]
            for split,count in (('train',8),('qual_dev',2),('test',1),('qual_test',1)):
                for i in range(count):
                    eid=split+'_'+str(i); rows.append(dict(example_id=eid,split=split,image_file=eid+'.jpg',description='full original description'))
                    if split in ('train','qual_dev'): Image.new('RGB',(32,24)).save(images/(eid+'.jpg'))
            with (root/'docci_descriptions.jsonlines').open('x') as f:
                for r in rows: f.write(json.dumps(r)+'\n')
            tr,val,r=records(root,42); index=image_index(root)
            self.assertEqual(len(tr),8); self.assertEqual(len(val),2); self.assertEqual(r['official_counts']['test'],1)
            for item in tr+val: self.assertTrue(image_path(root,index,item).exists())
            with (root/'docci_descriptions.jsonlines').open('a') as f: f.write(json.dumps(rows[0])+'\n')
            with self.assertRaises(ValueError): records(root,42)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError): records(directory,42)

    def test_image_path_and_write_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'link').symlink_to('/private/tmp',target_is_directory=True)
            with self.assertRaises(ValueError): safe_write(root/'link/file',root)
            with self.assertRaises(ValueError): safe_write(root/'../escape',root)
            with self.assertRaises(ValueError): safe_write(root,root)

    def test_exact_512_budget_r1(self):
        c=load(); g=geometry(c)
        self.assertEqual((g['common_reals'],g['image_reals'],g['text_reals'],g['payload_complex_symbols']),(512,8192,4096,6400))
        small=geometry(load(image_size=256)); self.assertEqual(small['payload_complex_symbols'],3328)
        r=resource(c); self.assertGreater(r['scheduled_samples'],6400*4)
        self.assertAlmostEqual(r['scheduled_seconds'],r['scheduled_samples']/64e9)
        self.assertEqual(schedule(c,'smoke')['text_steps'],8); self.assertFalse(schedule(c,'smoke')['quality_gate_required'])

    def test_proxy_snr_before_fade_and_payload_reference(self):
        c=load(); x=np.ones(geometry(c)['total_reals'],np.float32)
        y,h,_,stats=proxy_numpy(x,c,'fso_moderate',5.,12,34)
        zero,_,_,s=proxy_numpy(np.zeros_like(x),c,'fso_moderate',5.,12,34)
        z=(x[::2]+1j*x[1::2])/np.sqrt(2); expected=np.stack(((z*h).real,(z*h).imag),-1).reshape(-1)*np.sqrt(2)+zero
        np.testing.assert_allclose(y,expected,rtol=1e-5,atol=2e-6)
        self.assertEqual(stats['noise_variance_before_fade'],s['noise_variance_before_fade'])
        self.assertTrue(stats['approximate_training_only']); self.assertEqual(stats['payload_complex_symbols'],6400)

    def test_real_6400_slot_numpy_physical_frame(self):
        c=load(); p=physical_config(); case=p['conditions'][0]; path=paths(p,case['weather'],1,'v8-test-only-not-model')[0]
        rng=np.random.default_rng(7); x=rng.standard_normal(geometry(c)['total_reals']).astype(np.float32); x/=np.sqrt(np.mean(x*x))
        y,s,_=send(c,p,case,path,9,x)
        self.assertEqual(y.shape,x.shape); self.assertTrue(np.isfinite(y).all())
        self.assertEqual(s['used_samples'],s['slot_samples']); self.assertTrue(s['resource_feasible'])
        self.assertEqual(s['analog_repeats'],1); self.assertIsNone(s['pre_fec_ber'])


if __name__=='__main__': unittest.main(verbosity=2)
