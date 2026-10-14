"""Local NumPy contracts; neither random Torch tests nor syntax prove quality."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
import numpy as np
from PIL import Image
from docci_modal_v9.core import load,geometry,initialization,schedule,gate,metrics
from docci_modal_v9.data import token_plan,letterbox
from docci_modal_v9.channel import resource,send,physical_config
from fso_fluctuating_v2.core import paths


class Tokenizer:
    bos_token_id=0; eos_token_id=2; pad_token_id=1
    def __call__(self,text,**kwargs):
        assert kwargs['truncation'] is False
        return dict(input_ids=[3]*len(text.split()))


class Contracts(unittest.TestCase):
    def test_config_and_real_private_frame(self):
        c=load(); self.assertEqual(geometry(c,'text')['payload_complex_symbols'],1024)
        self.assertEqual(geometry(c,'image')['payload_complex_symbols'],4096)
        c256=load(image_size=256); self.assertEqual(geometry(c256,'image')['payload_complex_symbols'],1024)
        rt=resource(c,'text'); ri=resource(c,'image'); self.assertLess(rt['scheduled_seconds'],ri['scheduled_seconds'])
        self.assertEqual(rt['analog_repeats'],1); self.assertEqual(rt['common_reals'],0)
        self.assertAlmostEqual(rt['scheduled_seconds'],rt['slot_samples']/64e9)
    def test_no_truncation_or_CLIP_77_dependency(self):
        plan=token_plan('x '*470,Tokenizer(),512); self.assertEqual(plan['source_tokens'],472)
        self.assertEqual(len(plan['ids']),512); self.assertFalse(plan['input_truncated'])
        with self.assertRaises(ValueError): token_plan('x '*511,Tokenizer(),512)
    def test_native_view_not_upscaled_128(self):
        image=Image.fromarray(np.full((700,1200,3),117,np.uint8)); view,rect=letterbox(image,512)
        self.assertEqual(view.size,(512,512)); self.assertEqual(tuple(rect),(0,106,512,405))
    def test_optimizer_updates_are_separate_from_accumulation(self):
        c=load(); p=schedule(c,'diagnostic','image')
        self.assertEqual(p['diagnostic_updates'],256); self.assertEqual(c['image_grad_accum'],2)
        self.assertEqual(schedule(c,'diagnostic','text')['diagnostic_updates'],1200)
        self.assertEqual(schedule(c,'diagnostic','text')['clean_updates'],0)
    def test_historical_missing_never_fresh_fallback(self):
        info=initialization('text','fresh'); self.assertEqual(info['mode'],'explicit_fresh')
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'last.pth'; file.write_bytes(b'not a pickle'); sha=hashlib.sha256(file.read_bytes()).hexdigest()
            with self.assertRaises(ValueError): initialization('text',str(file))
            (file.parent/'TRAIN_COMPLETE.json').write_text(json.dumps(dict(checkpoint_sha256=sha)))
            self.assertEqual(initialization('text',str(file),sha)['sha256'],sha)
            with self.assertRaises(ValueError): initialization('text',str(file),'0'*64)
    def test_failed_text_quality_is_not_engineering_exception(self):
        c=load(); rows=[]
        for control in ('matched','zero','shuffled'):
            rows.append(dict(control=control,input_truncated=False,stopped_by_capacity=True,**metrics('the cat','the cat','a '*511)))
        self.assertFalse(gate(rows,'text',c,True)['passed'])
    def test_physical_private_short_frame_is_charged(self):
        c=load(); cfg=physical_config(); condition=cfg['conditions'][0]; draw=paths(cfg,condition['weather'],1,'v9-unit')[0]
        wire=np.ones(geometry(c,'text')['total_reals'],np.float32)
        received,stats,_=send(c,'text',cfg,condition,draw,123,wire)
        self.assertEqual(received.shape,wire.shape); self.assertTrue(np.isfinite(received).all())
        self.assertEqual(stats['payload_complex_symbols'],1024); self.assertEqual(stats['raw_utf8_wire_bytes'],0)
        self.assertIsNone(stats['pre_fec_ber']); self.assertEqual(stats['resources']['analog_repeats'],1)
    def test_python38_source_parse(self):
        for file in Path(__file__).parent.glob('*.py'): ast.parse(file.read_text(),filename=str(file),feature_version=(3,8))


if __name__=='__main__': unittest.main(verbosity=2)
