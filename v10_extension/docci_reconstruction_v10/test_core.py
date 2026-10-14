"""Executable NumPy/frame contracts; no neural effectiveness claim."""
import ast
from pathlib import Path
import sys
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
import numpy as np
from docci_reconstruction_v10.core import load,chunk_plan,chunks,joined_ids,geometry,plan,objective,choose_indices
from docci_reconstruction_v10.channel import resource,send,physical_config
from fso_fluctuating_v2.core import paths


class Contracts(unittest.TestCase):
    def test_full_BPE_no_drop_reorder_or_implicit_word_spaces(self):
        c=load()
        for count in (0,1,61,62,63,124,470,510):
            body=list(range(3,count+3)); ids=chunk_plan(body,c)
            generated=[[2]+r[:list(r).index(2)+1].tolist() for r in ids]
            joined,cap=joined_ids(generated)
            self.assertEqual(joined,body); self.assertFalse(cap); self.assertEqual(ids.shape,(9,64))
        with self.assertRaises(ValueError): chunk_plan([3]*511,c)
    def test_all_nine_empty_segments_charged_without_count_header(self):
        c=load(); a=resource(c,'T0_whole'); b=resource(c,'T1_chunks')
        self.assertEqual(a['payload_complex_symbols'],1024); self.assertEqual(b['payload_complex_symbols'],9216)
        self.assertEqual(b['fixed_text_chunks'],9); self.assertFalse(b['source_count_or_length_given_to_RX'])
        self.assertTrue(b['padded_empty_chunks_also_transmitted']); self.assertEqual(b['analog_repeats'],1)
        self.assertGreater(b['scheduled_seconds'],a['scheduled_seconds']); self.assertAlmostEqual(b['scheduled_seconds'],b['slot_samples']/64e9)
    def test_image_rates_and_objective_not_confused_with_better_compression(self):
        for width,symbols in ((8,4096),(16,8192),(32,16384)):
            c=load(image_width=width); self.assertEqual(geometry(c,'I2_shifted')['payload_complex_symbols'],symbols)
        c=load(); self.assertAlmostEqual(geometry(c,'I0_latent')['canvas_complex_symbols_per_RGB_value'],1/192)
        self.assertAlmostEqual(objective(c,'I0_latent',.16,.02),.162)
        self.assertAlmostEqual(objective(c,'I1_pixel',.16,.02),.0216)
        self.assertEqual(resource(c,'I0_latent')['slot_samples'],resource(c,'I1_pixel')['slot_samples'])
    def test_text_rate_widths_are_explicit(self):
        for width,symbols in ((16,2304),(32,4608),(64,9216)):
            self.assertEqual(geometry(load(text_width=width),'T1_chunks')['payload_complex_symbols'],symbols)
    def test_diverse_training_not_four_repeated_only(self):
        c=load(); p=plan(c,'diagnostic','text'); self.assertEqual(p['train_samples'],64)
        seen=[choose_indices(64,1,u,0,c['seed'])[0] for u in range(64)]
        self.assertEqual(set(seen),set(range(64)))
        accumulation=[choose_indices(64,1,u,m,c['seed'],2)[0] for u in range(32) for m in range(2)]
        self.assertEqual(seen,accumulation)
    def test_EOS_and_empty_padding_not_silently_discarded(self):
        ids,cap=joined_ids([[2,0,5,2,1],[2,0,6,7]])
        self.assertEqual(ids,[5,6,7]); self.assertTrue(cap)
        ids,cap=joined_ids([[2,0,2],[2,0,99,2]])
        self.assertEqual(ids,[99]); self.assertFalse(cap)
    def test_real_combined_chunk_optical_frame(self):
        c=load(); cfg=physical_config(); condition=cfg['conditions'][0]
        draw=paths(cfg,condition['weather'],1,'v10-frame-test')[0]
        wire=np.ones(geometry(c,'T1_chunks')['total_reals'],np.float32)
        rx,stats,_=send(c,'T1_chunks',cfg,condition,draw,123,wire)
        self.assertEqual(rx.shape,(18432,)); self.assertTrue(np.isfinite(rx).all())
        self.assertEqual(stats['payload_complex_symbols'],9216); self.assertEqual(stats['resources'],resource(c,'T1_chunks',cfg))
        self.assertLessEqual(stats['used_samples'],stats['resources']['slot_samples']); self.assertIsNone(stats['pre_fec_ber'])
    def test_python38_source_and_config_rejections(self):
        for file in Path(__file__).parent.glob('*.py'): ast.parse(file.read_text(),filename=str(file),feature_version=(3,8))
        with self.assertRaises(ValueError): load(image_width=7)
        with self.assertRaises(ValueError): load(text_width=8)


if __name__=='__main__': unittest.main(verbosity=2)
