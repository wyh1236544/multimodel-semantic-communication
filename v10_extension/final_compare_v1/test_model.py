"""Torch CPU frame contracts; no checkpoints, VAE, CLIP or dataset required."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from final_compare_v1.model import ReplayFrame


class FrameTests(unittest.TestCase):
    def frame(self, context):
        return ReplayFrame({},99.,1,torch.device('cpu'),context=context,fixed_mode=0,blocks=1)

    def test_capture_after_normalization(self):
        f=self.frame(dict(capture=True,probe_snr_db=40.))
        x=torch.randn(1,288,16)
        with torch.no_grad(): y,r=f.payload(x,None,None)
        np.testing.assert_array_equal(f.captured,x[0].numpy())
        self.assertTrue(torch.equal(x,y)); self.assertTrue(torch.all(r==1))

    def test_receiver_ignores_fresh_sender(self):
        f=self.frame(dict(received=np.zeros((288,16),np.float32),available=False,reliability=0.,probe_snr_db=-8.,stats=dict(physical_symbol_snr_db=80.)))
        with torch.no_grad(): y,r=f.payload(torch.ones(1,288,16)*999,None,None)
        self.assertTrue(torch.all(y==0)); self.assertTrue(torch.all(r==0)); self.assertFalse(bool(f.header_ok[0]))
        self.assertEqual(float(f.routing_snr[0]),-8.)

    def test_pilot_clamp_not_truth(self):
        f=self.frame(dict(received=np.ones((288,16),np.float32),probe_snr_db=-90.))
        self.assertEqual(float(f.routing_snr[0]),-40.)

    def test_no_training(self):
        f=self.frame(dict(capture=True))
        with self.assertRaises(ValueError): f.payload(torch.ones(1,288,16),None,None)

    def test_bad_shape_or_nonfinite(self):
        f=self.frame(dict(received=np.full((288,16),np.nan)))
        with torch.no_grad(), self.assertRaises(ValueError): f.payload(torch.ones(1,288,16),None,None)
        with self.assertRaises(ValueError): self.frame(dict(probe_snr_db=float('nan')))


if __name__=='__main__': unittest.main(verbosity=2)
