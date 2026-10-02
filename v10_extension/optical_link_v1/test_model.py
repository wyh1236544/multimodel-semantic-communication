"""Server Torch CPU adapter contract; does not need VAE/CLIP/data/checkpoints."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from optical_link_v1.model import OpticalFrame
from optical_link_v1.config import load


class FakeTransport:
    def __init__(self,*args):
        self.events=[{'estimated_snr_db':12.5}]
    def probe(self): return 12.5


class AdapterTests(unittest.TestCase):
    def test_only_pilot_snr_reaches_neural_gate(self):
        with patch('optical_link_v1.model.OpticalTransport',FakeTransport):
            frame=OpticalFrame({},99.,1,torch.device('cpu'),context=dict(config={},field_seed=1,noise_seed=2),fixed_mode=0)
        self.assertAlmostEqual(float(frame.routing_snr[0]),12.5)
        with patch('optical_link_v1.model.send_tokens',return_value=(np.zeros((288,16),np.float32),dict(feature_object_crc_accepted=False,physical_symbol_snr_db=80))):
            with torch.no_grad(): restored,mask=frame.payload(torch.ones(1,288,16),None,None)
        self.assertTrue(torch.all(restored==0)); self.assertTrue(torch.all(mask==0))
        self.assertFalse(bool(frame.header_ok[0]))
        self.assertAlmostEqual(float(frame.routing_snr[0]),12.5)

    def test_no_accidental_training(self):
        with patch('optical_link_v1.model.OpticalTransport',FakeTransport):
            frame=OpticalFrame({},0.,1,torch.device('cpu'),context=dict(config={},field_seed=1,noise_seed=2),fixed_mode=0)
        with self.assertRaises(ValueError): frame.payload(torch.ones(1,288,16,requires_grad=True),None,None)

    def test_real_numpy_forward_cpu(self):
        config=load()[1]['good']
        frame=OpticalFrame({},0.,1,torch.device('cpu'),context=dict(config=config,field_seed=1,noise_seed=2),fixed_mode=0)
        x=torch.zeros(1,288,16)
        with torch.no_grad(): y,mask=frame.payload(x,None,None)
        self.assertEqual(tuple(y.shape),(1,288,16)); self.assertTrue(torch.isfinite(y).all())
        self.assertEqual(frame.last_stats['raw_utf8_wire_bytes'],0)
        self.assertIn('power_stages',frame.last_stats)
        self.assertIn('received_optical_signal_dbm',frame.last_stats)
        self.assertIsNone(frame.last_stats['nominal_snr_db'])


if __name__ == '__main__': unittest.main(verbosity=2)
