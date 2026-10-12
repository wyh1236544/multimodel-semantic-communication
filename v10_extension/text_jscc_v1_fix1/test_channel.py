from pathlib import Path
import sys
import unittest
import numpy as np
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from text_jscc_v1_fix1.channel import short_resource,public_payload,transport,wave,fidelity_summary
from fso_fluctuating_v2.core import load,paths,fit_operator,as_complex,apply_operator
from optical_realism_v3.spec import sampled_count


class Channel(unittest.TestCase):
    def test_short_resource_charges_two_bursts(self):
        c=load(); res=short_resource(c)
        self.assertEqual(res['payload_complex_symbols'],384); self.assertEqual(res['fs_hz'],64e9)
        self.assertGreater(res['slot_samples'],384*4)
        self.assertAlmostEqual(res['slot_seconds'],res['slot_samples']/64e9)
    def test_public_payload_normalized_shape(self):
        x=public_payload(7); self.assertEqual(x.shape,(48,16)); self.assertAlmostEqual(float(np.mean(x*x)),1.,places=6)
    def test_short_operator_fit(self):
        x=public_payload(8); coefficient=np.zeros(11,complex); coefficient[2]=.8+.1j
        y=apply_operator(as_complex(x),coefficient,np.zeros(384,complex))
        real=np.stack((y.real,y.imag),-1).reshape(48,16)*2**.5
        fitted,res=fit_operator(x,real); self.assertLess(float(np.max(abs(res))),1e-6)
        self.assertLess(float(np.max(abs(fitted-coefficient))),1e-6)
    def test_wrong_legacy_size_and_repeats_refused(self):
        c=load(); cond=c['conditions'][0]; path=paths(c,cond['weather'],1,'test-short')[0]
        t=transport(c,cond,path,1)
        with self.assertRaises(ValueError): t.continuous(np.zeros((288,16)),1)
        with self.assertRaises(ValueError): t.continuous(public_payload(1),2)
    def test_real_numpy_short_wave_cost_and_finite(self):
        c=load(); cond=c['conditions'][0]; path=paths(c,cond['weather'],1,'test-short-frame')[0]
        y,rel,snr,stats,t=wave(c,cond,path,23,public_payload(6))
        self.assertEqual(y.shape,(48,16)); self.assertTrue(np.isfinite(y).all())
        self.assertEqual(len(t.events),2); self.assertEqual(sum(e['data_symbols'] for e in t.events),384)
        used=sum(e['awg_samples']+e['guard_samples'] for e in t.events)
        self.assertEqual(stats['used_samples'],used); self.assertEqual(used,short_resource(c)['slot_samples'])
        self.assertEqual(stats['analog_repeats'],1); self.assertIsNone(stats['pre_fec_ber'])
    def test_missing_or_unavailable_holdout_cannot_pass(self):
        c=load(); self.assertFalse(fidelity_summary([],c)['passed'])
        rows=[dict(condition=q['id'],available_fit=False,available_fresh=False,actual_evm2=1.,predicted_evm2=1.) for q in c['conditions'] for _ in range(4)]
        self.assertFalse(fidelity_summary(rows,c)['passed'])


if __name__=='__main__': unittest.main(verbosity=2)
