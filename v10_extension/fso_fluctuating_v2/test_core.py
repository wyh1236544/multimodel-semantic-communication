"""NumPy contracts; no Torch/data/checkpoints or network."""
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
EXT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXT))
from fso_fluctuating_v2.core import load, gamma_step, paths, fit_operator, apply_operator, as_complex, public_payload, schedule, environment
from fso_fluctuating_v2.report import contrast


class Contracts(unittest.TestCase):
    def test_fixed_resources_and_declared_domain(self):
        c=load(); cases,plan,r=environment(c)
        self.assertEqual(r['analog_repeats'],1); self.assertEqual(r['symbol_rate_baud'],16e9)
        self.assertEqual(r['analog_samples'],13178)
        self.assertIsNone(cases['strong']['turbulence']['cn2_m_minus_2_over_3'])
        self.assertEqual(cases['moderate']['path']['distance_m'],5000.)

    def test_cir_stationary_mean_variance_and_lag(self):
        rng=np.random.default_rng(712); alpha=2.1; rho=.8; x=rng.gamma(alpha,1/alpha)
        values=[]
        for i in range(40000):
            x=gamma_step(x,alpha,rho,rng)
            if i>200: values.append(x)
        self.assertAlmostEqual(float(np.mean(values)),1.,delta=.04)
        self.assertAlmostEqual(float(np.var(values)),1/alpha,delta=.05)
        self.assertAlmostEqual(float(np.corrcoef(values[:-1],values[1:])[0,1]),rho,delta=.02)

    def test_path_pairing_temporal_identity_and_field(self):
        c=load(); a=paths(c,'strong',12,'eval'); b=paths(c,'strong',12,'eval'); t=paths(c,'strong',12,'train')
        self.assertEqual(a,b); self.assertNotEqual(a,t)
        for i,p in enumerate(a):
            self.assertAlmostEqual(p['field_real']**2+p['field_imag']**2,p['coupled_irradiance_gain'])
            self.assertAlmostEqual(p['weather_time_s'],i*c['frame_spacing_s'])
            self.assertLess(p['aperture_pointing_collection'],1.)

    def test_operator_keeps_signal_and_residual(self):
        x=public_payload(2); z=as_complex(x); coeff=np.zeros(11,complex); coeff[2]=.7+.1j; coeff[7]=.05
        y=apply_operator(z,coeff,np.zeros(2304))
        real=np.stack((y.real,y.imag),-1).reshape(288,16)*math.sqrt(2)
        fitted,residual=fit_operator(x,real)
        self.assertTrue(np.allclose(apply_operator(z,fitted,residual),y,atol=1e-6))
        self.assertTrue(np.allclose(fitted,coeff,atol=1e-5))
        self.assertFalse(np.allclose(apply_operator(2*z,fitted,residual),y))

    def test_holdout_is_never_scheduled(self):
        c=load(); records=[]
        for condition in c['conditions']:
            for partition,n in [('train',c['bank_train_frames']),('holdout',c['bank_holdout_frames'])]:
                records.extend(dict(condition=condition['id'],partition=partition) for _ in range(n))
        b=dict(records=records)
        schedule1=[schedule(c,0,i,b) for i in range(400)]
        self.assertEqual(schedule1,[schedule(c,0,i,b) for i in range(400)])
        self.assertTrue(any(i is None for i in schedule1))
        self.assertTrue(all(records[i]['partition']=='train' for i in schedule1 if i is not None))

    def test_fallback_does_not_fake_delivered_gain(self):
        rows=[]
        for i,valid in enumerate((True,True,False,False)):
            for arm in ('B1_awgn_ft','B2_fso_ft'):
                rows.append(dict(arm=arm,image_id=i,condition='low',draw=0,receiver_available=valid,
                    psnr=18. if valid else (8. if arm=='B1_awgn_ft' else 12.)))
        result=contrast(rows,['low'],100)
        self.assertEqual(result['both_available']['mean'],0.)
        self.assertEqual(result['all_frames']['mean'],2.)
        self.assertEqual(result['trial_weighted_contributions']['fallback'],2.)

    def test_invalid_configs_fail_before_running(self):
        c=load()
        for key,value in [('sps',2),('epochs',0),('frame_spacing_s',2.),('unknown',3)]:
            bad=copy.deepcopy(c); bad[key]=value
            with tempfile.TemporaryDirectory() as t:
                f=Path(t)/'config.json'; f.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): load(f)


if __name__=='__main__': unittest.main(verbosity=2)
