"""V3-specific regressions: negative controls, full QAM and correlated reliability."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from physical_link.waveform import Pulse
from physical_link.digital import modulate,demodulate,fec_encode,fec_decode,scramble
from optical_realism_v3.dsp import frame_symbols,receive
from optical_realism_v3.hardware import load
from optical_realism_v3.reliability import combine
from optical_realism_v3.spec import load_plan,resources
from optical_realism_v3.transport import RealisticTransport
from optical_link_v1.config import load as optical
from optical_realism_v2.dsp import receive as old_receive
from optical_realism_v2.hardware import load as old_hardware


class RepairTests(unittest.TestCase):
    def test_low_snr_same_observation_v2_v3(self):
        h=load(preset="ideal"); p=Pulse(64e9,16e9,.1,32)
        rng=np.random.default_rng(731)
        source=(rng.normal(size=2304)+1j*rng.normal(size=2304))/2**.5
        frame,*_=frame_symbols(np.tile(source,24),h)
        wave=p.shape(frame)*2
        wave*=np.exp(2j*np.pi*50e6*np.arange(len(wave))/p.fs)
        wave+=(rng.normal(size=len(wave))+1j*rng.normal(size=len(wave)))*np.sqrt(10/2)
        observed=np.pad(wave,(7,256))
        new,_,nd,*_=receive(observed,p,24*len(source),h)
        old,_,od,*_=old_receive(observed,p,24*len(source),old_hardware(preset="ideal"))
        restored,stats=combine(new.reshape(24,-1),nd["pilot_residual_variance_before_gate"],
                              nd["pilot_correlated_error_floor"],nd["receiver_locked"],"lmmse")
        self.assertTrue(nd["receiver_locked"],nd)
        self.assertLess(abs(nd["estimated_cfo_hz"]-50e6),6e6)
        self.assertLess(float(np.mean(abs(restored-source)**2)),.35)
        # Fixed observed waveform, not two new noise draws or a tuning search.
        self.assertFalse(od["receiver_locked"],od)
        self.assertTrue(np.all(old==0))
        self.assertLess(stats["analog_reliability"],.99)

    def test_noise_only_multiple_independent_observations(self):
        h=load(); p=Pulse(64e9,16e9,.1,32)
        for draw in range(16):
            rng=np.random.default_rng(8200+draw)
            y,dummy,d,*_=receive(rng.normal(size=20000)+1j*rng.normal(size=20000),p,3000,h)
            self.assertFalse(d["receiver_locked"],(draw,d))
            self.assertTrue(np.all(y==0))
            self.assertTrue(np.isfinite(dummy).all())

    def test_outside_public_cfo_capture_is_not_truth_corrected(self):
        h=load(preset="ideal"); p=Pulse(64e9,16e9,.1,32)
        source=np.random.default_rng(44).normal(size=3000).astype(complex)
        frame,*_=frame_symbols(source,h)
        wave=p.shape(frame)*2*np.exp(2j*np.pi*900e6*np.arange(len(p.shape(frame)))/p.fs)
        _,_,d,*_=receive(np.pad(wave,(7,256)),p,len(source),h)
        self.assertFalse(d["receiver_locked"])
        self.assertLessEqual(abs(d["estimated_cfo_hz"]),h["cfo_capture_hz"])

    def test_full_16qam_block_repetition_soft_roundtrip(self):
        rng=np.random.default_rng(51); bits=rng.integers(0,2,4096,dtype=np.uint8)
        whitened=bits^scramble(len(bits)); coded=fec_encode(whitened)
        onwire=np.tile(coded,2); symbols=modulate(onwire,16)
        self.assertEqual(len(np.unique(symbols)),16)
        llr=demodulate(symbols,16,.05)
        decoded=fec_decode(llr.reshape(2,len(coded)).sum(0),len(bits))^scramble(len(bits))
        np.testing.assert_array_equal(decoded,bits)

    def test_actual_16qam_crc_transfer_and_budget(self):
        plan,_=load_plan(policy="16qam_repeat2")
        _,cases,_=optical(); c=cases["good"]; c["waveform"]["policy"]="16qam_repeat2"
        r=resources(c,plan); t=RealisticTransport(c,1,2,-9,r["slot_samples"],20,plan["hardware"])
        t.probe(); body=bytes(range(256))
        received,ok,_=t.transfer(body,"text","16qam_repeat2")
        self.assertTrue(ok); self.assertEqual(received,body)
        self.assertEqual(t.events[-1]["repetition_mapping"],"whole_coded_block_twice")
        self.assertEqual(t.events[-1]["transmitted_constellation_points"],16)
        self.assertLessEqual(t.totals()["used_samples"],r["slot_samples"])

    def test_shared_error_floor_is_not_divided_by_repetitions(self):
        copies=np.ones((24,2304),complex)
        _,d=combine(copies,.24,.5,True,"lmmse")
        self.assertAlmostEqual(d["analog_white_combined_variance"],.01)
        self.assertAlmostEqual(d["analog_effective_combined_variance"],.5)
        self.assertAlmostEqual(d["analog_reliability"],2/3)

    def test_split_inconsistency_is_observable_without_tx_truth(self):
        copies=np.tile(np.array([1.,-1.])[:,None],(12,100)).astype(complex)
        _,d=combine(copies,.01,0.,True,"lmmse")
        self.assertAlmostEqual(d["analog_split_half_variance"],1.)
        self.assertAlmostEqual(d["analog_reliability"],.5)

    def test_rejected_features_have_zero_reliability(self):
        y,d=combine(np.ones((3,80)),.1,.1,False,"zf")
        self.assertTrue(np.all(y==0)); self.assertEqual(d["analog_reliability"],0.)

    def test_effective_preset_identity(self):
        plan,_=load_plan(preset="ideal")
        self.assertEqual(plan["hardware_preset"],plan["hardware"]["preset"])
        self.assertEqual(plan["hardware"]["frequency_offset_hz"],0.)


if __name__=="__main__": unittest.main(verbosity=2)

