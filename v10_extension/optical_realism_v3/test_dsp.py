"""NumPy numerical contracts; never imply COCO/GPU or hardware validation."""
import copy
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from physical_link.waveform import Pulse
from optical_realism_v3.hardware import load, sample_clock, extra_converter_noise
from optical_realism_v3.dsp import frame_symbols, receive, layout
from optical_realism_v3.spec import load_plan, resources, sampled_count
from optical_link_v1.config import load as optical
from optical_realism_v3.transport import RealisticTransport


class DSPTests(unittest.TestCase):
    def test_fractional_timing_and_cfo_for_continuous_not_qam(self):
        pulse = Pulse(64e9,16e9,.1,32)
        for ppm in (-60.,20.,80.):
            h = load(); h['adc_clock_ppm'] = ppm
            rng = np.random.default_rng(31)
            source = (rng.standard_normal(12000)+1j*rng.standard_normal(12000))/2**.5
            frame,*_ = frame_symbols(source,h)
            wave = pulse.shape(frame)*2
            wave *= np.exp(1j*2*np.pi*50e6*np.arange(len(wave))/pulse.fs)
            wave,_ = sample_clock(wave,h,pulse.fs,rng)
            received,_,report,*_ = receive(np.pad(wave,(7,256)),pulse,len(source),h)
            self.assertTrue(report['receiver_locked'],report)
            self.assertLess(abs(report['estimated_clock_ppm']-ppm),3.)
            self.assertLess(abs(report['estimated_cfo_hz']-50e6),1e6)
            self.assertLess(np.mean(abs(received-source)**2),.001)

    def test_receiver_truth_fields_have_no_effect(self):
        h=load(); p=Pulse(64e9,16e9,.1,32)
        source=np.random.default_rng(2).standard_normal(3000).astype(complex)
        f,*_=frame_symbols(source,h); obs=np.pad(p.shape(f)*2,(7,256))
        a=receive(obs,p,len(source),h)
        changed=dict(h,adc_clock_ppm=-99,frequency_offset_hz=-99e6,fractional_delay_samples=-1)
        b=receive(obs,p,len(source),changed)
        np.testing.assert_array_equal(a[0],b[0]); self.assertEqual(a[2],b[2])

    def test_no_signal_rejected_and_zero_filled(self):
        h=load(); p=Pulse(64e9,16e9,.1,32)
        received,_,d,*_=receive(np.zeros(20000,complex),p,3000,h)
        self.assertFalse(d['receiver_locked']); self.assertTrue(np.all(received==0))
        rng=np.random.default_rng(13)
        _,_,d,*_=receive(rng.normal(size=20000)+1j*rng.normal(size=20000),p,3000,h)
        self.assertFalse(d['receiver_locked'])

    def test_enob_extra_noise_not_double_counted(self):
        x=np.zeros(100000,complex); rng=np.random.default_rng(5)
        np.testing.assert_array_equal(extra_converter_noise(x,4,8,8,rng),x)
        y=extra_converter_noise(x,4,8,6,rng)
        target=2*(8**2/12)*(2**(-12)-2**(-16))
        self.assertAlmostEqual(float(np.mean(abs(y)**2))/target,1.,delta=.015)

    def test_payload_layout_and_tail_pilots_counted(self):
        h=load(); _,cases,_=optical(); w=cases['good']['waveform']
        w.update(pilot_interval=h['pilot_interval'],pilot_block=h['pilot_block'])
        p=Pulse(64e9,16e9,.1,32)
        for n in (0,1,255,256,257,2304*24):
            f,at,pil,_,_=frame_symbols(np.ones(n),h)
            self.assertEqual(len(at),n)
            self.assertEqual(len(pil),256+int(np.ceil(n/256))*32)
            self.assertEqual(len(p.shape(f))+w['guard_samples'],sampled_count(n,w))

    def test_request_overbudget_repetition_not_silently_reduced(self):
        plan,_=load_plan(repeats=27); _,cases,_=optical()
        with self.assertRaises(ValueError): resources(cases['good'],plan)

    def test_low_power_no_false_digital_acceptance(self):
        plan,_=load_plan(); _,cases,_=optical(); c=cases['good']
        c['waveform']['policy']=plan['digital_policy']; r=resources(c,plan)
        t=RealisticTransport(c,1,2,-60,r['slot_samples'],20,plan['hardware'])
        t.probe(); rx,ok,_=t.transfer(b'payload','text',plan['digital_policy'])
        self.assertFalse(ok); self.assertEqual(rx,b'')
        self.assertTrue(any(not e['receiver_locked'] for e in t.events))
        self.assertFalse(t.events[-1]['crc_accepted'])


if __name__=='__main__': unittest.main(verbosity=2)

