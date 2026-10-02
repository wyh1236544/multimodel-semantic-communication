"""NumPy physics/protocol tests; no Torch, pretrained weights or hardware claims."""
import ast
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXT))
import numpy as np
from optical_link_v1.config import load, validate, merge
from optical_link_v1.physics import (dbm_to_w,w_to_dbm,ratio_db,power,amplifier,
    aperture_fraction,draw_path,gamma_parameters,complex_white,filter_enbw,filtered,propagate,
    H_PLANCK,C_LIGHT,Q_ELECTRON,modulate_field)
from optical_link_v1.transport import OpticalTransport,receive_si
from optical_link_v1.wire import send_tokens
from optical_link_v1.report import summarize,write_report
from optical_link_v1.plots import constellation_svg
from token_guidance.wire import quantized
from physical_link.waveform import Pulse,preamble
from physical_link.digital import modulate,demodulate


def case():
    return copy.deepcopy(load()[1]['good'])


class ConfigurationTests(unittest.TestCase):
    def test_default_cases_and_clock(self):
        _,cases,digest=load()
        self.assertEqual(set(cases),{'good','moderate','stress'})
        self.assertEqual(cases['good']['waveform']['fs_hz']/cases['good']['waveform']['sps'],16e9)
        self.assertEqual(len(digest),64)

    def test_unknown_parameter(self):
        with self.assertRaises(ValueError):
            merge(case(),{'path':{'attenution_db_per_km':2}})

    def test_nonfinite_and_negative(self):
        for value in (float('nan'),float('inf'),-1):
            c=case(); c['path']['distance_m']=value
            with self.assertRaises(ValueError): validate(c)

    def test_mutually_exclusive_turbulence(self):
        c=case(); c['turbulence']['alpha']=4
        with self.assertRaises(ValueError): validate(c)
        c=case(); c['turbulence'].update(mode='manual_gamma_gamma',cn2_m_minus_2_over_3=None,alpha=4,beta=2)
        validate(c)

    def test_mutually_exclusive_pointing(self):
        c=case(); c['pointing']['xi']=2
        with self.assertRaises(ValueError): validate(c)

    def test_phase_requires_receiver(self):
        c=case(); c['laser']['linewidth_hz']=1000
        with self.assertRaises(ValueError): validate(c)
        c['receiver']['carrier_recovery']='pilot_cfo_dd'; validate(c)

    def test_osnr_reference_grid(self):
        c=case(); c['receiver']['osnr_reference_bandwidth_nm']=1
        with self.assertRaises(ValueError): validate(c)

    def test_python38_parse(self):
        for p in Path(__file__).parent.glob('*.py'):
            ast.parse(p.read_text(),filename=str(p),feature_version=(3,8))


class PhysicsTests(unittest.TestCase):
    def test_dbm_roundtrip(self):
        self.assertAlmostEqual(dbm_to_w(0),.001)
        for value in (-80,-10,0,10,30): self.assertAlmostEqual(w_to_dbm(dbm_to_w(value)),value)
        self.assertIsNone(w_to_dbm(0))

    def test_circular_aperture_exact_center(self):
        self.assertAlmostEqual(aperture_fraction(.1,.04,0),1-math.exp(-.32),places=12)

    def test_aperture_continuity_and_offset(self):
        center=aperture_fraction(.1,.04,0)
        self.assertAlmostEqual(aperture_fraction(.1,.04,1e-10),center,places=9)
        self.assertLess(aperture_fraction(.1,.04,.1),center)
        self.assertGreater(aperture_fraction(.1,.04,.1),0)
        self.assertAlmostEqual(aperture_fraction(.001,1,.5),1,places=8)

    def test_loss_acts_on_power_and_sqrt_field(self):
        c=case(); c['turbulence']['mode']='disabled'; c['pointing']['jitter_per_axis_rad']=0
        d=draw_path(c,np.random.default_rng(1))
        c['path']['attenuation_db_per_km']+=10
        more=draw_path(c,np.random.default_rng(1))
        self.assertAlmostEqual(more['coupled_irradiance_gain']/d['coupled_irradiance_gain'],.1)
        h=complex(d['field_real'],d['field_imag']); hm=complex(more['field_real'],more['field_imag'])
        self.assertAlmostEqual(abs(hm/h),math.sqrt(.1))
        self.assertAlmostEqual(abs(h)**2,d['coupled_irradiance_gain'])

    def test_gamma_mean_and_variance(self):
        rng=np.random.default_rng(4)
        x=rng.gamma(4,1/4,100000)*rng.gamma(2,1/2,100000)
        self.assertLess(abs(x.mean()-1),.02)
        self.assertLess(abs(x.var()-(1/4+1/2+1/8)),.05)

    def test_cn2_parameters_and_weak_limit(self):
        c=case(); a,b,s=gamma_parameters(c['turbulence'],1000,1.55e-6)
        self.assertAlmostEqual(s,.01990954385892429,places=10)
        self.assertGreater(a,50); self.assertGreater(b,50)
        c['turbulence']['cn2_m_minus_2_over_3']=0
        self.assertEqual(gamma_parameters(c['turbulence'],1000,1.55e-6),(None,None,0.))

    def test_edfa_gain_and_ase_convention(self):
        c=case(); e=c['tx_edfa']; e['gain_db']=20
        result=amplifier(1e-6,0,e,64e9,1.55e-6)
        self.assertAlmostEqual(result['gain_linear'],100)
        expected=(10**(.5)*100-1)/2*H_PLANCK*C_LIGHT/1.55e-6
        self.assertAlmostEqual(result['added_ase_psd_w_per_hz_per_pol']/expected,1)
        self.assertAlmostEqual(result['ase_output_w'],2*expected*64e9)

    def test_edfa_total_limit_includes_noise(self):
        c=case(); e=c['tx_edfa']; e.update(gain_db=40,max_total_output_dbm=0)
        result=amplifier(1e-5,1e-18,e,64e9,1.55e-6)
        self.assertTrue(result['output_limited'])
        self.assertLessEqual(result['signal_output_w']+result['ase_output_w'],.001*(1+1e-12))

    def test_edfa_power_control_and_impossible_input(self):
        c=case(); e=c['tx_edfa']; e.update(control='total_power',gain_db=40,target_total_output_dbm=-10)
        result=amplifier(1e-6,0,e,64e9,1.55e-6)
        self.assertAlmostEqual(result['signal_output_w']+result['ase_output_w'],1e-4,places=12)
        with self.assertRaises(ValueError): amplifier(1,0,e,64e9,1.55e-6)

    def test_noise_psd_sampling_and_filter_bandwidth(self):
        fs,psd=64e9,2e-18
        x=complex_white(np.random.default_rng(5),131072,psd*fs)
        y=filtered(x,fs,10e9)
        self.assertLess(abs(power(x)/(psd*fs)-1),.015)
        self.assertLess(abs(power(y)/(psd*filter_enbw(fs,10e9))-1),.025)

    def test_hybrid_and_thermal_noise_units(self):
        c=case(); n=65536
        c['tx_edfa']['enabled']=c['rx_edfa']['enabled']=False
        c['receiver']['shot_noise']=False
        draw=draw_path(c,np.random.default_rng(6))
        _,_,_,d=propagate(np.ones(n,complex),c,draw,np.random.default_rng(7))
        self.assertLess(d['ideal_hybrid_power_error_w'],1e-14)
        rx=c['receiver']; expected=2*rx['thermal_density_a_per_sqrt_hz_per_pd']**2*filter_enbw(64e9,rx['electrical_bandwidth_hz'])
        self.assertLess(abs(d['receiver_noise_components_a2']['thermal']/expected-1),.04)

    def test_iq_modulator_is_passive_even_with_bias_error(self):
        c=case(); c['modulator'].update(insertion_loss_db=0,bias_i_v=0,bias_q_v=0,quadrature_error_rad=-math.pi/2)
        sig=modulate_field(np.zeros(256,complex),c,64e9)
        self.assertLessEqual(power(sig),dbm_to_w(c['laser']['power_dbm'])*(1+1e-12))

    def test_shot_noise_lo_power_and_one_sided_psd(self):
        c=case(); n=65536
        c['tx_edfa']['enabled']=c['rx_edfa']['enabled']=False
        c['receiver']['thermal_density_a_per_sqrt_hz_per_pd']=0
        draw=draw_path(c,np.random.default_rng(6))
        _,_,_,d=propagate(np.ones(n,complex),c,draw,np.random.default_rng(17))
        rx=c['receiver']; ps=d['power_stages']['detector_input']['signal_w']; plo=dbm_to_w(rx['lo_power_dbm'])
        expected=2*Q_ELECTRON*(rx['responsivity_a_per_w']*(ps+plo)/2+2*rx['dark_current_a'])*filter_enbw(64e9,rx['electrical_bandwidth_hz'])
        self.assertLess(abs(d['receiver_noise_components_a2']['shot']/expected-1),.04)

    def test_noiseless_and_absent_signal_are_not_fake_finite_snr(self):
        self.assertIsNone(ratio_db(1,0)); self.assertIsNone(ratio_db(0,1))
        self.assertAlmostEqual(ratio_db(1e-300,1e-100),-2000)


class ReceiverAndProtocolTests(unittest.TestCase):
    def test_physical_snr_vs_pilot_and_evm(self):
        t=OpticalTransport(case(),1,2); t.probe()
        x=modulate(np.random.default_rng(9).integers(0,2,16000,dtype=np.uint8),16)
        _,e=t.burst(x,'features')
        self.assertLess(abs(e['physical_symbol_snr_db']-e['estimated_snr_db']),2)
        self.assertLess(abs(-10*math.log10(e['evm_squared'])-e['physical_symbol_snr_db']),2)

    def test_pure_awgn_qpsk_ber_reference(self):
        pulse=Pulse(64e9,16e9,.1,32); training=preamble()
        bits=np.random.default_rng(10).integers(0,2,100000,dtype=np.uint8)
        symbols=modulate(bits,4); frame=np.concatenate([training,symbols]); snr=10**(.8)
        wave=pulse.shape(frame)*2
        noisy=np.pad(wave,(7,256))+complex_white(np.random.default_rng(11),len(wave)+263,4/snr)
        eq,_,_=receive_si(noisy,pulse,len(frame),training,'pilot_constant',0)
        hard=(demodulate(eq[256:],4,1/snr)<0).astype(np.uint8)
        ber=float(np.mean(hard!=bits)); theory=.5*math.erfc(math.sqrt(snr/2))
        self.assertLess(abs(ber-theory),.0015)

    def test_no_source_bits_at_receiver(self):
        import inspect
        self.assertEqual(list(inspect.signature(receive_si).parameters),['observed','pulse','count','training','mode','step','order'])

    def test_cfo_is_estimated_from_received_pilot(self):
        pulse=Pulse(64e9,16e9,.1,32); training=preamble(); frame=np.tile(training,4)
        freq=1e6
        wave=pulse.shape(frame)*2; wave*=np.exp(1j*2*math.pi*freq*np.arange(len(wave))/64e9)
        eq,_,e=receive_si(np.pad(wave,(7,256)),pulse,len(frame),training,'pilot_cfo_dd',.02)
        self.assertLess(abs(e['estimated_cfo_hz']-freq),20000)
        self.assertLess(float(np.mean(np.abs(eq[256:]-frame[256:])**2)),.001)

    def test_semantic_roundtrip_and_no_utf8(self):
        t=OpticalTransport(case(),1,2); t.probe()
        original=np.random.default_rng(12).normal(0,.3,(288,16)).astype(np.float32)
        restored,stats=send_tokens(original,t)
        self.assertTrue(stats['feature_object_crc_accepted'])
        np.testing.assert_array_equal(restored,np.asarray(quantized(original)[1],np.float32))
        self.assertEqual(stats['semantic_body_bytes'],4608)
        self.assertEqual(stats['feature_object_wire_bytes'],4622)
        self.assertEqual(stats['raw_utf8_wire_bytes'],0)
        self.assertEqual(stats['packet_attempts'],10)

    def test_failed_crc_erases_whole_object(self):
        c=case(); c['path']['attenuation_db_per_km']=100
        t=OpticalTransport(c,1,2); t.probe()
        restored,stats=send_tokens(np.zeros((288,16),np.float32),t)
        self.assertFalse(stats['feature_object_crc_accepted']); self.assertTrue(np.all(restored==0))
        self.assertEqual(stats['semantic_body_goodput_bps'],0)
        self.assertGreater(stats['packet_error_rate'],0)

    def test_linear_energy_aggregation(self):
        t=OpticalTransport(case(),1,2); t.probe(); t.burst(np.ones(32,complex),'features')
        summary=t.totals()
        self.assertAlmostEqual(summary['physical_symbol_snr_db'],ratio_db(sum(e['snr_signal_energy_a2'] for e in t.events),sum(e['snr_noise_energy_a2'] for e in t.events)))
        self.assertAlmostEqual(summary['estimated_snr_db'],ratio_db(sum(e['pilot_signal_energy_v2'] for e in t.events),sum(e['pilot_noise_energy_v2'] for e in t.events)))
        self.assertEqual(summary['nominal_snr_db'],None)

    def test_packet_snapshot_and_coherence_time(self):
        c=case(); c['turbulence']['coherence_time_s']=1e-12
        t=OpticalTransport(c,1,2)
        with self.assertRaises(ValueError): t.probe()

    def test_report_keeps_failure_and_svg_both_panels(self):
        rows=[dict(case='x',feature_object_crc_accepted=True,psnr=25,semantic_body_bytes=10,scheduled_seconds=1),
              dict(case='x',feature_object_crc_accepted=False,psnr=5,semantic_body_bytes=10,scheduled_seconds=1)]
        result=summarize(rows)['x']
        self.assertEqual(result['psnr_all_mean'],15); self.assertEqual(result['psnr_delivered_mean'],25)
        self.assertEqual(result['object_delivery_rate'],.5)
        self.assertEqual(result['semantic_body_goodput_bps'],40)
        temporary_parent=EXT.parent/'experiment_results/optical_v1_checks'
        temporary_parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(temporary_parent)) as folder:
            write_report(folder,rows,'synthetic')
            target=Path(folder)/'points.svg'
            constellation_svg([dict(purpose='before',policy='16qam_repeat2',tx=[[1,1]],rx=[[0,1]]),dict(purpose='after',policy='16qam_repeat2',tx=[[1,1]],rx=[[1,1]])],target)
            self.assertIn('before',target.read_text()); self.assertIn('after',target.read_text())

    def test_collection_rejects_duplicate_cells_and_packet_mismatch(self):
        from optical_link_v1.collect import verify
        parent=EXT.parent/'experiment_results/optical_v1_checks'; parent.mkdir(parents=True,exist_ok=True)
        t=OpticalTransport(case(),1,2); t.probe(); received,ok,_=t.transfer(b'x','features','16qam_repeat2')
        row=dict(t.totals(),case='good',repeat=0,object_accepted=ok)
        with tempfile.TemporaryDirectory(dir=str(parent)) as name:
            folder=Path(name)
            (folder/'design.json').write_text(json.dumps(dict(optical_config={'cases':{'good':{}}},frames=1,expected_rows=1)))
            (folder/'COMPLETE.json').write_text(json.dumps(dict(rows=1,expected_rows=1)))
            (folder/'samples.jsonl').write_text(json.dumps(row)+'\n')
            packet_rows=[dict(e,case='good',repeat=0) for e in t.events]
            (folder/'packets.jsonl').write_text('\n'.join(json.dumps(e) for e in packet_rows)+'\n')
            self.assertEqual(len(verify(folder)),1)
            packet_rows[1]['pre_errors']+=1
            (folder/'packets.jsonl').write_text('\n'.join(json.dumps(e) for e in packet_rows)+'\n')
            with self.assertRaises(ValueError): verify(folder)
            (folder/'samples.jsonl').write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
            with self.assertRaises(ValueError): verify(folder)


if __name__ == '__main__':
    unittest.main(verbosity=2)
