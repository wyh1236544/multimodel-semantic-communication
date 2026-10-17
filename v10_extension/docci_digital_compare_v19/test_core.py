"""Synthetic contracts only, not DOCCI model results or a MATLAB interoperability claim."""
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
EXT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_digital_compare_v19.ldpc import code,BASE12,BASE34
from docci_digital_compare_v19.digital import (encode,receive,evaluate_packet,compressions,decompress,
    bits,byte_data,qpsk,demod,strip_crc,add_crc,allocation)
from docci_digital_compare_v19.common import load,score,ci,trials,safe,PROJECT
from docci_digital_compare_v19.experiment import capacity
from docci_digital_compare_v19.report import matrix,metric_check

class Contracts(unittest.TestCase):
    def test_standard_matrices_systematic_and_independent_syndromes(self):
        rng=np.random.default_rng(41)
        for rate,base in (('12',BASE12),('34',BASE34)):
            c=code(rate);info=rng.integers(0,2,(16,c.k),dtype=np.uint8);cw=c.encode(info)
            self.assertTrue(np.array_equal(cw[:,:c.k],info))
            # Independent circulant/check calculation, not Code.syndrome.
            for br,row in enumerate(base):
                s=np.zeros((len(cw),27),dtype=np.uint8)
                for bc,shift in enumerate(row):
                    if shift>=0:s^=np.roll(cw[:,bc*27:(bc+1)*27],-shift,axis=1)
                self.assertFalse(s.any())
            decoded,diag=c.decode((1-2*cw.astype(float))*15)
            self.assertTrue(np.array_equal(decoded,info));self.assertTrue(all(diag['converged']))
    def test_nonzero_bit_errors_corrected(self):
        for rate in ('12','34'):
            c=code(rate);info=np.random.default_rng(8).integers(0,2,(4,c.k),dtype=np.uint8);cw=c.encode(info)
            llr=(1-2*cw.astype(float))*8;llr[:,[12,142,600]]*=-.1
            got,_=c.decode(llr);self.assertTrue(np.array_equal(got,info))
    def test_qpsk_unit_energy_and_soft_sign(self):
        b=np.array([0,0,0,1,1,0,1,1],dtype=np.uint8);x=qpsk(b)
        self.assertTrue(np.allclose(abs(x)**2,1));self.assertTrue(np.array_equal(demod(x,1)<0,b))
    def test_uncoded_AWGN_BER_theory_and_noise_scaling(self):
        rng=np.random.default_rng(912);b=rng.integers(0,2,200000,dtype=np.uint8);x=qpsk(b);snr=0
        y=x+(rng.standard_normal(len(x))+1j*rng.standard_normal(len(x)))/math.sqrt(2)
        measured=float(np.mean((demod(y,1)<0)!=b));theory=.5*math.erfc(math.sqrt(.5))
        self.assertLess(abs(measured-theory),.003)
    def test_all_compressors_unicode_roundtrip_and_bounds(self):
        text='A red sign says 你好. Café. '*10;raw=text.encode()
        for k,p in enumerate(compressions(text)):
            self.assertEqual(decompress(k,p,len(raw)),text)
            with self.assertRaises((ValueError,Exception)):decompress(k,p,len(raw)-1)
    def test_actual_full_frame_roundtrip_without_TX_hints(self):
        self.assertEqual(list(inspect.signature(receive).parameters),['observed','profile','c'])
        c=load();text='An orange bicycle stands beside a blue wooden fence. '*12
        for profile in c['profiles']:
            tx,a,_=encode(text,profile,c);r=receive(tx,profile,c)
            self.assertEqual(r['received_caption'],text);self.assertTrue(r['packet_crc_pass'])
            self.assertEqual(a['total_symbols'],9216);self.assertEqual(a['body_symbols']+a['header_symbols']+a['pilot_symbols'],9216)
            self.assertGreaterEqual(a['body_repeat_floor'],1)
    def test_crc_erasure_kept_in_denominator(self):
        c=load();text='A short description.';tx,_,_=encode(text,c['profiles'][0],c)
        rx=receive(np.zeros_like(tx),c['profiles'][0],c)
        self.assertFalse(rx['packet_crc_pass']);self.assertEqual(rx['received_caption'],'')
        self.assertEqual(score(text,rx['received_caption'])['normalized_wer'],1)
        with self.assertRaises(ValueError):strip_crc(add_crc(b'abc')[:-1]+b'x')
    def test_channel_repeated_bits_cost_power(self):
        c=load();tx,a,_=encode('text '*200,c['profiles'][0],c)
        original=a['coded_bits'];full=a['body_symbols']*2
        self.assertEqual(a['body_repeat_floor']*original+a['body_extra_repeated_bits'],full)
        self.assertAlmostEqual(float(np.sum(abs(tx)**2)),9216,places=6)
    def test_capacity_is_only_a_reference(self):
        c=load();r=capacity([dict(example_id='a',text='a'*200)],c)
        self.assertFalse(r['actual_transmission_success'])
        zero=next(x for x in r['table'] if x['symbol_budget']==9216 and x['snr_db']==0 and x['assumed_gap_db']==0)
        self.assertEqual(zero['budget_bits'],9216)
    def test_strict_matrix_and_missing_duplicate_detection(self):
        r=dict(arm='a',example_id='x',snr_db=0,repeat=0);expected={('a','x',0,0)}
        matrix([r],expected)
        for rows in ([],[r,r]):
            with self.assertRaises(ValueError):matrix(rows,expected)
    def test_no_free_zero_or_corrupt_metric(self):
        r=dict(received_caption='a red car',**score('a blue car','a red car'))
        metric_check(r,'a blue car');r['normalized_wer']=0
        with self.assertRaises(ValueError):metric_check(r,'a blue car')
    def test_schedule_and_description_bootstrap(self):
        c=load();self.assertEqual(len(list(trials(c))),34);self.assertEqual(len(list(trials(c,negative=True))),18)
        r=ci([0,1],1000);self.assertEqual(r['mean'],.5);self.assertLessEqual(r['CI95'][0],.5);self.assertGreaterEqual(r['CI95'][1],.5)
    def test_paths_no_traversal_or_outside_project(self):
        for p in ('/tmp/not-project','../escape',str(PROJECT)):
            with self.assertRaises((ValueError,RuntimeError)):safe(p)

if __name__=='__main__':unittest.main(verbosity=2)
