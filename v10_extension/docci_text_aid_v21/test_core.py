"""CPU contracts with positive/negative synthetic evidence, not model results."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
import numpy as np
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_text_aid_v21.core import (config,choose,order,impairment,route,
    expected_rows,audit_matrix,results,array_sha,seed,sha,FOLDER,PARENTS,check_anchor)
from docci_text_aid_v21.report import telemetry
from docci_text_guided_image_v20.channel import sample

def fixture(action='diagnose'):
    c=config(action,smoke=True)
    c['selected']={g:[dict(example_id=g+str(i),cohort=g,official_split='train',input_truncated=False)
                      for i in range(2 if g=='dev' or action=='diagnose' else 8)] for g in ('train','dev')}
    data=[]
    for arm,g,eid,s,control in sorted(expected_rows(c),key=str):
        value=seed(42,'v20-AWGN','v21-'+('diagnose' if action=='diagnose' else 'pilot-eval'),0,eid,'image',s,0)
        n=sample(8192,s,value);p=sample(64,s,seed(value,'pilot'))
        pn=float(np.mean(n.astype(np.float64)**2));n0=float(np.mean(p.astype(np.float64)**2))
        stats=dict(noise_seed=value,noise_sha256=array_sha(n),pilot_sha256=array_sha(p),
            nominal_snr_db=s,noise_power=pn,pilot_noise_variance=n0,signal_power=1.,
            estimated_snr_db=None if s is None else -10*math.log10(n0),
            realized_snr_db=None if s is None else -10*math.log10(pn),post_noise_renormalization=False)
        data.append(dict(arm=arm,cohort=g,example_id=eid,snr_db=s,control=control,
            MSE=.01,psnr=20.,ssim=.5,LPIPS=.3,image_stats=stats,source_text_at_RX=False,
            post_noise_renormalization=False,tx_sha256='tx'+eid,observed_sha256='rx'+eid,
            mask_sha256=array_sha(impairment(c,eid))))
    return c,data

class Contracts(unittest.TestCase):
    def test_complete_diagnostic_and_pilot_matrix(self):
        for action,n in (('diagnose',88),('pilot',52)):
            c,rr=fixture(action)
            self.assertEqual(len(audit_matrix(c,rr)),n)
            self.assertFalse(results(c,rr)['semantic_success'])
    def test_duplicate_missing_extra_rows_rejected(self):
        c,rr=fixture()
        for bad in (rr[:-1],rr+[rr[0]],rr+[dict(rr[0],example_id='fake')]):
            with self.assertRaises(ValueError):audit_matrix(c,bad)
    def test_RX_intervention_cannot_change_TX_or_observation(self):
        c,rr=fixture()
        i=next(i for i,r in enumerate(rr) if r['control']=='rx_shuffled')
        for k in ('tx_sha256','observed_sha256'):
            bad=copy.deepcopy(rr);bad[i][k]='changed'
            with self.assertRaises(ValueError):audit_matrix(c,bad)
    def test_TX_intervention_is_allowed_to_change_symbols_not_noise(self):
        c,rr=fixture()
        r=next(r for r in rr if r['control']=='tx_shuffled')
        r['tx_sha256']='different';r['observed_sha256']='different'
        audit_matrix(c,rr)
        r['image_stats']['noise_sha256']='changed'
        with self.assertRaises(ValueError):audit_matrix(c,rr)
    def test_pilot_all_arms_share_actual_image_observation(self):
        c,rr=fixture('pilot')
        rr[0]['observed_sha256']='bad'
        with self.assertRaises(ValueError):audit_matrix(c,rr)
    def test_finite_PSNR_energy_and_no_source_leak(self):
        c,rr=fixture()
        for field,value in (('psnr',19),('LPIPS',float('nan')),('source_text_at_RX',True),('post_noise_renormalization',True)):
            bad=copy.deepcopy(rr);bad[0][field]=value
            with self.assertRaises(ValueError):audit_matrix(c,bad)
        bad=copy.deepcopy(rr);bad[0]['image_stats']['signal_power']=2
        with self.assertRaises(ValueError):audit_matrix(c,bad)
    def test_masks_are_fixed_spatial_and_equal_across_runs(self):
        c=config('pilot','missing25')
        a=impairment(c,'eid').reshape(8,32,32)
        self.assertEqual(float((a==0).mean()),.25)
        self.assertTrue(np.array_equal(a[0],a[7]))
        self.assertEqual(array_sha(a),array_sha(impairment(c,'eid').reshape(8,32,32)))
        self.assertTrue((impairment(config('pilot'),'eid')==1).all())
    def test_same_schedule_donors_bijection_and_no_identity(self):
        c=config('pilot');plan=list(order(c))
        self.assertEqual(len(plan),1024)
        for epoch in range(8):
            pairs=[p for e,b in plan if e==epoch for p in b]
            self.assertEqual(sorted(i for i,j in pairs),list(range(512)))
            self.assertEqual(sorted(j for i,j in pairs),list(range(512)))
            self.assertTrue(all(i!=j for i,j in pairs))
        self.assertEqual(plan,list(order(c)))
    def test_selection_uses_only_original_train_and_dev(self):
        index=dict(records=[dict(example_id='t'+str(i),cohort='train',official_split='train',input_truncated=False) for i in range(8192)]+
                   [dict(example_id='d'+str(i),cohort='dev',official_split='train',input_truncated=False) for i in range(100)])
        c=config('pilot');selected=choose(index,c)
        self.assertEqual((len(selected['train']),len(selected['dev'])),(512,100))
        self.assertEqual(selected,choose(index,c))
        index['records'][0]['official_split']='qual_test'
        with self.assertRaises(ValueError):choose(index,c)
    def test_routes_keep_oracle_source_out_of_RX(self):
        self.assertEqual(route('C_tx_rx_text','rx_clean'),('source','clean'))
        self.assertEqual(route('B_tx_text','tx_shuffled'),('donor','self'))
        self.assertEqual(route('A_self','matched'),('self','self'))
    def test_clean_not_40db_and_deterministic_channel_replay(self):
        c,rr=fixture()
        for r in rr:telemetry(r['image_stats'],r['example_id'],r['snr_db'],'v21-diagnose')
        r=next(r for r in rr if r['snr_db'] is None)
        r['image_stats']['estimated_snr_db']=40
        with self.assertRaises(ValueError):telemetry(r['image_stats'],r['example_id'],None,'v21-diagnose')
    def test_no_synthetic_fixture_can_be_semantic_success(self):
        c,rr=fixture('pilot');out=results(c,rr)
        self.assertFalse(out['numeric_screen_passed']);self.assertFalse(out['semantic_success'])
        self.assertAlmostEqual(out['paired_effects']['0 / correct_over_self']['LPIPS']['mean'],0)
    def test_numerical_screen_needs_practical_gain_but_never_sets_semantic_success(self):
        c,rr=fixture('pilot')
        for r in rr:
            if r['arm']=='correct' and r['control']=='clean':r['LPIPS']=.299
        self.assertFalse(results(c,rr)['numeric_screen_passed'])
        for r in rr:
            if r['arm']=='correct' and r['control']=='clean':r['LPIPS']=.28
        out=results(c,rr)
        self.assertTrue(out['numeric_screen_passed'])
        self.assertFalse(out['semantic_success'])
    def test_python38_and_no_old_source_edits(self):
        for p in FOLDER.glob('*.py'):ast.parse(p.read_text(),feature_version=(3,8))
        from docci_text_guided_image_v20_fix1.compat import source_check
        source_check()
    def test_unknown_protocol_rejected(self):
        with self.assertRaises(ValueError):config('diagnose','missing25')
        with self.assertRaises(ValueError):config('train')
    def test_saved_weight_anchor_requires_exact_wire_and_close_metrics(self):
        actual=dict(tx_sha256='tx',observed_sha256='rx',psnr=20.,ssim=.5,LPIPS=.3,MSE=.01)
        reference=dict(tx_image_sha256='tx',observed_image_sha256='rx',psnr=20.,ssim=.5,LPIPS=.3,MSE=.01)
        check_anchor(actual,reference)
        for name,value in (('tx_sha256','bad'),('LPIPS',.30001)):
            bad=dict(actual);bad[name]=value
            with self.assertRaises(ValueError):check_anchor(bad,reference)

if __name__=='__main__':
    unittest.main(verbosity=2)
