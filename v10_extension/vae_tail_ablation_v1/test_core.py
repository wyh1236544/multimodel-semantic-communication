"""Non-Torch contracts: bounded configuration, initialization, paired design."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from vae_tail_ablation_v1.core import (load,tail_prefixes,check_bank,
    check_parent,check_history_pair,visual_positions,TEMPLATE)
from fso_fluctuating_v2.core import load as channel_load


class Contracts(unittest.TestCase):
    def test_precise_tail_scope(self):
        prefixes=tail_prefixes(4)
        self.assertEqual(prefixes[0],'base.vae_wrapper.vae.decoder.up_blocks.3.')
        for name in ('base.vae_wrapper.vae.encoder.x','base.vae_wrapper.vae.decoder.up_blocks.2.x','base.vae_wrapper.vae.decoder.up_blocks.30.x','base.text_encoder.x'):
            self.assertFalse(name.startswith(prefixes))
        for value in (0,True,9):
            with self.assertRaises(ValueError): tail_prefixes(value)

    def test_default_budget_and_channel(self):
        c=load(); check_bank(c,dict(smoke=False,config=channel_load()))
        self.assertEqual(c['train_samples']*c['epochs'],3000)
        self.assertEqual(c['val_samples']*(1+len(c['conditions'])*c['draws'])*3,2496)
        self.assertEqual(visual_positions(c),[0,1,32,33])
        c['sps']=2
        with self.assertRaisesRegex(ValueError,'configuration differs'): check_bank(c,dict(smoke=False,config=channel_load()))
        with self.assertRaises(ValueError): check_bank(load(),dict(smoke=True,config=channel_load()))

    def test_bad_configuration(self):
        base=json.loads(TEMPLATE.read_text())
        for change in (dict(epochs=0),dict(tail_lr=float('nan')),dict(primary_validation_offset=64),dict(warmup_optimizer_steps=500,train_samples=8,epochs=1),dict(extra=True)):
            with tempfile.TemporaryDirectory(dir=str(Path(__file__).parent)) as temp:
                p=Path(temp)/'bad.json'; p.write_text(json.dumps(dict(base,**change)))
                with self.assertRaises(ValueError): load(p)

    def test_exact_completed_parent(self):
        hashes=dict(base_checkpoint_sha256='base',a_delta_sha256='a',split_sha256='split')
        identity=dict(hashes,calibration_sha256='bank',init_awgn_delta_sha256='m2')
        payload=dict(identity,version='V10-FSO-fluctuating-v2',arm='B2_fso_ft',steps=6000,optimizer_steps=3000)
        record=dict(arm='B2_fso_ft',steps=6000,optimizer_steps=3000)
        design=dict(smoke=False,train_forward_steps=6000,optimizer_steps=3000)
        check_parent(payload,record,design,identity,hashes,'bank')
        for update in (dict(arm='B1_awgn_ft'),dict(steps=5),dict(split_sha256='other'),dict(calibration_sha256='other')):
            with self.assertRaises(ValueError): check_parent(dict(payload,**update),record,design,identity,hashes,'bank')
        with self.assertRaises(ValueError): check_parent(payload,record,dict(design,smoke=True),identity,hashes,'bank')

    def test_schedule_identity(self):
        a=[dict(epoch=1,image_id=2,bank_index=3,anchor=False,task_loss_applied=True,network_seed=4,noise_seed=5)]
        check_history_pair(a,copy.deepcopy(a))
        for key in a[0]:
            b=copy.deepcopy(a); b[0][key]=999
            with self.assertRaises(ValueError): check_history_pair(a,b)


if __name__=='__main__': unittest.main(verbosity=2)
