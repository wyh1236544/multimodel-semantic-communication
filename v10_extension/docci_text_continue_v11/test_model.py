"""Mandatory server Torch/HF smoke contracts; missing local dependencies NOT_EXECUTED."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
try:
    import torch
    from transformers import BartConfig,BartForConditionalGeneration
except ImportError:
    if os.environ.get('V11_REQUIRE_TORCH')=='1': raise
    print('NOT_EXECUTED locally: Torch/Transformers missing; mandatory existing-server tests and saved-parent GPU smoke.'); sys.exit(0)
import numpy as np
from docci_reconstruction_v10.core import load as old_load,chunk_plan
from docci_reconstruction_v10.model import TextBranch
from docci_text_continue_v11.core import PROJECT,load,read,state_hash
from docci_text_continue_v11.workflow import fit
from docci_text_continue_v11.evaluate import evaluate
from docci_text_continue_v11.diagnostic import diagnose
torch.set_num_threads(2)


def fixture():
    c=old_load(); c['text_batch_size']=2
    b=BartForConditionalGeneration(BartConfig(vocab_size=128,d_model=128,encoder_layers=1,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=256,decoder_ffn_dim=256,max_position_embeddings=1024,
        bos_token_id=0,eos_token_id=2,pad_token_id=1,decoder_start_token_id=2))
    m=TextBranch(b,c,'T1_chunks'); items=[]
    for i in range(2):
        ids=np.ones(512,np.int64); ids[:4]=[0,3,4,2]; text=('cat sits','dog runs')[i]
        import hashlib
        items.append(dict(example_id='qual_dev_'+str(i),official_split='qual_dev',text=text,text_sha256=hashlib.sha256(text.encode()).hexdigest(),source_tokens=4,
            ids=ids,bart_hidden=np.random.default_rng(i).normal(size=(512,128)).astype(np.float32),
            chunk_ids=chunk_plan([3,4],c),chunk_hidden=np.random.default_rng(i+2).normal(size=(9,64,128)).astype(np.float32)))
    return c,m,items


class Models(unittest.TestCase):
    def test_all_groups_optimizer_checkpoint_and_frozen_BART(self):
        c,m,items=fixture(); frozen={n:p.detach().clone() for n,p in m.bart.named_parameters()}; before=m.saved()
        plan=dict(parent_updates=10000,target_updates=10002,additional_updates=2)
        with tempfile.TemporaryDirectory(dir=str(PROJECT),prefix='v11_train_test_') as d:
            root=Path(d); fit(m,items,c,load(),plan,torch.device('cpu'),root/'train',lambda:None,lambda u:None)
            record=torch.load(str(root/'checkpoint_10002/last.pth'),map_location='cpu')
            self.assertTrue(record['optimizer']['state']); self.assertEqual(record['global_update'],10002)
            self.assertTrue(read(root/'train/learning_audit.json')['passed']); self.assertEqual(state_hash(m.saved()),state_hash(record['state']))
            self.assertTrue(all(p.grad is None and torch.equal(p,frozen[n]) for n,p in m.bart.named_parameters()))
            m.restore(before); self.assertEqual(state_hash(m.saved()),state_hash(before))
    def test_gold_prefix_diagnosis_does_not_change_weights(self):
        c,m,items=fixture(); before=state_hash(m.saved())
        class Tokenizer:
            bos_token_id=0; eos_token_id=2; pad_token_id=1
            def decode(self,ids,**kw): return ' '.join(str(x) for x in ids)
        tok=Tokenizer()
        with tempfile.TemporaryDirectory(dir=str(PROJECT),prefix='v11_prefix_test_') as d:
            root=Path(d); rows=evaluate(m,items,c,tok,torch.device('cpu'),lambda:None,root/'eval','probe','validation')
            result=diagnose(m,items,c,tok,torch.device('cpu'),lambda:None,root/'diagnostic',rows)
            self.assertEqual(len(result),2); self.assertTrue(all(r['gold_prefix_is_privileged_NOT_actual_RX'] for r in result))
            self.assertEqual(state_hash(m.saved()),before); self.assertTrue(all(p.grad is None for p in m.bart.parameters()))
    def test_receiver_signature_unchanged(self):
        import inspect
        _,m,_=fixture(); self.assertEqual(list(inspect.signature(m.receive).parameters),['observed','available','pilot_snr','reliability'])


if __name__=='__main__': unittest.main(verbosity=2)
