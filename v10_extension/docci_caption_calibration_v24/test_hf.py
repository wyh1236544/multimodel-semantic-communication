"""Mandatory server test with a REAL tiny random BLIP-2/OPT; never full weights."""
from pathlib import Path
import sys
import unittest
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]

class ActualHFContracts(unittest.TestCase):
    def setUp(self):
        import torch
        import transformers
        from transformers import Blip2Config, Blip2VisionConfig, Blip2QFormerConfig, OPTConfig, Blip2ForConditionalGeneration
        if torch.__version__.split('+')[0] != '1.12.0' or transformers.__version__ != '4.30.2':
            raise RuntimeError('Run in original server Torch1.12/HF4.30.2; do not install/upgrade')
        torch.set_num_threads(2); torch.manual_seed(42)
        vision = Blip2VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, image_size=16, patch_size=8)
        query = Blip2QFormerConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, encoder_hidden_size=32)
        text = OPTConfig(vocab_size=32, hidden_size=32, word_embed_proj_dim=32, ffn_dim=64,
            num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=256,
            bos_token_id=2, eos_token_id=2, pad_token_id=1, dropout=0., attention_dropout=0.)
        cfg = Blip2Config.from_vision_qformer_text_configs(vision, query, text, num_query_tokens=4)
        self.model = Blip2ForConditionalGeneration(cfg).requires_grad_(False).eval()
        class Processor:
            def __call__(inner, **kw):
                inner.last = kw
                result = {'pixel_values': torch.linspace(-1, 1, 3*16*16).reshape(1, 3, 16, 16)}
                if 'text' in kw:
                    result.update(input_ids=torch.tensor([[2, 10, 11]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
                return result
            def batch_decode(inner, ids, skip_special_tokens):
                values = ids[0].tolist()
                if skip_special_tokens: values = [x for x in values if x not in (1, 2)]
                return [' '.join(str(x) for x in values)]
        self.processor = Processor()
    def test_native_caption_omits_text_real_generate_and_all_hooks(self):
        import torch
        from docci_caption_calibration_v24.runtime import generate_one, state_digest
        before = state_digest(self.model, lambda: None)
        r = generate_one(self.model, self.processor, object(), None, 'caption', 'cpu', torch.float32, lambda: None)
        self.assertNotIn('text', self.processor.last)
        self.assertTrue(r['telemetry']['implicit_BOS'])
        self.assertEqual(len(r['telemetry']['forward_probes']), 4)
        self.assertEqual(len(r['telemetry']['new_token_ids']), len(r['telemetry']['score_trace']))
        self.assertEqual(before, state_digest(self.model, lambda: None))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.model.parameters()))
        self.assertTrue(all(not m._forward_hooks for m in self.model.modules()))
    def test_qa_real_token_prefix_and_hook_cleanup(self):
        import torch
        from docci_caption_calibration_v24.runtime import generate_one
        r = generate_one(self.model, self.processor, object(), 'Question: What is visible? Answer:',
            'qa', 'cpu', torch.float32, lambda: None)
        self.assertEqual(r['telemetry']['input_token_ids'], [2, 10, 11])
        self.assertFalse(r['telemetry']['implicit_BOS'])
        self.assertTrue(all(not m._forward_hooks for m in self.model.modules()))

if __name__ == '__main__': unittest.main(verbosity=2)
