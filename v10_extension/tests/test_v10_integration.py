"""Real model/runner code with tiny synthetic backbones, NOT pretrained evidence."""
import copy
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
import torch.nn.functional as F

class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(()))
        self.config = types.SimpleNamespace(scaling_factor=1.)
    def encode(self, x):
        x = F.avg_pool2d(x, 8)
        x = torch.cat([x, x.mean(1, keepdim=True)], 1)
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(sample=lambda: x))
    def decode(self, x):
        return types.SimpleNamespace(sample=torch.tanh(F.interpolate(x[:, :3], scale_factor=8)))
    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()

class FakeBatch(dict):
    def to(self, device):
        return FakeBatch({k: v.to(device) for k,v in self.items()})

class FakeTokenizer:
    eos_token_id = 2
    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()
    def __call__(self, texts, max_length=77, **kw):
        ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i,t in enumerate(texts):
            n = min(len(t.split())+2, max_length)
            ids[i, :n] = torch.arange(n) % 29 + 1
            mask[i, :n] = 1
        return FakeBatch(input_ids=ids, attention_mask=mask)
    def batch_decode(self, ids, **kw):
        return ['synthetic token text' for _ in ids]

class FakeText(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=32, vocab_size=32)
        self.embeddings = nn.Embedding(32, 32)
    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        pooled = (x*attention_mask[..., None]).sum(1)/attention_mask.sum(1, keepdim=True)
        return types.SimpleNamespace(last_hidden_state=x, pooler_output=pooled)

class FakeCLIP:
    @classmethod
    def from_pretrained(cls, *a, **kw):
        return types.SimpleNamespace(text_model=FakeText())

def install_stubs():
    transformers = types.ModuleType('transformers')
    transformers.AutoModel = transformers.CLIPModel = FakeCLIP
    transformers.AutoTokenizer = FakeTokenizer
    diffusers = types.ModuleType('diffusers')
    diffusers.AutoencoderKL = FakeVAE
    # Test-only module injection; never used in production scripts.
    return patch.dict(sys.modules, transformers=transformers, diffusers=diffusers)

def tiny_config():
    return {'model': dict(image_size=128, hidden_dim=32, latent_dim=16,
        vae_model_name_or_path='fake', text_model_name='fake', text_max_length=77,
        freeze_text_projection=True, num_common_tokens=32, num_img_tokens=256, num_txt_tokens=77,
        preserve_spatial_image_tokens=True, explicit_shared_semantics=True,
        symmetric_image_text_decoder=True, gated_text_image_fusion=True,
        shared_encoder_layers=1, shared_decoder_layers=1, num_heads=4,
        image_head_layers=1, text_head_layers=1,
        decoder_text_to_image_cross_attention=True, decoder_text_to_image_layers=1,
        decoder_image_to_text_cross_attention=True, decoder_image_to_text_layers=1,
        common_coarse_regression={'enable': True}, common_base_semantics={'enable': True, 'caption_max_tokens': 24},
        true_common_only_transmission={'enable': True}, decoder_text_snr_gate={'enable': True},
        residual_adapter={'enable': True, 'reliability_gate': {'enable': True}},
        dynamic_token_allocation={'enable': True, 'strategy': 'utility', 'modality_token_budget': 256,
                                  'min_text_tokens': 32, 'max_text_tokens': 192,
                                  'fixed_text_tokens': 32, 'text_selection_strategy': 'valid_prefix_repeat', 'quota_conditioning': True}),
        'channel': {'type': 'awgn'}, 'loss': {'lambda_common_semantic': .5, 'lambda_common_text': .25},
        'v10': {'uep': True, 'adapter': True, 'blocks': 1, 'early_fusion': False}}

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
    @classmethod
    def setUpClass(cls):
        cls.stub = install_stubs()
        cls.stub.start()
        torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls):
        cls.stub.stop()
    def make_model(self):
        from models.joint_image_text_model import JointImageTextSemanticCommunicationModel
        from v10.model import V10Model
        cfg = tiny_config()
        base = JointImageTextSemanticCommunicationModel(cfg['model'])
        return V10Model(base, cfg['v10'])
    def test_full_network_all_channels_backward_and_common_only(self):
        from losses import MultimodalCommunicationLoss
        model = self.make_model()
        model.configure_stage('joint')
        criterion = MultimodalCommunicationLoss(tiny_config()['loss'])
        images = torch.rand(2, 3, 128, 128)
        for profile in ('awgn', 'rayleigh', 'fso_moderate'):
            model.zero_grad(set_to_none=True)
            out = model(images, ['a red car', 'two people in a room'], profile, 10.)
            loss, _ = criterion(out, images)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertEqual(out['text_logits'].shape, (2, 77, 32))
            self.assertIsNone(out['common_coarse_latents'])
        model.eval()
        with torch.no_grad():
            out = model(images, ['a red car', 'two people'], 'fso_moderate', 10., 'common_only', common_task_only=True)
        self.assertEqual(out['z_transmitted'].shape, (2, 32, 16))
        self.assertEqual(out['link_stats']['payload_uses'], 1024)
    def test_strict_checkpoint_roundtrip_and_stage_freezing(self):
        model = self.make_model()
        other = self.make_model()
        other.load_state_dict(model.state_dict(), strict=True)
        model.configure_stage('adapter')
        self.assertTrue(model.base.channel.adapter.net[-1].weight.requires_grad)
        self.assertFalse(model.base.image_head.spatial_projection.weight.requires_grad if hasattr(model.base.image_head, 'spatial_projection') else next(model.base.image_head.parameters()).requires_grad)
        model.configure_stage('joint')
        self.assertTrue(next(model.base.image_head.parameters()).requires_grad)
        self.assertFalse(next(model.base.text_encoder.parameters()).requires_grad)
    def test_zero_gate_rejects_private_text_bypass(self):
        model = self.make_model().eval()
        # With fixed received common/image and zero reliability, altering private
        # received text cannot alter image output. Inject at channel boundary.
        images, texts = torch.rand(2, 3, 128, 128), ['one', 'two']
        model.base.decoder_text_snr_threshold_db = 1e6
        fixed = torch.randn(2, 288, 16)
        def channel_override(latent, *a, **kw):
            return fixed.to(latent)
        model.settings['fixed_mode'] = 0
        model.settings['feedback_snr_db'] = 100
        with torch.no_grad(), patch.object(model.base.channel, 'forward', side_effect=channel_override):
            torch.manual_seed(991)
            a = model(images, texts, 'awgn', 100.)['image']
            fixed[:, 256:] = 100*torch.randn(2, 32, 16)
            torch.manual_seed(991)
            b = model(images, texts, 'awgn', 100.)['image']
        torch.testing.assert_close(a, b)

    def test_train_eval_entrypoints_and_independent_verifier(self):
        import argparse
        from PIL import Image
        from v10.data import sha256, validate_manifest
        from v10.io import atomic_json
        from models.joint_image_text_model import JointImageTextSemanticCommunicationModel
        import train_v10
        import eval_v10
        from verify_v10_run import verify
        with tempfile.TemporaryDirectory(prefix='v10_synthetic_') as temporary:
            root = Path(temporary)
            for part, ids in [('train', [1, 2, 3, 4]), ('val', [5, 6, 7, 8])]:
                folder = root/(part+'2017')
                folder.mkdir()
                records = []
                for i in ids:
                    Image.new('RGB', (128, 128), color=(20*i, 40, 70)).save(folder/('{:012d}.jpg'.format(i)))
                    records.append({'image_id': i, 'caption': 'a synthetic red object {}'.format(i)})
                (root/(part+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
            manifest = {'train': [1, 2, 3, 4], 'validation': [5, 6], 'calibration': [7], 'test': [8],
                        'excluded_history': [], 'annotation_sha256': {'train': sha256(root/'train.jsonl'), 'val': sha256(root/'val.jsonl')}}
            validate_manifest(manifest)
            atomic_json(root/'splits.json', manifest)
            cfg = tiny_config()
            cfg['train'] = {'batch_size': 2, 'num_workers': 0}
            cfg['data'] = {part+'_'+key: str(root/(part+'2017' if key == 'image_root' else part+'.jsonl'))
                           for part in ('train', 'val') for key in ('image_root', 'annotation_file')}
            cfg['v10'].update(train_profiles=['awgn', 'fso_moderate'], train_snr=[10],
                validation_profiles=['awgn'], validation_snr=[10], common_probe_every=1,
                common_probe_weight=.5, anchor_every=1, anchor_weight=.05, adapter_epochs=1,
                worst_group_weight=.5)
            base = JointImageTextSemanticCommunicationModel(cfg['model'])
            checkpoint = root/'initial.pth'
            torch.save({'model': base.state_dict()}, checkpoint)
            initial_hash = sha256(checkpoint)
            shared = ['--checkpoint', str(checkpoint), '--split-manifest', str(root/'splits.json'),
                      '--output-dir', str(root/'train_run'), '--data-root', str(root),
                      '--vae-path', 'fake', '--text-model-path', 'fake', '--device', 'cpu', '--no-amp']
            args = train_v10.build_parser().parse_args(shared+['--epochs', '2', '--train-samples', '4', '--val-samples', '2'])
            with patch.object(train_v10, 'load_config', return_value=cfg):
                train_v10.main(args)
            self.assertEqual(sha256(checkpoint), initial_hash)
            self.assertEqual(verify(root/'train_run')['epochs'], 2)
            eval_args = eval_v10.build_parser().parse_args(shared+['--checkpoint', str(root/'train_run/best.pth'),
                    '--output-dir', str(root/'eval_run'), '--profiles', 'awgn', 'fso_moderate',
                    '--snr-list', '0', '--samples', '2', '--repeats', '1', '--save-images', '0'])
            with patch.object(eval_v10, 'load_config', return_value=cfg):
                eval_v10.main(eval_args)
            self.assertEqual(verify(root/'eval_run')['verified_rows'], 4)
            from compare_v10 import compare
            paired = compare(root/'eval_run', root/'eval_run', draws=100)
            self.assertEqual(paired['summary']['awgn/0.0']['delta_candidate_minus_baseline']['psnr']['mean'], 0.)
            # A time-budget stop is not allowed to masquerade as completion.
            timeout_args = copy.copy(args)
            timeout_args.output_dir = str(root/'timed_out')
            timeout_args.max_hours = 1e-12
            with patch.object(train_v10, 'load_config', return_value=cfg):
                with self.assertRaises(TimeoutError):
                    train_v10.main(timeout_args)
            self.assertFalse((root/'timed_out/COMPLETE.json').exists())
            # Partial/duplicate evidence must not be accepted as complete.
            samples = root/'eval_run/samples.jsonl'
            lines = samples.read_text(encoding='utf-8').splitlines()
            samples.write_text('\n'.join(lines[:-1]), encoding='utf-8')
            with self.assertRaises(ValueError):
                verify(root/'eval_run')

if __name__ == '__main__':
    unittest.main(verbosity=2)
