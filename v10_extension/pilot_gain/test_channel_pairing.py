"""Run on the existing server torch environment: actual channel, no CLIP/VAE."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch unavailable: actual numerical test NOT executed')
class ChannelPairingTests(unittest.TestCase):
    def test_real_channel_paired_draws_and_equal_budget(self):
        import eval_v10
        import v10.model as model
        from v10.channels import PROFILES
        from pilot_gain.worker import main
        from pilot_gain.run import STRATEGIES
        for profile in ('awgn', 'fso_moderate', 'fso_strong'):
            for snr in (0., 10., 20.):
                seen = []
                for strategy, mode in STRATEGIES.items():
                    def evaluate(args):
                        torch.manual_seed(42)
                        frame = model.LinkFrame(PROFILES[profile], snr, 1, torch.device('cpu'))
                        if mode is not None:
                            self.assertEqual(frame.requested_mode.item(), mode)
                        x = torch.linspace(-1., 1., 4608).reshape(1, 288, 16)
                        a = torch.ones(1, 288)
                        y, reliability = frame.payload(x, a, a)
                        self.assertEqual(tuple(y.shape), (1, 288, 16))
                        self.assertTrue(torch.isfinite(y).all().item())
                        stats = frame.last_stats
                        seen.append((stats['pilot_draw_sha256'], stats['payload_rng_sha256'],
                                     stats['payload_shape'], stats['forward_uses'], stats['reverse_rf_real_uses']))
                    argv = ['--strategy', strategy, '--checkpoint', 'unused', '--split-manifest', 'unused',
                            '--output-dir', 'unused', '--data-root', 'unused', '--vae-path', 'unused',
                            '--text-model-path', 'unused']
                    with patch.object(eval_v10, 'main', evaluate):
                        main(argv)
                self.assertTrue(all(x == seen[0] for x in seen))
                self.assertEqual(seen[0][-2], 2400 if profile == 'awgn' else 9408)


if __name__ == '__main__':
    unittest.main(verbosity=2)
