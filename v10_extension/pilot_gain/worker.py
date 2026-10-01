"""Instrument pairing only; use the existing V10 evaluator and channel unchanged."""
import hashlib
from pathlib import Path
import sys

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT))


def digest_tensors(tensors):
    h = hashlib.sha256()
    for tensor in tensors:
        v = tensor.detach().contiguous().cpu()
        h.update(str((str(v.dtype), tuple(v.shape))).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def main(argv=None):
    import argparse
    import torch
    import eval_v10
    import v10.model as model_module
    from v10.channels import LinkFrame
    from pilot_gain.run import STRATEGIES
    from unittest.mock import patch
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument('--strategy', required=True, choices=list(STRATEGIES))
    own, rest = p.parse_known_args(argv)
    args = eval_v10.build_parser().parse_args(rest)
    args.fixed_mode = STRATEGIES[own.strategy]
    args.strategy = own.strategy

    class PairedFrame(LinkFrame):
        def __init__(self, *values, **settings):
            # Explicitly override even a custom config's non-null fixed_mode.
            settings['fixed_mode'] = STRATEGIES[own.strategy]
            super().__init__(*values, **settings)
            self.pilot_draw_sha256 = digest_tensors([
                self.true_gain, self.true_noise_var, self.receiver.gain,
                self.receiver.noise_var, self.receiver.estimation_var, self.routing_snr])

        def payload(self, tokens, *values, **settings):
            # Equal tensor shapes + equal generator state prove paired noise draws
            # in unchanged LinkFrame.payload; don't compare restored noisy signals.
            states = [torch.get_rng_state()]
            if tokens.device.type == 'cuda':
                states.append(torch.cuda.get_rng_state(tokens.device))
            noise_identity = digest_tensors(states)
            out = super().payload(tokens, *values, **settings)
            self.last_stats.update(strategy=own.strategy,
                                   pilot_draw_sha256=self.pilot_draw_sha256,
                                   payload_rng_sha256=noise_identity,
                                   payload_shape=list(tokens.shape))
            return out

    with patch.object(model_module, 'LinkFrame', PairedFrame):
        eval_v10.main(args)


if __name__ == '__main__':
    main()
