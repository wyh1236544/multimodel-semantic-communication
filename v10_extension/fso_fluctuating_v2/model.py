"""Frozen stochastic calibration parameters, trainable semantic encoder/decoder."""
import math
import numpy as np
import torch
from fso_jscc_v1.model import ContinuousChannel, Frame as OldFrame
from fso_fluctuating_v2.core import OFFSETS, seed_for


class Frame(OldFrame):
    def __init__(self, *args, context, config, **kwargs):
        surrogate = context.get('surrogate')
        self.surrogate = surrogate
        if surrogate is None:
            super().__init__(*args, context=context, config=config, **kwargs)
            return
        # Public receiver metadata only. Start with clean interface then replace
        # with calibration's pilot observations; neither TX map reads this.
        super().__init__(*args, context=dict(profile='clean',snr=40.,field_seed=0,noise_seed=0), config=config, **kwargs)
        self.context = context
        record = surrogate['record']
        self.routing_snr.fill_(max(-40., min(40., record['probe_snr_db'])))
        self.header_ok.fill_(record['available'])
        self.reliability = record['reliability']

    def payload(self, tokens):
        if self.surrogate is None: return super().payload(tokens)
        self.captured = tokens.detach().cpu().numpy()[0].copy()
        s = self.surrogate
        x = tokens.float().reshape(-1,2)
        x = torch.complex(x[:,0],x[:,1])/math.sqrt(2)
        columns = []
        for conjugate in (False, True):
            v = x.conj() if conjugate else x
            for shift in OFFSETS:
                if shift < 0: z = torch.cat((v[-shift:],v.new_zeros(-shift)))
                elif shift > 0: z = torch.cat((v.new_zeros(shift),v[:-shift]))
                else: z = v
                columns.append(z)
        matrix = torch.stack(columns+[torch.ones_like(x)], -1)
        coefficient = torch.as_tensor(s['coefficient'],device=x.device,dtype=x.dtype)
        # Random shift/phase changes noise realization without erasing measured
        # short-range correlation. No deterministic mean-only channel fit.
        seed = self.context['noise_seed']
        rng = np.random.default_rng(seed_for(seed, 'residual'))
        residual = np.roll(s['residual'],int(rng.integers(len(x))))*np.exp(1j*rng.uniform(-math.pi,math.pi))
        y = matrix@coefficient+torch.as_tensor(residual,device=x.device,dtype=x.dtype)
        restored = torch.stack((y.real,y.imag),-1).reshape_as(tokens)*math.sqrt(2)
        record = s['record']
        self.last_stats = dict(record['stats'], surrogate_record=record['index'],
            surrogate_partition=record['partition'], nominal_snr_db=None,
            pilot_snr_db=float(self.routing_snr[0]), receiver_available=record['available'],
            feature_mse=float((restored.detach()-tokens.detach()).square().mean()),
            surrogate_not_full_differentiable_hardware=True)
        return restored, tokens.new_full((1,288,1),self.reliability)


def surrogate_context(bank, index, noise_seed):
    r = bank['records'][index]
    if r['partition'] != 'train': raise ValueError('Holdout calibration forbidden in training')
    return dict(noise_seed=noise_seed, surrogate=dict(record=r,
        coefficient=bank['coefficients'][index],residual=bank['residuals'][index]))
