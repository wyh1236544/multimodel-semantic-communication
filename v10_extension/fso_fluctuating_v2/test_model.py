"""Server Torch CPU parity/gradient/erasure contracts; not real model smoke."""
import math
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from fso_fluctuating_v2.core import load, apply_operator, as_complex, seed_for
from fso_fluctuating_v2.model import Frame, ContinuousChannel, surrogate_context


class TorchContracts(unittest.TestCase):
    def setup_channel(self,available=True):
        c=load(); old=nn.Module(); old.adapter=nn.Identity(); old.use_adapter=False
        channel=ContinuousChannel(old,c)
        coeff=np.zeros(11,np.complex64); coeff[2]=.6+.1j; coeff[7]=.03
        rng=np.random.default_rng(15); residual=(rng.normal(size=2304)+1j*rng.normal(size=2304))*.2
        record=dict(index=0,partition='train',probe_snr_db=1.,reliability=.5,available=available,
            stats=dict(analog_repeats=1,raw_utf8_wire_bytes=0))
        bank=dict(records=[record],coefficients=np.array([coeff]),residuals=np.array([residual]))
        ctx=surrogate_context(bank,0,94)
        frame=Frame({},0.,1,torch.device('cpu'),context=ctx,config=c,fixed_mode=0)
        channel.frame=frame
        return channel,frame,ctx

    def test_exact_numpy_surrogate_parity_and_gradient(self):
        channel,frame,ctx=self.setup_channel(); x=torch.randn(1,288,16,requires_grad=True)
        y=channel(x); s=ctx['surrogate']; rng=np.random.default_rng(seed_for(94,'residual'))
        residual=np.roll(s['residual'],int(rng.integers(2304)))*np.exp(1j*rng.uniform(-math.pi,math.pi))
        expected=apply_operator(as_complex(frame.captured),s['coefficient'],residual)
        actual=as_complex(y.detach().numpy())
        self.assertTrue(np.allclose(actual,expected,atol=3e-6))
        y.square().mean().backward()
        self.assertGreater(float(x.grad.square().sum()),0.)
        for mapping in (channel.tx_map,channel.rx_map): self.assertGreater(float(mapping.last.weight.grad.square().sum()),0.)

    def test_unavailable_is_zero_even_with_rx_bias(self):
        channel,_,_=self.setup_channel(False); channel.rx_map.last.bias.data.fill_(10.)
        self.assertEqual(int(torch.count_nonzero(channel(torch.randn(1,288,16)))),0)

    def test_holdout_rejected(self):
        with self.assertRaises(ValueError): surrogate_context(dict(records=[dict(partition='holdout')]),0,9)

    def test_stochastic_residual_not_mean_only(self):
        channel,_,ctx=self.setup_channel(); x=torch.randn(1,288,16)
        a=channel(x); ctx['noise_seed']=96; channel.frame.context['noise_seed']=96
        b=channel(x); self.assertFalse(torch.equal(a,b))


if __name__=='__main__': unittest.main(verbosity=2)
