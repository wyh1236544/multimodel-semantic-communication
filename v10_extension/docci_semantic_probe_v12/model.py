"""Frozen I2/VAE with equal registered and active receiver-adapter architectures."""
import torch
from torch import nn


class SpatialResidual(nn.Module):
    """Image-derived spatial gate; optional text affects channel-wise modulation."""
    def __init__(self, channels):
        super().__init__()
        self.self_context = nn.Linear(channels, 512)
        self.affine = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 2*channels))
        self.spatial = nn.Sequential(nn.Conv2d(channels, 16, 1), nn.GELU(), nn.Conv2d(16, 1, 1), nn.Sigmoid())
        self.norm = nn.GroupNorm(8, channels, affine=False)
        # Exactly the old decoder at initialization; not a random corrupting branch.
        nn.init.zeros_(self.affine[-1].weight); nn.init.zeros_(self.affine[-1].bias)

    def forward(self, feature, text=None):
        context = self.self_context(feature.mean((2, 3)))
        if text is not None:
            if text.shape != context.shape or not torch.isfinite(text).all(): raise ValueError('Wrong finite text semantic vector')
            context = context + text
        gamma, beta = self.affine(context).chunk(2, -1)
        residual = self.spatial(feature) * (gamma.tanh()[:, :, None, None] * self.norm(feature) + beta.tanh()[:, :, None, None])
        return feature + .1 * residual


class ImageProbe(nn.Module):
    """Only an explicit ORACLE diagnostic may pass clean text to this receiver."""
    def __init__(self, base, arm):
        super().__init__()
        if arm not in ('A0_image_only', 'A1_oracle_text'): raise ValueError('Unknown image probe')
        self.base = base.requires_grad_(False).eval(); self.arm = arm
        self.adapters = nn.ModuleList([SpatialResidual(96), SpatialResidual(96), SpatialResidual(64)])
        self.configure()

    def configure(self):
        self.eval(); self.base.requires_grad_(False)
        self.adapters.requires_grad_(True)

    def receive(self, observed, oracle_text=None):
        c = self.base.c; grid = c['image_size']//16
        if observed.ndim != 2 or observed.shape[1] != c['image_width']*grid*grid or not torch.isfinite(observed).all():
            raise ValueError('Invalid fixed image observation')
        text = oracle_text if self.arm == 'A1_oracle_text' else None
        if self.arm == 'A1_oracle_text' and text is None: raise ValueError('Explicit ORACLE semantic vector required')
        d = self.base.image_decoder; wire = observed.reshape(-1, c['image_width'], grid, grid)
        # Fixed clean sentinels, no channel-truth or source mask side channel.
        cond = observed.new_tensor([1., 1.]).expand(observed.shape[0], -1)
        middle = self.adapters[0](d.a(d.input(wire)+d.condition(cond)[:, :, None, None]), text)
        middle = self.adapters[1](d.b(middle+d.coarse(middle)), text)
        high = self.adapters[2](d.c(d.up(middle)), text)
        return d.output(high)

    def saved(self): return {n:p.detach().cpu().clone() for n,p in self.adapters.named_parameters()}

    def restore(self, state):
        parameters = dict(self.adapters.named_parameters())
        if set(state) != set(parameters): raise ValueError('Adapter state keys differ')
        with torch.no_grad():
            for n,p in parameters.items():
                if p.shape != state[n].shape or not torch.isfinite(state[n]).all(): raise ValueError('Invalid adapter tensor')
                p.copy_(state[n].to(p))

    def groups(self): return {str(i):('adapters.'+str(i)+'.',) for i in range(3)}
