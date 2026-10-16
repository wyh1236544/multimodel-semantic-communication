"""AWGN at the original TX wire; observation-derived RX memory and EOS gate."""
from collections import OrderedDict
import hashlib
from pathlib import Path
import numpy as np
from docci_segment_bridge_v17.core import read, sha, noise_seed, kind, CACHE_INDEX


def tensor_sha(t):
    x = t.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(x.tobytes()).hexdigest()


class Receiver:
    def __init__(self, model):
        self.model = model

    def __call__(self, wire, channel, seed_value):
        import torch
        from docci_segment_bridge_v17.model import content_gate
        with torch.no_grad():
            wire = wire.float()
            if wire.ndim != 2 or wire.shape[1] != 18432 or not torch.isfinite(wire).all():
                raise ValueError('Original public TX wire required')
            variance = 0. if channel == 'clean' else 10.**(-int(channel.split('_')[1])/10.)
            if channel not in ('clean', 'awgn_0', 'awgn_5', 'awgn_10', 'awgn_20'):
                raise ValueError('Unknown frozen symbol-level channel')
            generator = torch.Generator(device='cpu').manual_seed(seed_value)
            noise = torch.randn(tuple(wire.shape), generator=generator, dtype=torch.float32).to(wire.device)*variance**.5
            observed = wire+noise
            memory = self.model.receive(observed)['memory'].detach()
            gate = content_gate(self.model.bart, memory).detach()
            signal_power = float(wire.square().mean())
            noise_power = float(noise.square().mean())
            stats = dict(nominal_snr_db=None if channel == 'clean' else int(channel.split('_')[1]),
                         realized_snr_db=None if not variance else 10.*np.log10(signal_power/noise_power),
                         signal_power=signal_power, noise_power=noise_power, noise_variance=variance,
                         noise_seed=seed_value, tx_sha256=tensor_sha(wire), observed_sha256=tensor_sha(observed),
                         memory_sha256=tensor_sha(memory), gate_sha256=tensor_sha(gate),
                         gate_recomputed_from_received_memory=True, clean_gate_reused=False)
            return memory, gate, stats


class Items:
    def __init__(self, cache, split, limit=None, lru=8):
        self.root = Path(cache)
        if sha(self.root/'index.json') != CACHE_INDEX:
            raise ValueError('Use the already approved immutable v16_cache_01')
        self.entries = [r for r in read(self.root/'index.json')['records'] if r['official_split'] == split]
        if limit is not None:
            self.entries = self.entries[:limit]
        self.lru, self.values = lru, OrderedDict()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        if i not in self.values:
            r = self.entries[i]
            value = dict(r)
            for part, keys in (('text', ('ids', 'wire')), ('image', ('oracle_clip',))):
                p = self.root/r[part+'_file']
                if sha(p) != r[part+'_file_sha256']:
                    raise ValueError('Original frozen cache array changed')
                with np.load(p, allow_pickle=False) as z:
                    for k in keys:
                        value['text_wire' if k == 'wire' else k] = z[k].copy()
            self.values[i] = value
            while len(self.values) > self.lru:
                self.values.popitem(last=False)
        self.values.move_to_end(i)
        return self.values[i]


def materialize(items, indices, receiver, arm, channels, phase, epoch, device):
    import torch
    from docci_segment_bridge_v17.model import canonical
    values = [items[i] for i in indices]
    observed, gates, identities = [], [], []
    for r, channel in zip(values, channels):
        wire = torch.as_tensor(r['text_wire'][None], device=device)
        memory, gate, stats = receiver(wire, channel, noise_seed(phase, epoch, r['example_id'], channel))
        observed.append(memory)
        gates.append(gate)
        identities.append(stats)
    memory, gate = torch.cat(observed), torch.cat(gates)
    visual = torch.as_tensor(np.stack([r['oracle_clip'] for r in values]), device=device)
    aux = canonical(memory, visual, kind(arm))
    ids = torch.as_tensor(np.stack([r['ids'] for r in values]), device=device)
    return memory, aux, gate, ids, identities


def save_npz(path, **values):
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError('Nonfinite prepared array')
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **values)
    return sha(path)
