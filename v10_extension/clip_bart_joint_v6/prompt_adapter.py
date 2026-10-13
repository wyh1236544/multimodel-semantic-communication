"""Small received-C -> SD text-conditioning residual, NOT ControlNet/DiT."""
import torch
from torch import nn
from clip_bart_joint_v5.model import Attention


class PromptAdapter(nn.Module):
    def __init__(self, text_dim, hidden=128, heads=4):
        super().__init__()
        self.common = nn.Linear(16, hidden)
        self.snr = nn.Linear(1, hidden)
        self.attention = Attention(text_dim, hidden, hidden, heads)
        self.output = nn.Linear(hidden, text_dim)
        self.gate = nn.Parameter(torch.tensor(-2.))
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, common, received_prompt, pilot_snr=None):
        if common.shape[1:] != (32,16) or received_prompt.ndim!=3:
            raise ValueError('Receiver public condition shapes')
        if not torch.isfinite(common).all() or not torch.isfinite(received_prompt).all():
            raise ValueError('Nonfinite received condition')
        snr=common.new_zeros(common.shape[0],1) if pilot_snr is None else pilot_snr.to(common).reshape(-1,1).clamp(-20,20)/20
        residual = self.output(self.attention(received_prompt,self.common(common)+self.snr(snr)[:,None]))
        return received_prompt+torch.sigmoid(self.gate)*residual
