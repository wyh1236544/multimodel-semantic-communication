import torch
from torch import Tensor, nn


class TextReconstructionHead(nn.Module):
    """Non-autoregressive text recovery using only post-channel tokens."""

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        text_max_length: int,
        num_heads: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.queries = nn.Parameter(
            torch.randn(1, text_max_length, hidden_dim) * 0.02
        )
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def decode_features(self, memory: Tensor) -> Tensor:
        queries = self.queries.expand(memory.size(0), -1, -1)
        return self.decoder(queries, memory)

    def forward(self, memory: Tensor) -> Tensor:
        return self.output(self.decode_features(memory))
