import torch
from torch import Tensor, nn


class TextDecoder(nn.Module):
    """Optional non-autoregressive caption head returning [B, L, vocab] logits."""

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        max_length: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.max_length = max_length
        self.position_queries = nn.Parameter(torch.randn(1, max_length, hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, semantic_tokens: Tensor) -> Tensor:
        queries = self.position_queries.expand(semantic_tokens.size(0), -1, -1)
        return self.output(self.decoder(queries, semantic_tokens))
