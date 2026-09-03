"""
models/transformer.py — Standard Transformer for LOB prediction.

Changes (2.2):
  - d_model, nhead, num_layers, dim_feedforward, dropout exposed via
    config['model_params'] (previously hard-coded — prevented ablations).
  - Added 'temporal' token_mode: tokens are time-steps from a (T, F) window,
    each projected to d_model. Gives the model the same temporal context as
    DeepLOB (2.2 requirement). Requires WindowedDataset from data/loaders.py.
  - Logs parameter count.
"""

import math
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class StandardTransformer(nn.Module):
    """
    Standard Scalar-Token or Temporal Transformer.

    token_mode options
    ------------------
    'scalar'   (default) : each scalar feature is one token (no temporal context).
    'temporal' (new)     : each time-step from a (T, F) window is one token;
                           features are projected to d_model.  Requires input
                           shape (B, T, F) from WindowedDataset.

    The 'scalar' mode is kept to preserve existing results; 'temporal' is the
    recommended mode for a fair comparison with DeepLOB.
    """
    def __init__(
        self,
        in_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = None,
        dropout: float = 0.1,
        num_classes: int = 3,
        token_mode: str = 'scalar',
    ):
        super().__init__()
        self.token_mode = token_mode
        dim_feedforward = dim_feedforward or d_model * 4

        if token_mode == 'temporal':
            # Project each time-step's F features to d_model
            self.token_proj = nn.Linear(in_features, d_model)
        else:
            # 'scalar': project each scalar feature to d_model
            self.token_proj = nn.Linear(1, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.token_mode == 'temporal':
            # x: (B, T, F)  → already a sequence of F-dimensional tokens
            x = self.token_proj(x)   # (B, T, d_model)
        else:
            # x: (B, F)  → sequence of scalar tokens
            x = x.unsqueeze(-1)      # (B, F, 1)
            x = self.token_proj(x)   # (B, F, d_model)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # Mean pooling over sequence length
        pooled = x.mean(dim=1)
        return self.fc(pooled)


def build_model(config: dict) -> nn.Module:
    market = config.get('market')
    in_features = 144 if market == 'fi2010' else 40

    mp = config.get('model_params', {})
    model = StandardTransformer(
        in_features     = in_features,
        d_model         = mp.get('d_model', 64),
        nhead           = mp.get('nhead', 4),
        num_layers      = mp.get('num_layers', 2),
        dim_feedforward = mp.get('dim_feedforward', None),
        dropout         = mp.get('dropout', 0.1),
        token_mode      = mp.get('token_mode', 'scalar'),
    )

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"StandardTransformer — param count: {param_count:,}")
    return model
