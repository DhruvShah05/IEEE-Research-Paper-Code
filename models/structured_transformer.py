"""
models/structured_transformer.py — LOB-Structured Transformer.

Rename: "Improved Transformer" → "Structured Transformer" / "LOB-Structured Transformer"
(fix 0.3 — the paper claimed extra depth/capacity that does not exist; renamed honestly).

Changes (2.3):
  - Same config exposure as transformer.py (d_model, nhead, num_layers, dim_feedforward,
    dropout) so ablations can be launched from config without code changes.
  - token_mode: 'flat' | 'grouped' | 'level'
      'flat'    : each scalar feature = one token (identical to StandardTransformer).
      'grouped' : price/vol pairs = one token per pair (group_size=2).
                  For FI-2010 (144 features), grouping is restricted to the raw-40
                  price/vol block (columns 0–39); remaining 104 derived features are
                  treated as individual tokens (fix — previously grouped arbitrary cols).
      'level'   : one token per LOB level = (bid_p, bid_v, ask_p, ask_v), group_size=4.
                  This is what "grouped by LOB level" in the paper actually implies.
  - depth/width (num_layers/d_model) exposed for parameter-matched ablation (2.3).
  - Logs parameter count.
"""

import logging
import math

import torch
import torch.nn as nn
from .transformer import PositionalEncoding

logger = logging.getLogger(__name__)


class StructuredTransformer(nn.Module):
    """
    LOB-Structured Transformer.

    Supports independent ablation flags:
      token_mode   : 'flat' | 'grouped' | 'level'
      pooling_mode : 'mean' | 'cls' | 'attention'

    For FI-2010 with token_mode='grouped' or 'level', grouping is restricted
    to the raw 40-feature LOB block (columns 0–39); the remaining 104 derived
    features (columns 40–143) are each treated as individual scalar tokens.
    This makes the "level-structured" claim meaningful for FI-2010.
    """
    def __init__(
        self,
        in_features: int,
        token_mode: str = 'level',
        pooling_mode: str = 'attention',
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = None,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()
        self.token_mode   = token_mode
        self.pooling_mode = pooling_mode
        dim_feedforward   = dim_feedforward or d_model * 4

        # ----- Token projection -----
        if token_mode == 'level':
            # One token per LOB level: (bid_p, bid_v, ask_p, ask_v) → group_size=4
            # Raw LOB block has 10 bid levels + 10 ask levels = 20 levels of 2 cols each.
            # In the 40-feature layout: cols 0–1 = bid1 price/vol, 2–3 = bid2 price/vol, ...
            # Here we interleave as (bid_p_i, bid_v_i, ask_p_i, ask_v_i) per level.
            self.group_size = 4
            # 10 levels × 4 features = 40 (the raw LOB block exactly)
            n_lob_levels = 10
            lob_tokens = n_lob_levels       # 10 level-tokens from first 40 features
            extra_tokens = max(0, in_features - 40)  # derived features: 1 token each
            self.n_lob_tokens   = lob_tokens
            self.extra_features = extra_tokens
            self.seq_len = lob_tokens + extra_tokens

            self.lob_token_proj   = nn.Linear(self.group_size, d_model)
            self.extra_token_proj = nn.Linear(1, d_model) if extra_tokens > 0 else None

        elif token_mode == 'grouped':
            # Price/vol pairs: group_size=2
            # For FI-2010: only first 40 cols grouped; remaining 104 are scalar tokens
            self.group_size = 2
            n_lob_pairs  = 40 // self.group_size   # = 20
            extra_tokens = max(0, in_features - 40)
            self.n_lob_tokens   = n_lob_pairs
            self.extra_features = extra_tokens
            self.seq_len = n_lob_pairs + extra_tokens

            self.lob_token_proj   = nn.Linear(self.group_size, d_model)
            self.extra_token_proj = nn.Linear(1, d_model) if extra_tokens > 0 else None

        else:  # 'flat'
            self.seq_len = in_features
            self.token_proj = nn.Linear(1, d_model)

        # ----- Positional encoding -----
        n_cls = 1 if pooling_mode == 'cls' else 0
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_len + n_cls + 1)

        # ----- CLS token -----
        if pooling_mode == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # ----- Transformer encoder -----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ----- Attention pooling -----
        if pooling_mode == 'attention':
            self.attn_pool_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.attn_pool = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=1, batch_first=True
            )

        self.fc = nn.Linear(d_model, num_classes)

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (B, in_features) → (B, seq_len, d_model)."""
        if self.token_mode == 'flat':
            tokens = self.token_proj(x.unsqueeze(-1))   # (B, F, d_model)
            return tokens

        B = x.size(0)
        lob_block = x[:, :40]  # first 40 = raw LOB features

        if self.token_mode == 'level':
            # Reshape 40 features into 10 levels × (bid_p, bid_v, ask_p, ask_v)
            # Column layout in our 40-feature block:
            #   cols [0,1,2,3,...,38,39] = [bid1_p, bid1_v, bid2_p, bid2_v, ..., ask1_p, ask1_v, ...]
            # Interleave bid/ask per level:
            bid_p = lob_block[:, 0:20:2]   # bid prices: cols 0,2,4,...18
            bid_v = lob_block[:, 1:20:2]   # bid vols:   cols 1,3,5,...19
            ask_p = lob_block[:, 20:40:2]  # ask prices: cols 20,22,...38
            ask_v = lob_block[:, 21:40:2]  # ask vols:   cols 21,23,...39
            level_tokens = torch.stack([bid_p, bid_v, ask_p, ask_v], dim=-1)  # (B,10,4)
            lob_tokens = self.lob_token_proj(level_tokens)                     # (B,10,d)
        else:  # 'grouped'
            pairs = lob_block.view(B, self.n_lob_tokens, self.group_size)      # (B,20,2)
            lob_tokens = self.lob_token_proj(pairs)                            # (B,20,d)

        if self.extra_features > 0:
            extra_block = x[:, 40:]   # (B, extra_features)
            extra_tokens = self.extra_token_proj(extra_block.unsqueeze(-1))    # (B, extra, d)
            return torch.cat([lob_tokens, extra_tokens], dim=1)
        else:
            return lob_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features)
        B = x.size(0)
        tokens = self._tokenize(x)   # (B, seq_len, d_model)

        if self.pooling_mode == 'cls':
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        tokens = self.pos_encoder(tokens)
        tokens = self.transformer_encoder(tokens)

        if self.pooling_mode == 'cls':
            pooled = tokens[:, 0, :]
        elif self.pooling_mode == 'attention':
            query = self.attn_pool_query.expand(B, -1, -1)
            pooled, _ = self.attn_pool(query, tokens, tokens)
            pooled = pooled.squeeze(1)
        else:  # 'mean'
            pooled = tokens.mean(dim=1)

        return self.fc(pooled)


def build_model(config: dict) -> nn.Module:
    market = config.get('market')
    in_features = 144 if market == 'fi2010' else 40

    mp = config.get('model_params', {})

    model = StructuredTransformer(
        in_features     = in_features,
        token_mode      = mp.get('token_mode', 'level'),
        pooling_mode    = mp.get('pooling_mode', 'attention'),
        d_model         = mp.get('d_model', 64),
        nhead           = mp.get('nhead', 4),
        num_layers      = mp.get('num_layers', 2),
        dim_feedforward = mp.get('dim_feedforward', None),
        dropout         = mp.get('dropout', 0.1),
    )

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        f"StructuredTransformer(token={mp.get('token_mode','level')}, "
        f"pooling={mp.get('pooling_mode','attention')}, "
        f"layers={mp.get('num_layers',2)}, d_model={mp.get('d_model',64)}) "
        f"— param count: {param_count:,}"
    )
    return model
