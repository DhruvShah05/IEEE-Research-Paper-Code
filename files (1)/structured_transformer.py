"""
models/structured_transformer.py — LOB-Structured Transformer.

Naming: this is the model the earlier paper called the "Improved Transformer".
It has the SAME depth/width as the standard Transformer by default; the only
differences are how the 40 raw LOB features are tokenised and how the sequence
is pooled. Call it "Structured Transformer" in the paper (fix 0.3).

Input layout (A2): the model assumes the CANONICAL interleaved column layout

    col 4i+0 = ask_price_(i+1)
    col 4i+1 = ask_vol_(i+1)
    col 4i+2 = bid_price_(i+1)
    col 4i+3 = bid_vol_(i+1)            i = 0..9  ->  cols 0..39

main.py applies data.loaders.reorder_to_canonical() to crypto data before
training when token_mode is 'level' or 'grouped'. FI-2010 is already canonical.
Columns 40.. (FI-2010 derived features in full144 mode) are treated as one
scalar token each.

token_mode
    'flat'    : every scalar feature is a token (identical to StandardTransformer).
    'grouped' : one token per (price, volume) pair  -> 20 LOB tokens (+ extras).
    'level'   : one token per LOB level (ask_p, ask_v, bid_p, bid_v) -> 10 LOB tokens (+ extras).

pooling_mode
    'mean' | 'cls' | 'attention'

All of d_model / nhead / num_layers / dim_feedforward / dropout are configurable
through config['model_params'] so the ablation grid can be driven from YAML.
"""

import logging

import torch
import torch.nn as nn

from .transformer import PositionalEncoding

logger = logging.getLogger(__name__)

N_LOB_COLS = 40
N_LEVELS = 10


class StructuredTransformer(nn.Module):
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
        if token_mode not in ('flat', 'grouped', 'level'):
            raise ValueError(f"token_mode must be flat|grouped|level, got {token_mode!r}")
        if pooling_mode not in ('mean', 'cls', 'attention'):
            raise ValueError(f"pooling_mode must be mean|cls|attention, got {pooling_mode!r}")
        if token_mode != 'flat' and in_features < N_LOB_COLS:
            raise ValueError(
                f"token_mode={token_mode!r} needs >= {N_LOB_COLS} features, got {in_features}"
            )

        self.in_features = in_features
        self.token_mode = token_mode
        self.pooling_mode = pooling_mode
        dim_feedforward = dim_feedforward or d_model * 4

        # ----- Tokenisation -----
        if token_mode == 'flat':
            self.group_size = 1
            self.n_lob_tokens = in_features
            self.extra_features = 0
            self.token_proj = nn.Linear(1, d_model)
        else:
            self.group_size = 4 if token_mode == 'level' else 2
            self.n_lob_tokens = N_LOB_COLS // self.group_size   # 10 or 20
            self.extra_features = max(0, in_features - N_LOB_COLS)
            self.lob_token_proj = nn.Linear(self.group_size, d_model)
            self.extra_token_proj = (
                nn.Linear(1, d_model) if self.extra_features > 0 else None
            )
        self.seq_len = self.n_lob_tokens + self.extra_features

        # ----- Positional encoding / CLS -----
        n_cls = 1 if pooling_mode == 'cls' else 0
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_len + n_cls + 1)
        if pooling_mode == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ----- Encoder -----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ----- Pooling -----
        if pooling_mode == 'attention':
            self.attn_pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.attn_pool = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=1, batch_first=True, dropout=dropout
            )

        self.fc = nn.Linear(d_model, num_classes)

    # ------------------------------------------------------------------
    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_features) -> (B, seq_len, d_model). Assumes canonical layout."""
        if self.token_mode == 'flat':
            return self.token_proj(x.unsqueeze(-1))

        B = x.size(0)
        lob = x[:, :N_LOB_COLS]
        # Canonical layout is contiguous per level, so a plain view gives
        #   level  : (B, 10, 4) = (ask_p, ask_v, bid_p, bid_v)
        #   grouped: (B, 20, 2) = (ask_p, ask_v), (bid_p, bid_v), ...
        lob_tokens = self.lob_token_proj(
            lob.reshape(B, self.n_lob_tokens, self.group_size)
        )
        if self.extra_features > 0:
            extra_tokens = self.extra_token_proj(x[:, N_LOB_COLS:].unsqueeze(-1))
            return torch.cat([lob_tokens, extra_tokens], dim=1)
        return lob_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        tokens = self._tokenize(x)

        if self.pooling_mode == 'cls':
            tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)

        tokens = self.pos_encoder(tokens)
        tokens = self.transformer_encoder(tokens)

        if self.pooling_mode == 'cls':
            pooled = tokens[:, 0, :]
        elif self.pooling_mode == 'attention':
            query = self.attn_pool_query.expand(B, -1, -1)
            pooled, _ = self.attn_pool(query, tokens, tokens)
            pooled = pooled.squeeze(1)
        else:
            pooled = tokens.mean(dim=1)

        return self.fc(pooled)


def _resolve_in_features(config: dict) -> int:
    market = config.get('market')
    feature_set = config.get('data', {}).get('feature_set', 'full144')
    if market == 'fi2010' and feature_set == 'full144':
        return 144
    return 40


def build_model(config: dict) -> nn.Module:
    mp = config.get('model_params', {})
    in_features = _resolve_in_features(config)

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
        f"StructuredTransformer(token={model.token_mode}, pooling={model.pooling_mode}, "
        f"layers={mp.get('num_layers', 2)}, d_model={mp.get('d_model', 64)}, "
        f"in_features={in_features}) — param count: {param_count:,}"
    )
    return model
