import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class StandardTransformer(nn.Module):
    """
    Standard Scalar-Token Transformer (Preserves old notebook structure).
    Treats each scalar feature as a token in a sequence.
    """
    def __init__(self, in_features: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, num_classes: int = 3):
        super().__init__()
        
        # Map scalar token to d_model
        self.token_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=in_features)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Mean pooling
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x shape: (batch, in_features)
        # Convert to sequence of scalars: (batch, in_features, 1)
        x = x.unsqueeze(-1)
        
        # Project to (batch, in_features, d_model)
        x = self.token_proj(x)
        x = self.pos_encoder(x)
        
        # Encode
        x = self.transformer_encoder(x)
        
        # Mean Pooling over the sequence length
        pooled = x.mean(dim=1)
        
        logits = self.fc(pooled)
        return logits

def build_model(config: dict) -> nn.Module:
    market = config.get('market')
    in_features = 144 if market == 'fi2010' else 40
    return StandardTransformer(in_features=in_features)
