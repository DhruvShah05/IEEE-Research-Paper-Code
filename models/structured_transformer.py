import torch
import torch.nn as nn
from .transformer import PositionalEncoding

class StructuredTransformer(nn.Module):
    """
    Structured Transformer.
    Must support independent ablation flags:
    - token_mode: 'flat' (scalars) vs 'grouped' (level-grouped tokens, e.g., price/vol pairs)
    - pooling_mode: 'mean', 'cls', or 'attention'
    """
    def __init__(self, in_features: int, token_mode: str = 'grouped', pooling_mode: str = 'attention', 
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2, num_classes: int = 3):
        super().__init__()
        
        self.token_mode = token_mode
        self.pooling_mode = pooling_mode
        
        if self.token_mode == 'grouped':
            # Assuming pairs (price, volume) for Crypto (40 features -> 20 pairs)
            # For FI-2010, the first 40 features are the raw LOB, the rest are derivations.
            # To keep it generic/simple as specified, we group adjacent features in pairs.
            self.group_size = 2
            self.seq_len = in_features // self.group_size
            self.token_proj = nn.Linear(self.group_size, d_model)
        else: # 'flat'
            self.seq_len = in_features
            self.token_proj = nn.Linear(1, d_model)
            
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_len + (1 if pooling_mode == 'cls' else 0))
        
        if self.pooling_mode == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
            
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        if self.pooling_mode == 'attention':
            # Attention pooling: learned query vector attends to all sequence outputs
            self.attn_pool_query = nn.Parameter(torch.randn(1, 1, d_model))
            # Q: (batch, 1, d_model), K,V: (batch, seq_len, d_model)
            self.attn_pool = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)
            
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x shape: (batch, in_features)
        batch_size = x.size(0)
        
        if self.token_mode == 'grouped':
            # (batch, seq_len, group_size)
            x = x.view(batch_size, self.seq_len, self.group_size)
        else:
            # (batch, in_features, 1)
            x = x.unsqueeze(-1)
            
        x = self.token_proj(x)
        
        if self.pooling_mode == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        if self.pooling_mode == 'cls':
            pooled = x[:, 0, :]
        elif self.pooling_mode == 'attention':
            query = self.attn_pool_query.expand(batch_size, -1, -1)
            # attn_output shape: (batch, 1, d_model)
            pooled, _ = self.attn_pool(query, x, x)
            pooled = pooled.squeeze(1)
        else: # 'mean'
            pooled = x.mean(dim=1)
            
        logits = self.fc(pooled)
        return logits

def build_model(config: dict) -> nn.Module:
    market = config.get('market')
    in_features = 144 if market == 'fi2010' else 40
    
    model_params = config.get('model_params', {})
    token_mode = model_params.get('token_mode', 'flat')
    pooling_mode = model_params.get('pooling_mode', 'mean')
    
    return StructuredTransformer(
        in_features=in_features, 
        token_mode=token_mode, 
        pooling_mode=pooling_mode
    )
