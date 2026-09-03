"""
models/deeplob.py — DeepLOB implementation.

Provides two classes:

  DeepLOBSnapshot  : original single-snapshot model kept for reference.
                     Input shape (B, F) where F=40 or F=144.
                     Produces ~60% on FI-2010 — NOT the published architecture.

  DeepLOB          : Zhang et al. (2019) architecture.
                     Input shape (B, 1, T=100, 40).
                     Expected to reproduce ~84% on FI-2010 k=10.
                     Requires WindowedDataset from data/loaders.py (1.5).

  build_model(config) returns DeepLOB (windowed) by default.
  To select the snapshot variant: config['model_params']['variant'] = 'snapshot'.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Inception module (Zhang et al. 2019 §III-B)
# ---------------------------------------------------------------------------

class _InceptionModule(nn.Module):
    """
    Inception module with three parallel 1D convolution branches
    (kernel sizes 1, 3, 5 along the time axis) + max-pool branch,
    followed by concatenation. Operates on (B, C, T) tensors.
    """
    def __init__(self, in_channels: int, nb_filters: int = 32):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, nb_filters, kernel_size=(1, 1), padding=(0, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(nb_filters),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, nb_filters, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(nb_filters),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, nb_filters, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(nb_filters),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_channels, nb_filters, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(nb_filters),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(x)
        return torch.cat([b1, b3, b5, bp], dim=1)  # (B, 4*nb_filters, T, 1)


# ---------------------------------------------------------------------------
# Real DeepLOB (Zhang et al. 2019)
# ---------------------------------------------------------------------------

class DeepLOB(nn.Module):
    """
    Zhang et al. (2019) DeepLOB architecture.

    Input shape : (B, 1, T, 40)   — batch, channel, time-steps, 40 LOB features
    Output shape: (B, num_classes)

    Architecture:
      Block 1-3 : Conv2d stride-2 kernels across price/volume pairs, then bid/ask,
                  then levels — progressively aggregating LOB structure.
      Block 4   : Inception module.
      Block 5   : LSTM(64, 2 layers).
      Head      : Linear(64, 3).
    """
    def __init__(self, T: int = 100, in_features: int = 40,
                 lstm_hidden: int = 64, num_classes: int = 3):
        super().__init__()
        self.T = T

        # --- Conv Block 1: (1,2) kernels across price/vol pairs ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),   # 40 → 20
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        # --- Conv Block 2: (1,2) across bid/ask alternating pairs ---
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),  # 20 → 10
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        # --- Conv Block 3: (1,10) across all 10 LOB levels → (B, 32, T, 1) ---
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10)),                 # 10 → 1
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        # --- Inception module ---
        self.inception = _InceptionModule(in_channels=32, nb_filters=32)
        # Output: (B, 128, T', 1) where T' ~ T (with padding)

        # --- LSTM ---
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=lstm_hidden,
            num_layers=2, batch_first=True, dropout=0.2
        )

        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # x: (B, 1, T, 40)
        x = self.conv1(x)   # (B, 32, T, 20)
        x = self.conv2(x)   # (B, 32, T, 10)
        x = self.conv3(x)   # (B, 32, T', 1)
        x = self.inception(x)   # (B, 128, T', 1)

        # Reshape for LSTM: (B, T', 128)
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, T', 128)
        lstm_out, _ = self.lstm(x)

        # Take last time step
        out = self.fc(lstm_out[:, -1, :])   # (B, num_classes)
        return out


# ---------------------------------------------------------------------------
# Legacy single-snapshot model (kept for reference only)
# ---------------------------------------------------------------------------

class DeepLOBSnapshot(nn.Module):
    """
    Original single-snapshot variant — NOT the Zhang et al. (2019) architecture.

    Kept as a reference row (deeplob_snapshot) for ablation purposes only.
    Uses Conv1d across the feature axis + BiLSTM; produces ~60% on FI-2010
    because it has no temporal context.

    Do NOT use as the main DeepLOB entry in result tables.
    """
    def __init__(self, in_features: int, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=4, padding=1),
            nn.LeakyReLU(0.01), nn.BatchNorm1d(16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=4, padding=1),
            nn.LeakyReLU(0.01), nn.BatchNorm1d(16)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=4, padding=1),
            nn.LeakyReLU(0.01), nn.BatchNorm1d(32)
        )
        self.lstm = nn.LSTM(32, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)          # (B, 1, F)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.transpose(1, 2)       # (B, F', 32)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """
    Returns the real DeepLOB (windowed) by default.
    Set config['model_params']['variant'] = 'snapshot' for the legacy variant.
    """
    mp      = config.get('model_params', {})
    variant = mp.get('variant', 'windowed')
    market  = config.get('market')
    in_features = 144 if market == 'fi2010' else 40

    if variant == 'snapshot':
        model = DeepLOBSnapshot(in_features=in_features)
    else:
        T          = mp.get('T', 100)
        lstm_hidden = mp.get('lstm_hidden', 64)
        # Real DeepLOB uses raw 40-feature input regardless of market (raw40 block)
        model = DeepLOB(T=T, in_features=40, lstm_hidden=lstm_hidden)

    param_count = sum(p.numel() for p in model.parameters())
    import logging
    logging.getLogger(__name__).info(
        f"DeepLOB({'Snapshot' if variant == 'snapshot' else ''}) — "
        f"param count: {param_count:,}"
    )
    return model
