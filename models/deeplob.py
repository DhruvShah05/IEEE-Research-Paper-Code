import torch
import torch.nn as nn

class DeepLOB(nn.Module):
    """
    Single-snapshot variant of DeepLOB.
    Takes input of shape (batch, 1, features).
    Uses 3 1D Convolutional blocks followed by a 2-layer BiLSTM.
    """
    def __init__(self, in_features: int, num_classes: int = 3):
        super(DeepLOB, self).__init__()
        
        # 1D Conv blocks. Input shape expects (batch, channels, seq_len)
        # So we treat features as the sequence length (channels=1).
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(16)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(32)
        )
        
        # BiLSTM
        # In LSTM, input shape should be (batch, seq, features) if batch_first=True
        # So after conv, shape is (batch, 32, feature_length). 
        # We transpose to (batch, feature_length, 32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        
        # We will take the last hidden state of the LSTM sequence to classify
        self.fc = nn.Linear(128, num_classes) # 64 * 2 (bidirectional)
        
    def forward(self, x):
        # x shape: (batch, features)
        x = x.unsqueeze(1) # (batch, 1, features)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Transpose for LSTM: (batch, seq_len=features_left, channels=32)
        x = x.transpose(1, 2)
        
        # lstm_out shape: (batch, seq_len, 128)
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step
        last_step_out = lstm_out[:, -1, :]
        
        logits = self.fc(last_step_out)
        return logits

def build_model(config: dict) -> nn.Module:
    market = config.get('market')
    in_features = 144 if market == 'fi2010' else 40
    return DeepLOB(in_features=in_features)
