import torch
import torch.nn as nn
import torch.nn.functional as F

class CornerLSTM(nn.Module):
    def __init__(self, input_size=20, hidden_size=128, num_layers=2, dropout=0.2):
        super(CornerLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size * 2, 2)  # Concatenated output: [mean + last step]

    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)  # shape: (batch, seq_len, hidden)
        last_step = lstm_out[:, -1, :]  # shape: (batch, hidden)
        mean_pool = lstm_out.mean(dim=1)  # shape: (batch, hidden)
        concat = torch.cat([last_step, mean_pool], dim=1)  # shape: (batch, hidden * 2)
        output = self.fc(concat)  # shape: (batch, 2)
        return output
