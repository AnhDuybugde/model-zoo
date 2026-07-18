import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    """FFN: Linear -> ReLU -> Dropout -> Linear (áp dụng độc lập từng vị trí token)."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Attention is All You Need)."""

    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_seq_len, d_model) — buffer, không tính gradient
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


if __name__ == "__main__":
    batch, seq_len, d_model, d_ff = 2, 10, 64, 256
    x = torch.randn(batch, seq_len, d_model)

    ffn = PositionWiseFeedForward(d_model, d_ff)
    pe = PositionalEncoding(d_model, max_seq_len=100)

    print("FFN :", x.shape, "->", ffn(x).shape)
    print("PE  :", x.shape, "->", pe(x).shape)
