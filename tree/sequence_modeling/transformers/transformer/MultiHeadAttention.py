import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Kích thước của mỗi head

        # Ma trận trọng số cho Q, K, V và Output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 1. Linear projection và tách thành các heads
        # (B, L, d_model) -> (B, L, H, d_k) -> (B, H, L, d_k)
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        # scores: (B, H, L_q, L_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # mask == 0 -> chặn attention (padding / causal)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)

        # 3. Nhân với V và gộp heads lại
        # (B, H, L, d_k) -> (B, L, H, d_k) -> (B, L, d_model)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 4. Linear projection cuối
        return self.W_o(output)


if __name__ == "__main__":
    batch, seq_len, d_model, num_heads = 2, 10, 64, 8
    x = torch.randn(batch, seq_len, d_model)

    mha = MultiHeadAttention(d_model, num_heads)
    y = mha(x, x, x)

    print("Input :", x.shape)
    print("Output:", y.shape)
