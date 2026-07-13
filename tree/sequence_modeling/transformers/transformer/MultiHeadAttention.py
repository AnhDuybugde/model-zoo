import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Kích thước của mỗi head
        
        # Tạo các ma trận trọng số cho Q, K, V và Output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Linear projection và tách thành các heads
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k) -> Hoán vị thành (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Tính Scaled Dot-Product Attention
        # Scores: (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Áp dụng mask nếu có (ví dụ: Padding mask hoặc Causal mask)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 3. Nhân với V và gộp các heads lại
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. Linear projection cuối cùng
        return self.W_o(output)