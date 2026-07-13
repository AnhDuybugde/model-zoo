class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Sub-layer 1: Self-Attention
        attn_out = self.mha(x, x, x, mask)
        x = self.layernorm1(x + self.dropout(attn_out))
        # Sub-layer 2: Feed Forward
        ffn_out = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_out))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_mha = MultiHeadAttention(d_model, num_heads)
        self.cross_mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Sub-layer 1: Masked Self-Attention (Không nhìn thấy từ tương lai)
        self_attn = self.self_mha(x, x, x, tgt_mask)
        x = self.layernorm1(x + self.dropout(self_attn))
        
        # Sub-layer 2: Cross-Attention (Query từ Decoder, Key-Value từ Encoder)
        cross_attn = self.cross_mha(x, enc_output, enc_output, src_mask)
        x = self.layernorm2(x + self.dropout(cross_attn))
        
        # Sub-layer 3: Feed Forward
        ffn_out = self.ffn(x)
        x = self.layernorm3(x + self.dropout(ffn_out))
        return x