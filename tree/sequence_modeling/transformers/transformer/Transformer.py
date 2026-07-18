import math

import torch
import torch.nn as nn

from Encoder_Decoder_Block import DecoderLayer, EncoderLayer
from PositionBlock import PositionalEncoding


def create_padding_mask(seq, pad_idx=0):
    """
    Mask các vị trí padding.
    seq: (batch, seq_len) token ids
    return: (batch, 1, 1, seq_len) — broadcast được cho (B, H, L_q, L_k)
    """
    # 1 = giữ, 0 = che
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(size, device=None):
    """
    Causal (look-ahead) mask: vị trí i không được nhìn j > i.
    return: (1, 1, size, size)
    """
    # lower-triangular = 1
    mask = torch.tril(torch.ones(size, size, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def create_target_mask(tgt, pad_idx=0):
    """
    Kết hợp padding mask + causal mask cho decoder self-attention.
    tgt: (batch, tgt_len)
    return: (batch, 1, tgt_len, tgt_len)
    """
    tgt_len = tgt.size(1)
    padding_mask = create_padding_mask(tgt, pad_idx)  # (B, 1, 1, L)
    causal_mask = create_causal_mask(tgt_len, device=tgt.device)  # (1, 1, L, L)
    return padding_mask & causal_mask.bool()


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer (Vaswani et al., 2017).

    Input:
      src: (batch, src_len) token ids
      tgt: (batch, tgt_len) token ids
    Output:
      logits: (batch, tgt_len, tgt_vocab_size)
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=5000,
        dropout=0.1,
        pad_idx=0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        # scale embedding theo paper: sqrt(d_model)
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        return self.encoder(x, src_mask)

    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        return self.decoder(x, enc_output, src_mask, tgt_mask)

    def forward(self, src, tgt):
        src_mask = create_padding_mask(src, self.pad_idx)
        tgt_mask = create_target_mask(tgt, self.pad_idx)

        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.fc_out(dec_output)

    @torch.no_grad()
    def generate(self, src, max_len=50, bos_idx=1, eos_idx=2):
        """
        Greedy decode đơn giản (phục vụ debug / demo).
        src: (batch, src_len)
        return: (batch, out_len) token ids
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        src_mask = create_padding_mask(src, self.pad_idx)
        enc_output = self.encode(src, src_mask)

        # bắt đầu bằng BOS
        ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = create_target_mask(ys, self.pad_idx)
            out = self.decode(ys, enc_output, src_mask, tgt_mask)
            logits = self.fc_out(out[:, -1, :])  # (B, vocab)
            next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            ys = torch.cat([ys, next_token], dim=1)

            if (next_token == eos_idx).all():
                break

        return ys


def transformer_base(src_vocab_size, tgt_vocab_size, pad_idx=0):
    """Config chuẩn paper: d_model=512, heads=8, layers=6, d_ff=2048."""
    return Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        pad_idx=pad_idx,
    )


def transformer_small(src_vocab_size, tgt_vocab_size, pad_idx=0):
    """Config nhỏ để debug nhanh."""
    return Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_seq_len=256,
        pad_idx=pad_idx,
    )


if __name__ == "__main__":
    src_vocab, tgt_vocab = 1000, 1200
    batch, src_len, tgt_len = 2, 12, 10

    model = transformer_small(src_vocab, tgt_vocab, pad_idx=0)

    src = torch.randint(1, src_vocab, (batch, src_len))
    tgt = torch.randint(1, tgt_vocab, (batch, tgt_len))
    # giả lập padding ở cuối
    src[:, -2:] = 0
    tgt[:, -1:] = 0

    logits = model(src, tgt)
    params = sum(p.numel() for p in model.parameters())

    print("Transformer (small)")
    print("src   :", src.shape)
    print("tgt   :", tgt.shape)
    print("logits:", logits.shape)
    print("params:", f"{params:,}")

    generated = model.generate(src, max_len=8, bos_idx=1, eos_idx=2)
    print("generate:", generated.shape)
