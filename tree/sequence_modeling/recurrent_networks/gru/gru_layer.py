import torch
import torch.nn as nn

from gru_cell import GRUCell


class GRULayer(nn.Module):
    """
    Một layer GRU (unidirectional).

    Input:
      x: (batch, seq_len, input_size)
      h0: (batch, hidden_size) hoặc None
    Output:
      outputs: (batch, seq_len, hidden_size)
      h_n:     (batch, hidden_size)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.cell = GRUCell(input_size, hidden_size, bias=bias)
        self.hidden_size = hidden_size

    def forward(self, x, h0=None):
        batch_size, seq_len, _ = x.shape
        h_t = h0
        outputs = []

        for t in range(seq_len):
            h_t = self.cell(x[:, t, :], h_t)
            outputs.append(h_t)

        outputs = torch.stack(outputs, dim=1)
        return outputs, h_t


class BidirectionalGRULayer(nn.Module):
    """
    GRU 2 chiều — concat hidden theo feature dim.

    Output:
      outputs: (batch, seq_len, hidden_size * 2)
      h_n:     (batch, hidden_size * 2)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.forward_layer = GRULayer(input_size, hidden_size, bias)
        self.backward_layer = GRULayer(input_size, hidden_size, bias)
        self.hidden_size = hidden_size

    def forward(self, x, h0=None):
        if h0 is None:
            h_fwd = h_bwd = None
        elif isinstance(h0, tuple):
            h_fwd, h_bwd = h0
        else:
            h_fwd = h0[:, : self.hidden_size]
            h_bwd = h0[:, self.hidden_size :]

        out_fwd, h_fwd_n = self.forward_layer(x, h_fwd)

        x_rev = torch.flip(x, dims=[1])
        out_bwd, h_bwd_n = self.backward_layer(x_rev, h_bwd)
        out_bwd = torch.flip(out_bwd, dims=[1])

        outputs = torch.cat([out_fwd, out_bwd], dim=-1)
        h_n = torch.cat([h_fwd_n, h_bwd_n], dim=-1)
        return outputs, h_n


if __name__ == "__main__":
    batch, seq_len, input_size, hidden = 2, 7, 16, 32
    x = torch.randn(batch, seq_len, input_size)

    layer = GRULayer(input_size, hidden)
    y, h = layer(x)
    print("GRULayer   :", x.shape, "->", y.shape, "| h_n:", h.shape)

    bi = BidirectionalGRULayer(input_size, hidden)
    y2, h2 = bi(x)
    print("BiGRULayer :", x.shape, "->", y2.shape, "| h_n:", h2.shape)
