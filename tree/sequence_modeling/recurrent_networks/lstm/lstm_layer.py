import torch
import torch.nn as nn

from lstm_cell import LSTMCell


class LSTMLayer(nn.Module):
    """
    Một layer LSTM (unidirectional).

    Input:
      x: (batch, seq_len, input_size)
      state0: (h0, c0) mỗi cái (batch, hidden_size) hoặc None
    Output:
      outputs: (batch, seq_len, hidden_size)
      (h_n, c_n)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size, bias=bias)
        self.hidden_size = hidden_size

    def forward(self, x, state0=None):
        batch_size, seq_len, _ = x.shape

        if state0 is None:
            h_t = c_t = None
            state = None
        else:
            h_t, c_t = state0
            state = (h_t, c_t)

        outputs = []
        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :], state)
            state = (h_t, c_t)
            outputs.append(h_t)

        outputs = torch.stack(outputs, dim=1)
        return outputs, (h_t, c_t)


class BidirectionalLSTMLayer(nn.Module):
    """
    LSTM 2 chiều — concat hidden theo feature dim.

    Output:
      outputs: (batch, seq_len, hidden_size * 2)
      (h_n, c_n): mỗi cái (batch, hidden_size * 2)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.forward_layer = LSTMLayer(input_size, hidden_size, bias)
        self.backward_layer = LSTMLayer(input_size, hidden_size, bias)
        self.hidden_size = hidden_size

    def forward(self, x, state0=None):
        if state0 is None:
            state_fwd = state_bwd = None
        else:
            h0, c0 = state0
            # h0/c0: (B, 2H) -> tách 2 chiều
            state_fwd = (h0[:, : self.hidden_size], c0[:, : self.hidden_size])
            state_bwd = (h0[:, self.hidden_size :], c0[:, self.hidden_size :])

        out_fwd, (h_fwd, c_fwd) = self.forward_layer(x, state_fwd)

        x_rev = torch.flip(x, dims=[1])
        out_bwd, (h_bwd, c_bwd) = self.backward_layer(x_rev, state_bwd)
        out_bwd = torch.flip(out_bwd, dims=[1])

        outputs = torch.cat([out_fwd, out_bwd], dim=-1)
        h_n = torch.cat([h_fwd, h_bwd], dim=-1)
        c_n = torch.cat([c_fwd, c_bwd], dim=-1)
        return outputs, (h_n, c_n)


if __name__ == "__main__":
    batch, seq_len, input_size, hidden = 2, 7, 16, 32
    x = torch.randn(batch, seq_len, input_size)

    layer = LSTMLayer(input_size, hidden)
    y, (h, c) = layer(x)
    print("LSTMLayer   :", x.shape, "->", y.shape, "| h:", h.shape, "c:", c.shape)

    bi = BidirectionalLSTMLayer(input_size, hidden)
    y2, (h2, c2) = bi(x)
    print("BiLSTMLayer :", x.shape, "->", y2.shape, "| h:", h2.shape, "c:", c2.shape)
