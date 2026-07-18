import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """
    LSTM cell:

        i_t = σ(W_ii x_t + W_hi h_{t-1} + b_i)   # input gate
        f_t = σ(W_if x_t + W_hf h_{t-1} + b_f)   # forget gate
        g_t = tanh(W_ig x_t + W_hg h_{t-1} + b_g) # cell candidate
        o_t = σ(W_io x_t + W_ho h_{t-1} + b_o)   # output gate

        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        h_t = o_t ⊙ tanh(c_t)

    Input:
      x_t: (batch, input_size)
      state: (h_prev, c_prev) mỗi cái (batch, hidden_size), hoặc None
    Output:
      h_t, c_t
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Gộp 4 gate vào 1 linear cho gọn / nhanh hơn (vẫn dễ debug)
        self.W_x = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.orthogonal_(self.W_h.weight)
        if self.W_x.bias is not None:
            # forget gate bias = 1 giúp nhớ dài hơn lúc mới train
            nn.init.zeros_(self.W_x.bias)
            hidden = self.hidden_size
            self.W_x.bias.data[hidden : 2 * hidden].fill_(1.0)

    def forward(self, x_t, state=None):
        batch_size = x_t.size(0)
        if state is None:
            h_prev = torch.zeros(
                batch_size, self.hidden_size, device=x_t.device, dtype=x_t.dtype
            )
            c_prev = torch.zeros(
                batch_size, self.hidden_size, device=x_t.device, dtype=x_t.dtype
            )
        else:
            h_prev, c_prev = state

        gates = self.W_x(x_t) + self.W_h(h_prev)
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


if __name__ == "__main__":
    batch, input_size, hidden_size = 4, 16, 32
    x_t = torch.randn(batch, input_size)

    cell = LSTMCell(input_size, hidden_size)
    h, c = cell(x_t)

    print("x_t :", x_t.shape)
    print("h_t :", h.shape)
    print("c_t :", c.shape)
