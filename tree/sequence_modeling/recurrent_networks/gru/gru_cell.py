import torch
import torch.nn as nn


class GRUCell(nn.Module):
    """
    GRU cell:

        r_t = σ(W_ir x_t + W_hr h_{t-1} + b_r)          # reset gate
        z_t = σ(W_iz x_t + W_hz h_{t-1} + b_z)          # update gate
        n_t = tanh(W_in x_t + r_t ⊙ (W_hn h_{t-1}) + b_n)  # candidate
        h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}

    Input:
      x_t: (batch, input_size)
      h_prev: (batch, hidden_size) hoặc None
    Output:
      h_t: (batch, hidden_size)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # reset + update gates
        self.W_x_rz = nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.W_h_rz = nn.Linear(hidden_size, 2 * hidden_size, bias=False)

        # candidate
        self.W_x_n = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_h_n = nn.Linear(hidden_size, hidden_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in (self.W_x_rz, self.W_x_n):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        for module in (self.W_h_rz, self.W_h_n):
            nn.init.orthogonal_(module.weight)

    def forward(self, x_t, h_prev=None):
        batch_size = x_t.size(0)
        if h_prev is None:
            h_prev = torch.zeros(
                batch_size, self.hidden_size, device=x_t.device, dtype=x_t.dtype
            )

        rz = self.W_x_rz(x_t) + self.W_h_rz(h_prev)
        r, z = rz.chunk(2, dim=-1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)

        n = torch.tanh(self.W_x_n(x_t) + self.W_h_n(r * h_prev))
        h_t = (1.0 - z) * n + z * h_prev
        return h_t


if __name__ == "__main__":
    batch, input_size, hidden_size = 4, 16, 32
    x_t = torch.randn(batch, input_size)

    cell = GRUCell(input_size, hidden_size)
    h = cell(x_t)

    print("x_t :", x_t.shape)
    print("h_t :", h.shape)
