import torch
import torch.nn as nn


class RNNCell(nn.Module):
    """
    Vanilla RNN cell (Elman):

        h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)

    Input:
      x_t: (batch, input_size)
      h_prev: (batch, hidden_size) hoặc None
    Output:
      h_t: (batch, hidden_size)
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if nonlinearity == "tanh":
            self.activation = torch.tanh
        elif nonlinearity == "relu":
            self.activation = torch.relu
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

        self.W_xh = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        # Khởi tạo ổn định hơn cho RNN (tránh explode/vanish ngay từ đầu)
        nn.init.xavier_uniform_(self.W_xh.weight)
        nn.init.orthogonal_(self.W_hh.weight)
        if self.W_xh.bias is not None:
            nn.init.zeros_(self.W_xh.bias)

    def forward(self, x_t, h_prev=None):
        batch_size = x_t.size(0)
        if h_prev is None:
            h_prev = torch.zeros(
                batch_size, self.hidden_size, device=x_t.device, dtype=x_t.dtype
            )

        h_t = self.activation(self.W_xh(x_t) + self.W_hh(h_prev))
        return h_t


if __name__ == "__main__":
    batch, input_size, hidden_size = 4, 16, 32
    x_t = torch.randn(batch, input_size)

    cell = RNNCell(input_size, hidden_size)
    h0 = torch.zeros(batch, hidden_size)
    h1 = cell(x_t, h0)

    print("x_t :", x_t.shape)
    print("h_t :", h1.shape)
