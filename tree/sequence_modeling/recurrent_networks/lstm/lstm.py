import torch
import torch.nn as nn

from lstm_layer import BidirectionalLSTMLayer, LSTMLayer


class LSTM(nn.Module):
    """
    Multi-layer LSTM.

    Input:
      x: (batch, seq_len, input_size)
      state0: optional ((h0, c0))
        h0, c0: (num_layers, batch, hidden * num_directions)
    Output:
      outputs: (batch, seq_len, hidden * num_directions)
      (h_n, c_n): mỗi cái (num_layers, batch, hidden * num_directions)
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=True,
        dropout=0.0,
        bidirectional=False,
    ):
        super().__init__()
        assert batch_first, "Chỉ hỗ trợ batch_first=True"

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else None

        layers = []
        for i in range(num_layers):
            layer_input = input_size if i == 0 else hidden_size * self.num_directions
            if bidirectional:
                layers.append(BidirectionalLSTMLayer(layer_input, hidden_size, bias))
            else:
                layers.append(LSTMLayer(layer_input, hidden_size, bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, state0=None):
        h_n_list, c_n_list = [], []

        for i, layer in enumerate(self.layers):
            if state0 is None:
                state_i = None
            else:
                h0, c0 = state0
                state_i = (h0[i], c0[i])

            x, (h_n, c_n) = layer(x, state_i)
            h_n_list.append(h_n)
            c_n_list.append(c_n)

            if self.dropout is not None and i < self.num_layers - 1:
                x = self.dropout(x)

        h_n = torch.stack(h_n_list, dim=0)
        c_n = torch.stack(c_n_list, dim=0)
        return x, (h_n, c_n)


class LSTMClassifier(nn.Module):
    """
    Embedding + LSTM + Linear — sequence classification.

    Input:  (batch, seq_len) token ids
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        vocab_size,
        embed_size,
        hidden_size,
        num_classes,
        num_layers=1,
        padding_idx=0,
        dropout=0.1,
        bidirectional=False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.lstm = LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        outputs, (h_n, c_n) = self.lstm(emb)
        last_hidden = h_n[-1]  # layer cuối
        return self.fc(self.dropout(last_hidden))


def lstm_small(input_size=32, hidden_size=64, num_layers=2, bidirectional=True):
    return LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.1,
        bidirectional=bidirectional,
    )


if __name__ == "__main__":
    batch, seq_len, input_size = 2, 10, 32
    hidden, layers = 64, 2

    x = torch.randn(batch, seq_len, input_size)
    model = lstm_small(input_size, hidden, layers)
    y, (h, c) = model(x)
    params = sum(p.numel() for p in model.parameters())

    print("LSTM")
    print("input  :", x.shape)
    print("output :", y.shape)
    print("h_n    :", h.shape)
    print("c_n    :", c.shape)
    print("params :", f"{params:,}")

    tokens = torch.randint(1, 100, (batch, seq_len))
    clf = LSTMClassifier(
        vocab_size=100,
        embed_size=32,
        hidden_size=64,
        num_classes=5,
        num_layers=2,
        bidirectional=True,
    )
    logits = clf(tokens)
    print("\nLSTMClassifier")
    print("tokens :", tokens.shape)
    print("logits :", logits.shape)
