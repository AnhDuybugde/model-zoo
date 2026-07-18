import torch
import torch.nn as nn

from gru_layer import BidirectionalGRULayer, GRULayer


class GRU(nn.Module):
    """
    Multi-layer GRU.

    Input:
      x:  (batch, seq_len, input_size)
      h0: (num_layers, batch, hidden * num_directions) hoặc None
    Output:
      outputs: (batch, seq_len, hidden * num_directions)
      h_n:     (num_layers, batch, hidden * num_directions)
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
                layers.append(BidirectionalGRULayer(layer_input, hidden_size, bias))
            else:
                layers.append(GRULayer(layer_input, hidden_size, bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, h0=None):
        h_n_list = []

        for i, layer in enumerate(self.layers):
            h_i = None if h0 is None else h0[i]
            x, h_n = layer(x, h_i)
            h_n_list.append(h_n)
            if self.dropout is not None and i < self.num_layers - 1:
                x = self.dropout(x)

        h_n = torch.stack(h_n_list, dim=0)
        return x, h_n


class GRUClassifier(nn.Module):
    """
    Embedding + GRU + Linear — sequence classification.

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
        self.gru = GRU(
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
        outputs, h_n = self.gru(emb)
        last_hidden = h_n[-1]
        return self.fc(self.dropout(last_hidden))


def gru_small(input_size=32, hidden_size=64, num_layers=2, bidirectional=False):
    return GRU(
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
    model = gru_small(input_size, hidden, layers)
    y, h = model(x)
    params = sum(p.numel() for p in model.parameters())

    print("GRU")
    print("input  :", x.shape)
    print("output :", y.shape)
    print("h_n    :", h.shape)
    print("params :", f"{params:,}")

    tokens = torch.randint(1, 100, (batch, seq_len))
    clf = GRUClassifier(
        vocab_size=100,
        embed_size=32,
        hidden_size=64,
        num_classes=5,
        num_layers=2,
        bidirectional=True,
    )
    logits = clf(tokens)
    print("\nGRUClassifier")
    print("tokens :", tokens.shape)
    print("logits :", logits.shape)
