import torch.nn


class SimpleLstmModel(torch.nn.Module):
    def __init__(self, *,
                 encoding_size,
                 embedding_size,
                 lstm_size,
                 lstm_layers):
        super().__init__()

        self.embedding = torch.nn.Embedding(encoding_size, embedding_size)
        self.lstm = torch.nn.LSTM(input_size=embedding_size,
                                  hidden_size=lstm_size,
                                  num_layers=lstm_layers)
        self.fc = torch.nn.Linear(lstm_size, encoding_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, h0, c0):
        # shape of x is [seq, batch, feat]
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        logits = self.fc(out)

        return self.softmax(logits), logits, (hn, cn)
