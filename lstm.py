import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(
        self,
        num_embeddings,
        num_classes,
        padding_idx=None,
        n_layers=1,
        bidirectional=True,
        embedding_dim=250,
        hidden_dim=256,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, text):
        embeddings = self.embedding(text)
        return self.forward_embedding(embeddings)

    def get_embedding(self, text):
        return self.embedding(text)

    def forward_embedding(self, embeddings):
        embeddings = self.dropout(embeddings)
        _, (hidden, _) = self.rnn(embeddings)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)
