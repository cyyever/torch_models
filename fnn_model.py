import torch.nn as nn


class FNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        token_num,
        max_len,
        hidden_dim=None,
        embedding_dim=None,
        use_direct_connection=True,
        tie_weights=False,
    ):
        super().__init__()
        self.token_num = token_num
        self.max_len = max_len
        if embedding_dim is None:
            embedding_dim = 30
        self.embedding = nn.Embedding(token_num, embedding_dim=embedding_dim)
        self.use_direct_connection = use_direct_connection
        if self.use_direct_connection:
            self.direct_connection_fc = nn.Linear(max_len * embedding_dim, token_num)
        else:
            print("no direct_connection_fc")

        if hidden_dim is None:
            hidden_dim = 100

        self.fc2 = nn.Linear(max_len * embedding_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, token_num)
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if embedding_dim != hidden_dim:
                raise ValueError(
                    "When using the tied flag, embedding_dim must be equal to hidden_dim"
                )
            self.fc3.weight = self.embedding.weight

        self.activation = nn.LogSoftmax()

    def forward(self, input):
        embedding = self.embedding(input).view(input.shape[0], -1)
        output = self.fc3(self.fc2(embedding).tanh())
        if self.use_direct_connection:
            output += self.direct_connection_fc(embedding)
        return self.activation(output)
