from typing import Dict

import torch
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,          # the number of features in the hidden state h
        num_layers: int,           # number of recurrent layers
        dropout: float,            # a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout
        bidirectional: bool,       # num_directions: true = 2, false = 1
        num_class: int,            # the number of output classes
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.embed_dim = len(self.embed.weight[1])  # word feature: 300

        # TODO: model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class

        # input_size: The number of expected features in the input x
        self.gru = torch.nn.GRU(input_size=self.embed_dim, \
                                hidden_size=self.hidden_size, \
                                num_layers=self.num_layers, \
                                dropout=self.dropout, \
                                bidirectional=self.bidirectional, \
                                batch_first=True)
        self.fc = torch.nn.Linear(self.encoder_output_size, self.num_class)
        # self.relu = torch.nn.ReLU()

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        """
        for initializing the last classification layer
           if bidirectional = True, output size = 2 * hidden_size, 
           else: output size = hidden_size
        """
        return self.hidden_size * (self.bidirectional + 1)

    def forward(self, batch) -> torch.Tensor:
        # TODO: implement model forward
        """
            out: (batch_size, seq_len, hudden_size * num_directions)
            h_n: (num_layers * num_directions, batch_size, hidden_size)
            c_n: (num_layers * num_directions, batch_size, hidden_size)
        """

        batch = self.embed.weight[batch]    # batch: [batch_size, sequence_length(max_len), embedded_dim]
        out, h = self.gru(batch)            # out: [batch_size, sequence_length(max_len), hidden_size * 2(bidirectional)]
        # out = self.relu(out)

        # out[:, -1, :]: [batch_size, hidden_size * 2(bidirectional)], get the outputs of the last hidden layer
        out = self.fc(out[:, -1, :])        # out: [batch_size, num_class]

        return out


class SeqTagger(SeqClassifier):
    # TODO: implement slot model
    def forward(self, batch) -> torch.Tensor:
        batch = self.embed.weight[batch]
        out, h = self.gru(batch)
        # out = self.relu(out)
        out = self.fc(out)                  # out: [batch_size, sequence_length(max_len), num_class]

        return out
        