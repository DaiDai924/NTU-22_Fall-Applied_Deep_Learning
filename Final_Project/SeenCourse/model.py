import torch
import torch.nn as nn


class HahowRanker(torch.nn.Module):
    def __init__(
        self,
        num_courses: int,
        ignore_idx: int,
        dim_embed: int = 256,
        num_head: int = 8,
        dropout: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super(HahowRanker, self).__init__()
        self.ignore_idx = ignore_idx
        # self.embed = nn.Embedding(num_courses+2, dim_embed - 16)  # * substract num of attrs
        self.embed = nn.Embedding(num_courses+2, dim_embed)
        self.linear = nn.Linear(dim_embed, num_courses + 1)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_embed,
                nhead=num_head,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers
        )

    def forward(self, tokens: torch.Tensor, padding_mask: torch.BoolTensor, attrs: torch.FloatTensor):
        # x = torch.cat((self.embed(tokens), attrs), dim=2)
        # x = attrs
        x = self.embed(tokens)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = x.masked_fill(torch.isnan(x), 0)

        return self.linear(x)
