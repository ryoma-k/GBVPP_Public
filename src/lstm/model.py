import torch
import torch.nn as nn


class GBVSSLSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config['hidden_size']
        self.r_emb = nn.Embedding(3, 2)
        self.c_emb = nn.Embedding(3, 2)
        self.seq_emb = nn.Sequential(
            nn.Linear(4 + len(config['cont_cols']), self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size,
                            num_layers=config['num_layers'], dropout=config['dropout'],
                            batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, 2),
        )
        self.softplus = nn.Softplus()

    def forward(self, cate_v, cont_v):
        bs, length, _ = cont_v.shape
        r_emb = self.r_emb(cate_v[:,:,0]).view(bs, length, -1)
        c_emb = self.c_emb(cate_v[:,:,1]).view(bs, length, -1)
        seq_x = torch.cat((r_emb, c_emb, cont_v), 2)
        seq_emb = self.seq_emb(seq_x)
        seq_emb, _ = self.lstm(seq_emb)
        output = self.head(seq_emb)
        mu = output[...,0]
        b = self.softplus(output[...,1])
        return torch.stack([mu, b], -1)

    def get_parameters(self, lr=None):
        return [self.parameters()]
