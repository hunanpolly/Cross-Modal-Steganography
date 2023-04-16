import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class SecretEncoder(nn.Module):
    def __init__(self, secret_size):
        super(SecretEncoder, self).__init__()
        self.linear1 = nn.Linear(secret_size, 2048)
        self.linear2 = nn.Linear(1, 196)
        self.relu = nn.ReLU()
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)

    def forward(self, x):
        x = self.linear1(x)
        x = x.unsqueeze(2)
        x = self.linear2(x)
        x = self.relu(x.permute(0,2,1))
        return x


class SecretExtractor(nn.Module):
    def __init__(self, opt):
        super(SecretExtractor, self).__init__()
        self.sec_embed = nn.Embedding(opt.vocab_size + 1, opt.input_encoding_size)
        self.key_embed = nn.Linear(opt.vocab_size + 1, opt.input_encoding_size)
        self.linear = nn.Linear(opt.input_encoding_size, 1)
        self.linear1 = nn.Linear(41, opt.secret_size)

        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.kaiming_normal_(self.linear1.weight)

    def forward(self, sec_seq, key):
        seq_emd = self.sec_embed(sec_seq)
        key_emd = self.key_embed(key)
        seq_fea = torch.cat([seq_emd, key_emd], dim=1)
        seq_fea = self.linear(seq_fea)
        seq_fea = self.linear1(seq_fea.squeeze(2))
        return torch.sigmoid(seq_fea)

