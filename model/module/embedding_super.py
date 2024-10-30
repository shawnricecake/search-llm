import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Embedding_Super(nn.Module):
    def __init__(self, vocab_size, hidden_size, padding_idx):
        super(Embedding_Super, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx)
        self.weight = self.embedding.weight
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        # sampled_
        self.sample_hidden_size = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_embedding = None

    def set_sample_config(self, sample_hidden_size, sample_weight_start):
        self.sample_hidden_size = sample_hidden_size
        self.sampled_embedding = nn.Embedding(self.vocab_size, sample_hidden_size, self.padding_idx)

        self.sample_weight_start = int(sample_weight_start * (self.weight.shape[1] - sample_hidden_size))

        self.sampled_weight = self.weight[:, self.sample_weight_start:(self.sample_weight_start+sample_hidden_size)]
        self.sampled_embedding.weight = torch.nn.Parameter(self.sampled_weight)

    def forward(self, x):
        if self.sampled_embedding is None:
            return self.embedding(x)
        return self.sampled_embedding(x)

    def calc_sampled_param_num(self):
        return self.sampled_weight.numel() if self.sampled_weight is not None else self.weight.numel()
