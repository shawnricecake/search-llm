import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers.activations import ACT2FN
from model.module.linear_super import Linear_Super


class LlamaMLP_Super(nn.Module):
    def __init__(
            self, config,
            hidden_size,
            intermediate_size,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = Linear_Super(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = Linear_Super(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = Linear_Super(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        # sample_
        self.sample_hidden_size = None
        self.sample_hidden_size_out = None
        self.sample_intermediate_size = None
        self.sample_weight_start = None

    def set_sample_config(self, sample_hidden_size, sample_hidden_size_out, sample_intermediate_size, sample_weight_start):
        self.sample_hidden_size = sample_hidden_size
        self.sample_hidden_size_out = sample_hidden_size_out
        self.sample_intermediate_size = sample_intermediate_size
        self.sample_weight_start = sample_weight_start
        self.gate_proj.set_sample_config(self.hidden_size, sample_intermediate_size, sample_weight_start)
        self.up_proj.set_sample_config(self.hidden_size, sample_intermediate_size, sample_weight_start)
        self.down_proj.set_sample_config(sample_intermediate_size, self.hidden_size, sample_weight_start)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))