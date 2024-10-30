import torch
import torch.nn as nn


class LlamaRMSNorm_Super(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

        # sample_
        self.sample_hidden_size = None
        self.sample_weight = None
        self.sample_weight_start = None

    def set_sample_config(self, sample_hidden_size, sample_weight_start):
        self.sample_hidden_size = sample_hidden_size

        self.sample_weight_start = int(sample_weight_start * (self.weight.shape[0] - sample_hidden_size))

        sample_weight_parameter = self.weight[self.sample_weight_start: (self.sample_weight_start + sample_hidden_size)]
        self.sample_weight = nn.Parameter(sample_weight_parameter).to(self.weight.device)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.sample_weight is None:
            # when layers are put on different gpus,
            # we need to add '.to(self.weight.device)' to avoid the computation on different device
            return self.weight * hidden_states.to(input_dtype).to(self.weight.device)
        return self.sample_weight * hidden_states.to(input_dtype)

    def calc_sampled_param_num(self):
        return self.sample_weight.numel() if self.sample_weight is not None else self.weight.numel()

