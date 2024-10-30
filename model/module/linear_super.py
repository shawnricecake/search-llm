import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Linear_Super(nn.Linear):
    def __init__(self, super_in_dim, super_out_dim, bias=True, num_head=None, head_dim=None, scale=False):
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None
        self.sample_weight_start = None

        self.samples = {}

        self.scale = scale

        # original
        self.num_head = num_head
        self.head_dim = head_dim

    def set_sample_config(
            self, sample_in_dim, sample_out_dim, sample_weight_start, qkv=False,
            sample_head_num=None, sample_head_dim=None
    ):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self.sample_weight_start = sample_weight_start

        if qkv:
            self._sample_parameters_qkv(sample_head_num, sample_head_dim)
        else:
            self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(
            self.weight, self.sample_in_dim, self.sample_out_dim, self.sample_weight_start
        )
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim / self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias(
                self.bias, self.sample_out_dim, self.sample_weight_start
            )
        return self.samples

    def _sample_parameters_qkv(self, sample_head_num, sample_head_dim):
        self.samples['weight'] = sample_weight_qkv(
            self.weight, self.sample_weight_start,
            sample_head_num, sample_head_dim,
            self.num_head, self.head_dim
        )
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim / self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias_qkv(
                self.bias, self.sample_weight_start,
                sample_head_num, sample_head_dim,
                self.num_head, self.head_dim
            )
        return self.samples

    def forward(self, x):
        if len(self.samples) == 0:
            return F.linear(x, self.weight, self.bias)
        return F.linear(
            x,
            self.samples['weight'].to(x.device),
            self.samples['bias'].to(x.device) if self.samples['bias'] is not None else None
        ) * (self.sample_scale if self.scale else 1)

    def calc_sampled_param_num(self):
        if 'weight' not in self.samples.keys():
            return self.weight.numel() + (self.bias.numel() if self.bias else 0)
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length * np.prod(self.samples['weight'].size())
        return total_flops


def sample_weight(weight, sample_in_dim, sample_out_dim, sample_weight_start):
    sample_weight_start_column = int(sample_weight_start * (weight.shape[1] - sample_in_dim))
    sample_weight_start_row = int(sample_weight_start * (weight.shape[0] - sample_out_dim))

    sample_weight = weight[:, sample_weight_start_column:sample_in_dim + sample_weight_start_column]
    sample_weight = sample_weight[sample_weight_start_row:sample_out_dim + sample_weight_start_row, :]

    return sample_weight


def sample_bias(bias, sample_out_dim, sample_weight_start):
    sample_weight_start = int(sample_weight_start * (bias.shape[0] - sample_out_dim))

    sample_bias = bias[sample_weight_start:sample_out_dim + sample_weight_start]

    return sample_bias


def sample_weight_qkv(
        weight, sample_weight_start,
        sample_head_num, sample_head_dim,
        original_head_num, original_head_dim
):
    sample_weight_columns = []

    sample_weight_each_head_column_start = int((original_head_dim - sample_head_dim) * sample_weight_start)
    sample_weight_each_head_column_end = int(sample_weight_each_head_column_start + sample_head_dim)

    sample_head_start = int((original_head_num - sample_head_num) * sample_weight_start)

    for i in range(sample_head_start, sample_head_start+sample_head_num):
        for j in range(sample_weight_each_head_column_start, sample_weight_each_head_column_end):
            sample_weight_columns.append(i * original_head_dim + j)

    sample_weight = weight[sample_weight_columns, :]

    return sample_weight


def sample_bias_qkv(
        bias, sample_weight_start,
        sample_head_num, sample_head_dim,
        original_head_num, original_head_dim
):
    sample_weight_columns = []

    sample_weight_each_head_column_start = int((original_head_dim - sample_head_dim) * sample_weight_start)
    sample_weight_each_head_column_end = int(sample_weight_each_head_column_start + sample_head_dim)

    sample_weight_head_start = int((original_head_num - sample_head_num) * sample_weight_start)

    for i in range(sample_weight_head_start, sample_weight_head_start+sample_head_num):
        for j in range(sample_weight_each_head_column_start, sample_weight_each_head_column_end):
            sample_weight_columns.append(i * original_head_dim + j)

    sample_bias = bias[sample_weight_columns]

    return sample_bias
