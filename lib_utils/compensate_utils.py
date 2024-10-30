import math
import time
import torch
import torch.nn as nn
import transformers

from model.module.linear_super import Linear_Super


def find_layers(module, layers=[nn.Conv2d, nn.Linear, Linear_Super], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEBUG = False


class CompensateGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.out_dim = layer.weight.data.shape[0]
        self.in_dim = layer.weight.data.shape[1]
        self.nsamples = 0

        # importance score
        self.baseline_inp = torch.zeros((self.in_dim), device=self.dev)
        self.fluc_inp = torch.zeros((self.in_dim), device=self.dev)

        # admm
        self.XX = torch.zeros((self.in_dim, self.in_dim), device=self.dev)

    def add_batch(self, inp, out):
        inp_original = inp
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        # importance score
        old_baseline_inp = self.baseline_inp
        self.baseline_inp *= self.nsamples / (self.nsamples + batch_size)
        self.baseline_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
        if self.nsamples == 0:
            self.fluc_inp = 0
        else:
            self.fluc_inp *= (self.nsamples - 1) / (self.nsamples + batch_size - 1)
            self.fluc_inp += torch.sum(
                (inp - self.baseline_inp.unsqueeze(1)) * (inp - old_baseline_inp.unsqueeze(1)), dim=1
            ) / (self.nsamples + batch_size)

        self.nsamples += batch_size

        # admm
        X = inp_original.reshape(-1, inp_original.shape[-1]).float()
        self.XX += X.T.matmul(X)

    def admm_optimize(self, percdamp=.1, iters=20):
        mask = self.layer.weight == 0
        mask = mask.to(torch.int)
        if torch.sum(mask) == mask.shape[0] * mask.shape[1]:
            print("jump the optimization for this layer because of the 0 sparsity")
            print()

        XX = self.XX
        norm = torch.diag(XX).sqrt() + 1e-8
        XX = XX / norm
        XX = (XX.T / norm).T
        W = (self.layer.weight.float().detach() * norm).T

        rho0 = percdamp * torch.diag(XX).mean()
        diag = torch.arange(XX.shape[0], device=XX.device)
        XX[diag, diag] += rho0

        rho = 1e1

        XY = XX.matmul(W)
        XX[diag, diag] += rho
        torch.cuda.empty_cache()
        XXinv = torch.inverse(XX)
        self.XX = None
        del XX

        U = torch.zeros_like(W)
        for itt in range(iters):
            Z = (W + U) * mask.T

            U = U + (W - Z)

            W = XXinv.matmul(XY + rho * (Z - U))

        Z = (W + U) * mask.T
        out = (Z.T / norm)

        print("double check current optimized weight sparsity: {:.2f}".format(
            (out == 0).sum().item() / out.numel() * 100
        ))
        print()

        self.layer.weight.data = out.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()