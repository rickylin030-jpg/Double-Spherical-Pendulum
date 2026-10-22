import torch
import torch.nn as nn
import math

def FCtanh(chin, chout, zero_bias=False, orthogonal_init=False):
    return nn.Sequential(Linear(chin, chout, zero_bias, orthogonal_init), nn.Tanh())

def Linear(chin, chout, zero_bias=False, orthogonal_init=False):
    linear = nn.Linear(chin, chout)
    if zero_bias:
        torch.nn.init.zeros_(linear.bias)
    if orthogonal_init:
        torch.nn.init.orthogonal_(linear.weight)
    return linear

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)