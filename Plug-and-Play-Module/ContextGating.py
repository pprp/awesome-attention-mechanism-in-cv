import torch
import torch.nn as nn
import math

'''
Learnable pooling with Context Gating for video classification
arXiv:1706.06905v2
source: https://github.com/ekingedik/loupe-pytorch/blob/master/loupe_pytorch.py
'''
class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(nn.init.normal_(
            torch.empty(dim, dim), mean=0, std=(1 / math.sqrt(dim)),))
        self.sigmoid = nn.Sigmoid()
        if add_batch_norm:
            self.gating_biases = None
            self.batch_norm = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(torch.nn.init.normal_(
                torch.empty(dim), mean=0, std=(1 / math.sqrt(dim)),))
            self.batch_norm = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)
        if self.add_batch_norm:
            gates = self.batch_norm(gates)
        else:
            gates = gates + self.gating_biases
        gates = self.sigmoid(gates)
        activation = x * gates
        return activation
