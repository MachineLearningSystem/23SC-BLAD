import torch
import torch.nn as nn
from torch.nn import init
from dgl.nn.pytorch import GraphConv
from torch.nn.parameter import Parameter


class GRUCell(torch.nn.Module):
    """
    GRU cell for matrix, similar to the official code.
    Please refer to section 3.4 of the paper for the formula.
    """

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.update = GRUGate(in_feats,
                                 out_feats,
                                 torch.nn.Sigmoid())

        self.reset = GRUGate(in_feats,
                                out_feats,
                                torch.nn.Sigmoid())

        self.htilda = GRUGate(in_feats,
                                 out_feats,
                                 torch.nn.Tanh())

    def forward(self, prev_Q, z_topk=None):
        if z_topk is None:
            z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class GRUGate(torch.nn.Module):
    """
    GRU gate for matrix, similar to the official code.
    Please refer to section 3.4 of the paper for the formula.
    """

    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows, rows))
        self.U = Parameter(torch.Tensor(rows, rows))
        self.bias = Parameter(torch.Tensor(rows, cols))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.W)
        init.xavier_uniform_(self.U)
        init.zeros_(self.bias)

    def forward(self, x, hidden):
        out = self.activation(torch.matmul(x, self.W) + \
                              torch.matmul(hidden, self.U))# + \

        return out