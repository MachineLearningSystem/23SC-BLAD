import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from torch.nn import init

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i_t = LSTMGate(input_size, hidden_size, torch.nn.Sigmoid())
        self.f_t = LSTMGate(input_size, hidden_size, torch.nn.Sigmoid())
        self.c_t = LSTMGate(input_size, hidden_size, torch.nn.Tanh())
        self.o_t = LSTMGate(input_size, hidden_size, torch.nn.Sigmoid())

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, feat, z_topk=None):

        if z_topk is None:
            z_topk = feat
        else:
            z_shape = z_topk.shape[0]
            f_shape = feat.shape[0]
            if(z_shape > f_shape):
                z_topk = z_topk[0:f_shape,:]
            else:
                tmp = feat[z_shape:f_shape,:]
                z_topk = torch.cat([z_topk, tmp], dim=0)

        x_t = feat
        i_t = self.i_t(x_t, z_topk)
        f_t = self.f_t(x_t, z_topk)
        g_t = self.c_t(x_t, z_topk)
        c_t = f_t * z_topk + i_t * g_t

        o_t = self.o_t(x_t, z_topk)
        newh = o_t * torch.tanh(c_t)
        return newh, newh

class LSTMGate(torch.nn.Module):
    """
    GRU gate for matrix, similar to the official code.
    Please refer to section 3.4 of the paper for the formula.
    """

    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows, cols))
        self.U = Parameter(torch.Tensor(rows, cols))
        self.bias = Parameter(torch.Tensor(cols))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.W)
        init.xavier_uniform_(self.U)
        init.zeros_(self.bias)

    def forward(self, x, hidden):
        out = self.activation(torch.matmul(x, self.W) + \
                              torch.matmul(hidden, self.U))# + \

        return out