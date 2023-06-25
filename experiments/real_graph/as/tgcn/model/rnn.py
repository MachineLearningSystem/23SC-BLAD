import torch
import torch.nn as nn
from torch.nn import init


class GRUGate(nn.Module):
    def __init__(self, in_feat, out_feat, activation=None):
        super(GRUGate, self).__init__()
        self.activation = activation
        self.W = nn.parameter.Parameter(torch.Tensor(in_feat, in_feat))
        self.U = nn.parameter.Parameter(torch.Tensor(in_feat, in_feat))
        self.bias = nn.parameter.Parameter(torch.Tensor(in_feat, out_feat))
        self.reset_parameters()
    
    def reset_parameters(self):
        init.xavier_uniform_(self.W)
        init.xavier_uniform_(self.U)
        init.zeros_(self.bias)
    
    def forward(self, input, hidden):
        out = self.W.matmul(input) + self.U.matmul(hidden) + self.bias
        out = self.activation(out)
        return out

class GRUCell(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GRUCell, self).__init__()
        self.update = GRUGate(in_feats, out_feats, nn.Sigmoid())
        self.reset = GRUGate(in_feats, out_feats, nn.Sigmoid())
        self.htilda = GRUGate(in_feats, out_feats, nn.Tanh())
    
    def forward(self, prev_q, z_topk=None):
        if(z_topk == None):
            z_topk = prev_q
        
        update = self.update(z_topk, prev_q)
        reset = self.reset(z_topk, prev_q)

        h_cap = reset * prev_q
        h_cap = self.htilda(z_topk, h_cap)

        new_q = (1 - update) * prev_q + update * h_cap

        return new_q

