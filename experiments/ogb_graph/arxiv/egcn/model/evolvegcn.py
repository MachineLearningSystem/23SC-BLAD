import torch
import torch.nn as nn
import model.gcn as gcn
import model.rnn as rnn
from torch.nn import init
import numpy as np
from time import time


class EvolveGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, class_feat, num_layers=2):
        super(EvolveGCN, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.ModuleList()
        self.gnn = nn.ModuleList()
        self.gnn_weight = nn.ParameterList()

        self.in_feat = in_feat
        self.hidden_feat = hidden_feat
        self.out_feat = out_feat
        self.rnn.append(rnn.GRUCell(in_feat, hidden_feat))
        self.gnn_weight.append(nn.parameter.Parameter(torch.Tensor(in_feat, hidden_feat)))
        self.gnn.append(gcn.GCN(in_feat, hidden_feat, nn.RReLU()))
        for _ in range(num_layers - 1):
            self.rnn.append(rnn.GRUCell(hidden_feat, hidden_feat-1))
            self.gnn_weight.append(nn.parameter.Parameter(torch.Tensor(hidden_feat, hidden_feat-1)))
        
        self.mlp = nn.Sequential(nn.Linear(hidden_feat-1, class_feat),
                                 nn.ReLU(),
                                 nn.Linear(class_feat, out_feat))

        self.reset_parameters()
    
    def reset_parameters(self):
        for gcn_weight in self.gnn_weight:
            init.xavier_uniform_(gcn_weight)
    
    def forward(self, g_list, feat_list, n_step, g_exe=None, W1=None, W2=None, pattern_list=None, full_is_pattern=False):
        if(W1 == None):
            self.reset_parameters()
            W1 = self.gnn_weight[0]
        if(W2 == None):
            W2 = self.gnn_weight[1]
        if(isinstance(g_exe, np.ndarray)):
            out = None
            for i, g in enumerate(g_list):
                if(i in g_exe):
                    W1 = self.rnn[0](W1)
                    W2 = self.rnn[1](W2)
                    weight = [W1, W2]
                    feat_list_end, gnn_fwd_time = self.gnn[0](g, feat_list[i], weights=weight, pattern=pattern_list[i]) 
                    if(i == n_step):
                        out = self.mlp(feat_list_end) 
                else:
                    continue
            return out, W1, W2
        else:
            pattern = 0
            if(full_is_pattern != 0):
                pattern = np.random.randint(0, 4)
            #pattern = np.random.randint(0, 4)
            fwd_time = []
            self.w1_fwd_tensors = []
            self.w2_fwd_tensors = []
            self.w1_results = []
            self.w2_results = []
            self.w1_gradients = []
            self.w2_gradients = []
            for i, g in enumerate(g_list):
                #W1 = W1.detach().clone().requires_grad_(True)
                #W2 = W2.detach().clone().requires_grad_(True)
                self.w1_fwd_tensors.append(W1)
                self.w2_fwd_tensors.append(W2)

                torch.cuda.synchronize()
                start1 = time()
                W1 = self.rnn[0](W1)
                W2 = self.rnn[1](W2)
                torch.cuda.synchronize()
                end = time()

                #rnn_fwd_time = end - start1

                weight = [W1, W2]
                feat_list_end, fwd_time1 = self.gnn[0](g, feat_list[i], weights=weight, pattern=pattern)  
                torch.cuda.synchronize()

                fwd_time1.append(end - start1)
                #fwd_time1 = [end - start1]

                fwd_time1.append(0)
                fwd_time.append(fwd_time1)
            
            
            #self.feat_end = feat_list[-1].detach().clone().requires_grad_(True)    
            out = self.mlp(feat_list_end)
            return feat_list_end, fwd_time, feat_list, out

    def backward(self, feature_end):

        torch.cuda.synchronize()
        start1 = time()
        torch.autograd.backward(feature_end)
        torch.cuda.synchronize()
        end = time()
        bwd_time = (end - start1)
        '''
        for num in range(len(self.w1_fwd_tensors) - 2, -1, -1):
            w1_gradients = self.w1_fwd_tensors[num + 1]
            w2_gradients = self.w2_fwd_tensors[num + 1]
            torch.cuda.synchronize()
            start1 = time()
            torch.autograd.backward(self.w1_results[num], w1_gradients)
            torch.autograd.backward(self.w2_results[num], w2_gradients)
            torch.cuda.synchronize()
            end1 = time()
            bwd_time.append(end1 - start1)
            
        bwd_time = np.array(bwd_time)
        bwd_time[0] = np.average(bwd_time[1:])    
        '''
        return bwd_time

    

            
