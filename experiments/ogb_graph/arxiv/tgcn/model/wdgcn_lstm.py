import torch
import torch.nn as nn
import model.gcn as gcn
import model.lstmcell as rnn
from torch.nn import init
import numpy as np
from time import time


class WDGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, class_feat, num_layers=2):
        super(WDGCN, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.ModuleList()
        self.gnn = nn.ModuleList()

        self.in_feat = in_feat
        self.hidden_feat = hidden_feat
        self.out_feat = out_feat
        self.rnn.append(rnn.LSTMCell(hidden_feat, hidden_feat))
        self.gnn.append(gcn.GCN(in_feat, hidden_feat, nn.ReLU(), weight=True))
        for _ in range(num_layers - 1):
            self.rnn.append(rnn.LSTMCell(hidden_feat, hidden_feat))
        
        self.mlp = nn.Sequential(nn.Linear(hidden_feat, class_feat),
                                 nn.ReLU(),
                                 nn.Linear(class_feat, out_feat))

    
    def forward(self, g_list, feat_list, n_step, g_exe=None, pattern_list=None, forward=False, prevs=None, full_is_pattern=0):

        if(isinstance(g_exe, np.ndarray)):
            out = None
            hidden_state = prevs
            for i, g in enumerate(g_list):
                if(i in g_exe):
                    if(forward == True):
                        feat_list_end = None
                        feat_list_end, feat_list_end1, gnn_fwd_time = self.gnn[0](g, feat_list[i], pattern=pattern_list[i]) 
                        feat_list_end, hidden_state = self.rnn[1](feat_list_end, hidden_state)   
                            
                    else:
                        feat_list_end = None
                        feat_list_end, feat_list_end1, gnn_fwd_time = self.gnn[0](g, feat_list[i], pattern=pattern_list[i]) 
                        feat_list_end, hidden_state = self.rnn[1](feat_list_end, hidden_state)
                    if(i == n_step):
                        feat_list_end = self.mlp(feat_list_end) 
                else:
                    continue
            return feat_list_end
        else:
            pattern = 0
            if(full_is_pattern != 0):
                pattern = np.random.randint(0, 4)
            pattern = 0
            fwd_time = []
            fwd_time = []
            self.feat_list_tensors = []
            self.feat_list_results = []
            self.feat_list_gradients = []
            hidden_state = None
            for i, g in enumerate(g_list):
                
                #rnn_fwd_time = end - start1
                torch.cuda.synchronize()
                feat_list_end, feat_list_end1, fwd_time1 = self.gnn[0](g, feat_list[i], pattern=pattern)  
                torch.cuda.synchronize()
                start_rnn = time()
                #feat_list_end1 = self.rnn[0](feat_list_end1)
                feat_list_end, hidden_state = self.rnn[1](feat_list_end, hidden_state)
                torch.cuda.synchronize()
                end_rnn = time()

                #fwd_time1 = [end - start1]

                #self.feat_list_results.append(feat_list_end)
                #self.feat_list_gradients.append(None)
                fwd_time1.append(end_rnn - start_rnn)
                fwd_time1.append(0)
                fwd_time.append(fwd_time1)
            
            
            #self.feat_end = feat_list[-1].detach().clone().requires_grad_(True)    
            out = self.mlp(feat_list_end)
            return feat_list_end, fwd_time, feat_list, out

    def backward(self, feature_end):
        bwd_time = []
        torch.cuda.synchronize()
        start1 = time()
        torch.autograd.backward(feature_end)
        torch.cuda.synchronize()
        end = time()
        bwd_time.append(end - start1)
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
    '''       
        bwd_time = np.array(bwd_time)
        #bwd_time[0] = np.average(bwd_time[1:])    
        return bwd_time[0]
    
    

            
