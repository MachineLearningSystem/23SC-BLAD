import torch
import dgl
import torch.nn as nn
import dgl.function as fn
from torch.nn import init
from time import time

class gcn_backbones(nn.Module):
    def __init__(self, in_feat, out_feat, weight=False, activation=None):
        super(gcn_backbones, self).__init__()
        if(weight != False):
            self.weight = nn.parameter.Parameter(torch.Tensor(in_feat, out_feat))
        else:
            self.register_parameter('weight', None)
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)

    def forward(self, feat, weight=None):
        if(weight == None):
            weight = self.weight
        feat = torch.matmul(feat, weight)
        if(self.activation != None):
            feat = self.activation(feat)

        return feat

class gather(nn.Module):
    def __init__(self):
        super(gather, self).__init__()
    
    def forward(self, graph, feat):
        graph.srcdata['h'] = feat
        aggregate_fn = fn.copy_src('h', 'm')
        graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
        rst = graph.dstdata['h']
        if(graph.is_block == True):
            graph.srcdata.pop('h')
        graph.dstdata.pop('h')
        return rst

class GCN(nn.Module):
    def __init__(self, in_feat, out_feat, activation=None, weight=False):
        super(GCN, self).__init__()
        self.linear1 = gcn_backbones(in_feat, out_feat, weight=weight, activation=None)
        self.gather = gather()
        self.linear2 = gcn_backbones(out_feat, out_feat, weight=weight, activation=None)
        self.relu = nn.RReLU()
    
    def forward(self, graph, feat, weights=None, pattern=0):
        if(isinstance(graph, list) == True):
            graph1 = graph[0]
            graph2 = graph[1]
        else:
            graph1 = graph
            graph2 = graph
        if(weights != None):
            torch.cuda.synchronize()
            start1 = time()
            pattern1 = int(pattern/2)
            pattern2 = pattern % 2
            if(pattern1 == 0):
                feat = self.linear1(feat, weights[0])
                torch.cuda.synchronize()
                start2 = time()
                feat = self.gather(graph1, feat)
                torch.cuda.synchronize()
                start3 = time()
            else:
                feat = self.gather(graph1, feat)
                torch.cuda.synchronize()
                start2 = time()
                feat = self.linear1(feat, weights[0])
                torch.cuda.synchronize()
                start3 = time()
            feat = self.relu(feat)
            if(pattern2 == 0):
                feat = self.linear2(feat, weights[1])
                torch.cuda.synchronize()
                start4 = time()
                feat = self.gather(graph2, feat)
                torch.cuda.synchronize()
                start5 = time()
            else:
                feat = self.gather(graph2, feat)
                torch.cuda.synchronize()
                start4 = time()
                feat = self.linear2(feat, weights[1])
                torch.cuda.synchronize()
                start5 = time()
            feat = self.relu(feat)
            #torch.cuda.synchronize()
            #start2 = time()
                
            #torch.cuda.synchronize()
            #start3 = time()
            #torch.cuda.synchronize()
            #start4 = time()
            end = time()
            #torch.cuda.synchronize()
            #end = time()

            return feat, [start2 - start1, start3 - start2, start4 - start3, start5 - start4]
        else:
            torch.cuda.synchronize()
            start1 = time()
            pattern1 = int(pattern/2)
            pattern2 = pattern % 2
            if(pattern1 == 0):
                feat = self.linear1(feat)
                torch.cuda.synchronize()
                start2 = time()
                feat = self.gather(graph1, feat)
                torch.cuda.synchronize()
                start3 = time()
                feat1 = feat
            else:
                feat = self.gather(graph1, feat)
                torch.cuda.synchronize()
                start2 = time()
                feat = self.linear1(feat)
                torch.cuda.synchronize()
                start3 = time()
                feat1 = feat
            feat = self.relu(feat)
            if(pattern2 == 0):
                feat = self.linear2(feat)
                torch.cuda.synchronize()
                start4 = time()
                feat = self.gather(graph2, feat)
                torch.cuda.synchronize()
                start5 = time()
            else:
                feat = self.gather(graph2, feat)
                torch.cuda.synchronize()
                start4 = time()
                feat = self.linear2(feat)
                torch.cuda.synchronize()
                start5 = time()
            feat = self.relu(feat)
            #torch.cuda.synchronize()
            #start2 = time()
                
            #torch.cuda.synchronize()
            #start3 = time()
            #torch.cuda.synchronize()
            #start4 = time()
            end = time()
            #torch.cuda.synchronize()
            #end = time()

            return feat, feat1, [start2 - start1, start3 - start2, start4 - start3, start5 - start4]

