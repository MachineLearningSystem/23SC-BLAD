import dgl.distributed as dist
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
import torch
import numpy as np


def load_reddit():
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=True)
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    return g, data.num_labels

'''
g, labels = load_reddit()
data = g
feat = g.ndata['feat']
train_idx = torch.where(g.ndata['train_mask'] == True)[0]
val_idx = torch.where(g.ndata['val_mask'] == True)[0]
test_idx = torch.where(g.ndata['test_mask'] == True)[0]

graph = g
'''
dataset = DglNodePropPredDataset('ogbn-arxiv', root='../dataset/')
split_idx = dataset.get_idx_split()
train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
all_idx = torch.cat([train_idx, val_idx, test_idx])

data = dataset[0]
graph = data[0]

orig_nids, orig_eids = dgl.distributed.partition_graph(graph, 'arxiv-partition', 2, 'arxiv/', return_mapping=True, reshuffle=False)


#node_map = np.load('node_map.npy')
#print(node_map.shape)
#a = 1
