import argparse
from tqdm import tqdm
import numpy as np
import random
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset
import sklearn.preprocessing
import tracemalloc
import gc
import struct
from torch_sparse import coalesce
import math
import pdb
import time

import dgl
from dgl.data.utils import save_graphs



np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

def load_reddit():
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=True, raw_dir='../dataset/')
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    return g, data.num_labels

def dropout_adj(edge_index, rmnode_idx, edge_attr=None, force_undirected=True,
                num_nodes=None):

    N = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    row, col = edge_index
    
    if force_undirected:
        row, col, edge_attr = filter_adj(row, col, edge_attr, row < col)
    convert_start = time.time()
    row_convert = row.numpy().tolist()
    col_convert = col.numpy().tolist()
    convert_end = time.time()
    print('convert cost:', convert_end - convert_start)

    idx_bool = []
    nodeidx_dict = dict(zip(rmnode_idx.tolist(), np.zeros(rmnode_idx.shape)))
    for i in tqdm(range(row.shape[0])):
        idx_bool.append(row_convert[i] in nodeidx_dict or col_convert[i] in nodeidx_dict)
        if (i + 1) % 1e8 == 0:
            print(i + 1, 'finished')


    drop_mask = torch.tensor(idx_bool)
    del row_convert
    del col_convert
    del idx_bool
    del nodeidx_dict

    mask = (1-np.array(drop_mask)) 
    mask = torch.LongTensor(mask).to(torch.bool)

    new_row, new_col, edge_attr = filter_adj(row, col, edge_attr, mask)
    drop_row, drop_col, edge_attr = filter_adj(row, col, edge_attr, drop_mask)
    print('init:',len(new_row), ', drop:', len(drop_row))
    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([new_row, new_col], dim=0),
             torch.cat([new_col, new_row], dim=0)], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
    else:
        edge_index = torch.stack([new_row, new_col], dim=0)
    drop_edge_index = torch.stack([drop_row, drop_col], dim=0)  ### only u->v (no v->u)

    return edge_index, drop_edge_index, edge_attr

def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]

def arxiv():
    dataset = DglNodePropPredDataset(name='ogbn-arxiv', root='../dataset/')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    all_idx = torch.cat([train_idx, val_idx, test_idx])
    mask_idx = torch.cat([train_idx, test_idx])
    feat = data[0].ndata['feat']
    feat = np.array(feat,dtype=np.float64)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)
    np.save('../data/arxiv/arxiv_feat.npy',feat)
    
    #get labels
    labels=data[1]
    train_labels=labels.data[train_idx]
    val_labels=labels.data[val_idx]
    test_labels=labels.data[test_idx]

    train_idx=train_idx.numpy()
    val_idx=val_idx.numpy()
    test_idx=test_idx.numpy()
    train_idx=np.array(train_idx, dtype=np.int32)
    val_idx=np.array(val_idx,dtype=np.int32)
    test_idx=np.array(test_idx,dtype=np.int32)

    train_labels=train_labels.numpy().T
    val_labels=val_labels.numpy().T
    test_labels=test_labels.numpy().T

    train_labels=np.array(train_labels,dtype=np.int32)
    val_labels=np.array(val_labels,dtype=np.int32)
    test_labels=np.array(test_labels,dtype=np.int32)
    train_labels=train_labels.reshape(train_labels.shape[1])
    val_labels=val_labels.reshape(val_labels.shape[1])
    test_labels=test_labels.reshape(test_labels.shape[1])
    np.savez('../data/arxiv/arxiv_labels.npz',train_idx=train_idx,val_idx=val_idx,test_idx=test_idx,train_labels=train_labels,val_labels=val_labels,test_labels=test_labels)

    num_nodes = data[0].num_nodes()
    edge_src = data[0].edges()[0].unsqueeze(0)
    edge_dst = data[0].edges()[1].unsqueeze(0)
    edge_index = torch.cat([edge_src, edge_dst], 0)
    edge_index = to_undirected(edge_index, num_nodes)
    edge_index, drop_edge_index, _ = dropout_adj(edge_index, mask_idx, num_nodes= num_nodes)
    edge_index = to_undirected(edge_index, num_nodes)
    
    row_drop, col_drop = np.array(drop_edge_index)
    '''
    f = open('../data+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))
    f.close()
    '''
    row, col = edge_index
    print(row_drop.shape)
    row=row.numpy()
    col=col.numpy()
    
    row, col = edge_index
    print(row_drop.shape)
    row=row.numpy()
    col=col.numpy()
    a = dgl.graph((row, col))
    file_name = '../data/full/arxiv/snapshot0.bin'
    save_graphs(file_name, a)

    num_snap = 30 
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)
    for sn in (range(num_snap)):
        #print(sn)
        row_sn = row_drop[ sn*snapshot : (sn+1)*snapshot ]
        col_sn = col_drop[ sn*snapshot : (sn+1)*snapshot ]
        if sn == 0:
            row_tmp=row
            col_tmp=col

        row_tmp=np.concatenate((row_tmp,row_sn))
        col_tmp=np.concatenate((col_tmp,col_sn))
        row_tmp=np.concatenate((row_tmp,col_sn))
        col_tmp=np.concatenate((col_tmp,row_sn))

        a = dgl.graph((row_tmp, col_tmp))
        file_name = '../data/full/arxiv/snapshot{}.bin'.format(sn+1)
        print(a.num_nodes(), a.num_edges())
        save_graphs(file_name, a)

    print('Arxiv -- save snapshots finish')

def products():
    #dataset=PygNodePropPredDataset(name='ogbn-products')
    dataset = DglNodePropPredDataset(name='ogbn-products', root='../dataset/')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    all_idx = torch.cat([train_idx, val_idx, test_idx])
    mask_idx = torch.cat([train_idx, test_idx])
    #save feat
    #feat=data.x.numpy()
    feat = data[0].ndata['feat']
    feat = np.array(feat,dtype=np.float64)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)
    np.save('../data/products/products_feat.npy',feat)

    #get labels
    print("save labels.....")
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    
    labels=data[1]
    train_labels=labels.data[train_idx]
    val_labels=labels.data[val_idx]
    test_labels=labels.data[test_idx]

    train_idx=train_idx.numpy()
    val_idx=val_idx.numpy()
    test_idx=test_idx.numpy()
    train_idx=np.array(train_idx, dtype=np.int32)
    val_idx=np.array(val_idx,dtype=np.int32)
    test_idx=np.array(test_idx,dtype=np.int32)

    train_labels=train_labels.numpy().T
    val_labels=val_labels.numpy().T
    test_labels=test_labels.numpy().T

    train_labels=np.array(train_labels,dtype=np.int32)
    val_labels=np.array(val_labels,dtype=np.int32)
    test_labels=np.array(test_labels,dtype=np.int32)
    train_labels=train_labels.reshape(train_labels.shape[1])
    val_labels=val_labels.reshape(val_labels.shape[1])
    test_labels=test_labels.reshape(test_labels.shape[1])
    np.savez('../data/products/products_labels.npz',train_idx=train_idx,val_idx=val_idx,test_idx=test_idx,train_labels=train_labels,val_labels=val_labels,test_labels=test_labels)
    
    num_nodes = data[0].num_nodes()
    edge_src = data[0].edges()[0].unsqueeze(0)
    edge_dst = data[0].edges()[1].unsqueeze(0)
    edge_index = torch.cat([edge_src, edge_dst], 0)
    edge_index = to_undirected(edge_index, num_nodes)
    edge_index, drop_edge_index, _ = dropout_adj(edge_index, mask_idx, num_nodes= num_nodes)
    edge_index = to_undirected(edge_index, num_nodes)
    
    row_drop, col_drop = np.array(drop_edge_index)
    print('row_drop.shape:', row_drop.shape)
    '''
    f = open('data/products/ogbn-products_update_full.txt', 'w+')
    for k in tqdm(range(row_drop.shape[0])):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))
    f.close()
    '''
    row, col = edge_index
    print(row_drop.shape)
    row=row.numpy()
    col=col.numpy()
    a = dgl.graph((row, col))
    file_name = '../data/products/full/snapshot0.bin'
    save_graphs(file_name, a)
    #save_adj(row, col, N=num_nodes, dataset_name='products', savename='products_init', snap='init')
    #num_snap = 40
    num_snap = 30
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)

    for sn in tqdm(range(num_snap)):
        #print(sn)
        row_sn = row_drop[ sn*snapshot : (sn+1)*snapshot ]
        col_sn = col_drop[ sn*snapshot : (sn+1)*snapshot ]
        if sn == 0:
            row_tmp=row
            col_tmp=col

        row_tmp=np.concatenate((row_tmp,row_sn))
        col_tmp=np.concatenate((col_tmp,col_sn))
        row_tmp=np.concatenate((row_tmp,col_sn))
        col_tmp=np.concatenate((col_tmp,row_sn))

        a = dgl.graph((row_tmp, col_tmp))
        file_name = '../data/products/full/snapshot{}.bin'.format(sn+1)
        print(a.num_nodes(), a.num_edges())
        save_graphs(file_name, a)
        
        #save_adj(row_tmp, col_tmp, N=data.num_nodes, dataset_name='products', savename='products_snap'+str(sn+1), snap=(sn+1))
        '''
        with open('../data/products/products_Edgeupdate_snap' + str(sn+1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('Products -- save snapshots finish')
    '''

from utils.mag_utils import load_data
def mag():
    #dataset=PygNodePropPredDataset(name='ogbn-products')
    
    g, num_rels, num_classes, labels, feat, train_idx, test_idx, val_idx = load_data(
        'ogbn-mag', get_norm=True)
    all_idx = torch.cat([train_idx, val_idx, test_idx])
    mask_idx = torch.cat([train_idx, test_idx])
    #save feat
    #feat=data.x.numpy()
    feat = np.array(feat,dtype=np.float64)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)
    np.save('../data/mag/mag_feat.npy',feat)

    #get labels
    print("save labels.....")
    
    train_labels=labels.data[train_idx]
    val_labels=labels.data[val_idx]
    test_labels=labels.data[test_idx]

    train_idx=train_idx.numpy()
    val_idx=val_idx.numpy()
    test_idx=test_idx.numpy()
    train_idx=np.array(train_idx, dtype=np.int32)
    val_idx=np.array(val_idx,dtype=np.int32)
    test_idx=np.array(test_idx,dtype=np.int32)

    train_labels=train_labels.numpy().T
    val_labels=val_labels.numpy().T
    test_labels=test_labels.numpy().T

    train_labels=np.array(train_labels,dtype=np.int32)
    val_labels=np.array(val_labels,dtype=np.int32)
    test_labels=np.array(test_labels,dtype=np.int32)
    train_labels=train_labels.reshape(train_labels.shape[0])
    val_labels=val_labels.reshape(val_labels.shape[0])
    test_labels=test_labels.reshape(test_labels.shape[0])
    np.savez('../data/mag/mag_labels.npz',train_idx=train_idx,val_idx=val_idx,test_idx=test_idx,train_labels=train_labels,val_labels=val_labels,test_labels=test_labels)
    
    num_nodes = g.num_nodes()
    edge_src = g.edges()[0].unsqueeze(0)
    edge_dst = g.edges()[1].unsqueeze(0)
    edge_index = torch.cat([edge_src, edge_dst], 0)
    edge_index = to_undirected(edge_index, num_nodes)
    edge_index, drop_edge_index, _ = dropout_adj(edge_index, mask_idx, num_nodes= num_nodes)
    edge_index = to_undirected(edge_index, num_nodes)
    
    row_drop, col_drop = np.array(drop_edge_index)
    print('row_drop.shape:', row_drop.shape)
    '''
    f = open('data/products/ogbn-products_update_full.txt', 'w+')
    for k in tqdm(range(row_drop.shape[0])):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))
    f.close()
    '''
    row, col = edge_index
    print(row_drop.shape)
    row=row.numpy()
    col=col.numpy()
    a = dgl.graph((row, col))
    file_name = '../data/mag/full/snapshot0.bin'
    save_graphs(file_name, a)
    #save_adj(row, col, N=num_nodes, dataset_name='products', savename='products_init', snap='init')
    #num_snap = 40
    num_snap = 30
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)

    for sn in tqdm(range(num_snap)):
        #print(sn)
        row_sn = row_drop[ sn*snapshot : (sn+1)*snapshot ]
        col_sn = col_drop[ sn*snapshot : (sn+1)*snapshot ]
        if sn == 0:
            row_tmp=row
            col_tmp=col

        row_tmp=np.concatenate((row_tmp,row_sn))
        col_tmp=np.concatenate((col_tmp,col_sn))
        row_tmp=np.concatenate((row_tmp,col_sn))
        col_tmp=np.concatenate((col_tmp,row_sn))

        a = dgl.graph((row_tmp, col_tmp))
        file_name = '../data/mag/full/snapshot{}.bin'.format(sn+1)
        print(a.num_nodes(), a.num_edges())
        save_graphs(file_name, a)
        
        #save_adj(row_tmp, col_tmp, N=data.num_nodes, dataset_name='products', savename='products_snap'+str(sn+1), snap=(sn+1))
        '''
        with open('../data/products/products_Edgeupdate_snap' + str(sn+1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('Products -- save snapshots finish')
    '''


def reddit():
    s_time = time.time()
    
    g, labels = load_reddit()
    data = g
    feat = g.ndata['feat']
    train_idx = torch.where(g.ndata['train_mask'] == True)[0]
    val_idx = torch.where(g.ndata['val_mask'] == True)[0]
    test_idx = torch.where(g.ndata['test_mask'] == True)[0]

    mask_idx = torch.cat([train_idx, test_idx], 0)

    feat=np.array(feat,dtype=np.float64)

    #normalize feats
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)

    #save feats
    np.save('../data/reddit/reddit.npy',feat)
    del feat
    gc.collect()

    #get labels
    #train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    all_idx = torch.cat([train_idx, val_idx, test_idx])
    labels=g.ndata['labels']
    train_labels=labels.data[train_idx]
    val_labels=labels.data[val_idx]
    test_labels=labels.data[test_idx]

    train_idx=train_idx.numpy()
    val_idx=val_idx.numpy()
    test_idx=test_idx.numpy()
    train_idx=np.array(train_idx, dtype=np.int32)
    val_idx=np.array(val_idx,dtype=np.int32)
    test_idx=np.array(test_idx,dtype=np.int32)

    train_labels=train_labels.numpy().T
    val_labels=val_labels.numpy().T
    test_labels=test_labels.numpy().T

    train_labels=np.array(train_labels,dtype=np.int32)
    val_labels=np.array(val_labels,dtype=np.int32)
    test_labels=np.array(test_labels,dtype=np.int32)
    train_labels=train_labels.reshape(train_labels.shape[0])
    val_labels=val_labels.reshape(val_labels.shape[0])
    test_labels=test_labels.reshape(test_labels.shape[0])
    np.savez('../data/reddit/reddit_labels.npz',train_idx=train_idx,val_idx=val_idx,test_idx=test_idx,train_labels=train_labels,val_labels=val_labels,test_labels=test_labels)

    num_nodes = data.num_nodes()
    edge_src = data.edges()[0].unsqueeze(0)
    edge_dst = data.edges()[1].unsqueeze(0)
    edge_index = torch.cat([edge_src, edge_dst], 0)
    edge_index = to_undirected(edge_index, num_nodes)
    edge_index, drop_edge_index, _ = dropout_adj(edge_index, mask_idx, num_nodes= num_nodes)
    edge_index = to_undirected(edge_index, num_nodes)
    
    row_drop, col_drop = np.array(drop_edge_index)
    print('row_drop.shape:', row_drop.shape)
    '''
    f = open('data/products/ogbn-products_update_full.txt', 'w+')
    for k in tqdm(range(row_drop.shape[0])):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))
    f.close()
    '''
    row, col = edge_index
    print(row_drop.shape)
    row=row.numpy()
    col=col.numpy()
    a = dgl.graph((row, col))
    file_name = '../data/reddit/full/snapshot0.bin'
    save_graphs(file_name, a)
    #save_adj(row, col, N=num_nodes, dataset_name='products', savename='products_init', snap='init')
    #num_snap = 40
    num_snap = 30
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)

    for sn in tqdm(range(num_snap)):
        #print(sn)
        row_sn = row_drop[ sn*snapshot : (sn+1)*snapshot ]
        col_sn = col_drop[ sn*snapshot : (sn+1)*snapshot ]
        if sn == 0:
            row_tmp=row
            col_tmp=col

        row_tmp=np.concatenate((row_tmp,row_sn))
        col_tmp=np.concatenate((col_tmp,col_sn))
        row_tmp=np.concatenate((row_tmp,col_sn))
        col_tmp=np.concatenate((col_tmp,row_sn))

        a = dgl.graph((row_tmp, col_tmp))
        file_name = '../data/reddit/full/snapshot{}.bin'.format(sn+1)
        print(a.num_nodes(), a.num_edges())
        save_graphs(file_name, a)
        
        #save_adj(row_tmp, col_tmp, N=data.num_nodes, dataset_name='products', savename='products_snap'+str(sn+1), snap=(sn+1))
        '''
        with open('../data/products/products_Edgeupdate_snap' + str(sn+1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('Products -- save snapshots finish')
    '''



def save_adj(row, col, N, dataset_name, savename, snap, full=False):
    adj=sp.csr_matrix((np.ones(row.shape[0]),(row,col)),shape=(N,N))
    adj=adj+sp.eye(adj.shape[0])
    print('snap:',snap,', edge:',adj.nnz)
    save_path='../data/'+ dataset_name +'/'

    EL=adj.indices
    PL=adj.indptr

    del adj
    gc.collect()

    EL=np.array(EL,dtype=np.uint32)
    PL=np.array(PL,dtype=np.uint32)
    EL_re=[]

    for i in range(1,PL.shape[0]):
        EL_re+=sorted(EL[PL[i-1]:PL[i]],key=lambda x:PL[x+1]-PL[x])
    EL_re=np.asarray(EL_re,dtype=np.uint32)

    #save graph
    f1=open(save_path+savename+'_adj_el.txt','wb')
    for i in EL_re:
        m=struct.pack('I',i)
        f1.write(m)
    f1.close()

    f2=open(save_path+savename+'_adj_pl.txt','wb')
    for i in PL:
        m=struct.pack('I',i)
        f2.write(m)
    f2.close()
    del EL
    del PL
    del EL_re
    gc.collect()

if __name__ == "__main__":
    reddit()
    #mag()
    #products()
    #arxiv()
