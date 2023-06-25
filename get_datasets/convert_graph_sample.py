import torch
import torch.nn as nn

import numpy as np

import dgl
from dgl.data.utils import load_graphs, save_graphs
from ogb.nodeproppred import DglNodePropPredDataset
from tqdm import tqdm
import argparse
import pandas as pd

torch.manual_seed(2)


def load_reddit():
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=True, raw_dir='../dataset')
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    return g, data.num_labels

def load_features(dataset):
    nameset = 'ogbn-{}'.format(dataset)
    dataset = DglNodePropPredDataset(name=nameset, root='../dataset/')
    data = dataset[0]
    labels=data[1]
    feat = data[0].ndata['feat']
    return feat, labels


def load_subtensor(nfeat, labels, seeds, input_nodes, device='cpu'):
    """
    Extracts features and labels for a subset of nodes
    """
    input_nodes = torch.tensor(input_nodes, dtype=torch.int64)
    seeds = torch.tensor(seeds, dtype=torch.int64)
    batch_inputs = nfeat[input_nodes]
    batch_labels = labels[seeds]
    return batch_inputs, batch_labels


def sample_products(instance):
    feat, labels = load_features('products')
    #node_map = np.load('tools/products/node_map.npy')
    #ids = torch.tensor(np.where(node_map == 0)[0])
    info = []
    for i in tqdm(range(0, 30)):
        file = '../data/products/full/snapshot{}.bin'.format(i)
        graph = load_graphs(file)[0][0]
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        train_ids = torch.arange(0, graph.num_nodes())
        batchsize = int(80/instance)
        dataloader = dgl.dataloading.NodeDataLoader(graph, indices=train_ids, graph_sampler=sampler, batch_size=batchsize, shuffle=True)
        for j in range(0, 5):
        #for j in range(0, 1200):
            file_name = f'../data/products/sample/sample{j}/snapshot{i}.bin'
            #graph_sample = load_graphs(file_name)
            input_nodes, output_nodes, blocks = next(iter(dataloader))
            inputs, batch_labels = load_subtensor(feat, labels, output_nodes, input_nodes)
            blocks[0].srcdata['feat'] = inputs
            blocks[1].dstdata['labels'] = batch_labels 
            info.append([blocks[0].num_src_nodes(), blocks[0].num_edges()])
            save_graphs(file_name, [dgl.block_to_graph(blocks[0]),dgl.block_to_graph(blocks[1])])
            #print(blocks[1].num_dst_nodes(), blocks[0].num_edges())
    info = np.array(info)
    return info


def sample_arxiv(instance):
    feat, labels = load_features('arxiv')
    #node_map = np.load('tools/arxiv/node_map.npy')
    #ids = torch.tensor(np.where(node_map == 0)[0])[0:83500]
    info = []
    for i in tqdm(range(0, 30)):
        file = '../data/arxiv/full/snapshot{}.bin'.format(i)
        graph = load_graphs(file)[0][0]
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        train_ids = torch.arange(0, graph.num_nodes())
        batchsize = int(512/instance)
        dataloader = dgl.dataloading.NodeDataLoader(graph, indices=train_ids, graph_sampler=sampler, batch_size=batchsize, shuffle=True)
        for j in range(0, 5):
            file_name = f'../data/arxiv/sample/sample{j}/snapshot{i}.bin'
            #graph_sample = load_graphs(file_name)
            input_nodes, output_nodes, blocks = next(iter(dataloader))
            inputs, batch_labels = load_subtensor(feat, labels, output_nodes, input_nodes)
            blocks[0].srcdata['feat'] = inputs
            blocks[1].dstdata['labels'] = batch_labels 
            info.append([blocks[0].num_src_nodes(), blocks[0].num_edges()])
            save_graphs(file_name, [dgl.block_to_graph(blocks[0]),dgl.block_to_graph(blocks[1])])
        #print(blocks[0].num_src_nodes(), blocks[0].num_edges())
    info = np.array(info)
    return info

def sample_reddit(instance):
    g, labels = load_reddit()
    feat = g.ndata['feat']
    labels = g.ndata['labels']
    #node_map = np.load('tools/reddit/node_map.npy')
    #ids = torch.tensor(np.where(node_map == 0)[0])
    info = []
    for i in tqdm(range(0, 30)):
        file = '../data/reddit/full/snapshot{}.bin'.format(i)
        graph = load_graphs(file)[0][0]
        #sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 90])
        train_ids = torch.arange(0, graph.num_nodes())
        batchsize = int(224/instance)
        dataloader = dgl.dataloading.NodeDataLoader(graph, indices=train_ids, graph_sampler=sampler, batch_size=batchsize, shuffle=True)
        for j in range(0, 5):
            file_name = f'../data/reddit/sample/sample{j}/snapshot{i}.bin'
            #graph_sample = load_graphs(file_name)
            input_nodes, output_nodes, blocks = next(iter(dataloader))
            inputs, batch_labels = load_subtensor(feat, labels, output_nodes, input_nodes)
            blocks[0].srcdata['feat'] = inputs
            blocks[1].dstdata['labels'] = batch_labels 
            save_graphs(file_name, [dgl.block_to_graph(blocks[0]),dgl.block_to_graph(blocks[1])])
            info.append([blocks[0].num_src_nodes(), blocks[0].num_edges()])
        #print(blocks[0].num_src_nodes(), blocks[0].num_edges())
    info = np.array(info)
    return info

def sample_reddit_body(instance):
    #feat, labels = load_features('products')
    #node_map = np.load('tools/products/node_map.npy')
    #ids = torch.tensor(np.where(node_map == 0)[0])
    info = []
    file = '../data/reddit-body/full/snapshot{}.bin'.format(0)
    graph = load_graphs(file)[0][0]
    feat = graph.ndata['feat']
    labels = graph.ndata['label']

    max_graph = None
    max_edges = -1
    
    num_nodes = graph.num_nodes()
    
    num_neighbor = np.zeros([num_nodes])
    for i in tqdm(range(0, 177)):
        file = '../data/reddit-body/full/snapshot{}.bin'.format(i)
        graph = load_graphs(file)[0][0]
        edges = graph.edges()[0]
        
        for item in edges:
            num_neighbor[item.item()] = num_neighbor[item.item()] + 1
    
    sample_index = np.where(num_neighbor > 150)[0]
    print(sample_index.shape[0])




    for i in tqdm(range(0, 177)):
        file = '../data/reddit-body/full/snapshot{}.bin'.format(i)
        graph = load_graphs(file)[0][0]
        sampler = dgl.dataloading.MultiLayerNeighborSampler([200, 200])
        train_ids = torch.arange(0, graph.num_nodes())
        batchsize = int(512/instance)
        dataloader = dgl.dataloading.NodeDataLoader(graph, indices=sample_index, graph_sampler=sampler, batch_size=batchsize, shuffle=True)
        #for j in range(0, 7):
        for j in range(0, 1):
            file_name = f'../data/reddit-body/sample/sample{j}/snapshot{i}.bin'
            #graph_sample = load_graphs(file_name)
            input_nodes, output_nodes, blocks = next(iter(dataloader))
            inputs, batch_labels = load_subtensor(feat, labels, output_nodes, input_nodes)
            blocks[0].srcdata['feat'] = inputs
            blocks[1].dstdata['labels'] = batch_labels 
            if(j == 0):
                info.append([blocks[0].num_src_nodes(), blocks[0].num_edges()])

            block_0 = dgl.block_to_graph(blocks[0])
            block_1 = dgl.block_to_graph(blocks[1])
            save_graphs(file_name, [block_0, block_1])
        #print(blocks[0].num_src_nodes(), blocks[0].num_edges())
    info = np.array(info)
    return info


def sample_as(instance):
    #feat, labels = load_features('products')
    #node_map = np.load('tools/products/node_map.npy')
    #ids = torch.tensor(np.where(node_map == 0)[0])
    info = []
    file = '../data/AS/full/snapshot{}.bin'.format(0)
    graph = load_graphs(file)[0][0]
    feat = graph.ndata['feat']
    labels = graph.ndata['label']

    max_graph = None
    max_edges = -1
    
    num_nodes = graph.num_nodes()
    
    num_neighbor = np.zeros([num_nodes])
    for i in tqdm(range(0, 733)):
        #file = 'data/products/full/snapshot{}.bin'.format(i)
        file = '../data/AS/full/snapshot{}.bin'.format(i)
        graph = load_graphs(file)[0][0]
        edges = graph.edges()[0]
        
        for item in edges:
            num_neighbor[item.item()] = num_neighbor[item.item()] + 1
    
    sample_index = np.where(num_neighbor > 1750)[0]
    print(sample_index.shape[0])

    for i in tqdm(range(0, 733)):
        #file = 'data/products/full/snapshot{}.bin'.format(i)
        file = '../data/AS/full/snapshot{}.bin'.format(i)
        graph = load_graphs(file)[0][0]
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        train_ids = torch.arange(0, graph.num_nodes())
        batchsize = int(1000/instance)
        dataloader = dgl.dataloading.NodeDataLoader(graph, indices=sample_index, graph_sampler=sampler, batch_size=batchsize, shuffle=True, drop_last=True)
        #for j in range(0, 7):
        for j in range(0, 1):
            file_name = f'../data/AS/sample/sample{j}/snapshot{i}.bin'
            #graph_sample = load_graphs(file_name)
            input_nodes, output_nodes, blocks = next(iter(dataloader))
            inputs, batch_labels = load_subtensor(feat, labels, output_nodes, input_nodes)
            blocks[0].srcdata['feat'] = inputs
            blocks[1].dstdata['labels'] = batch_labels 
            if(j == 0):
                info.append([blocks[0].num_src_nodes(), blocks[0].num_edges()])

            block_0 = dgl.block_to_graph(blocks[0])
            block_1 = dgl.block_to_graph(blocks[1])
            save_graphs(file_name, [block_0, block_1])
        #print(blocks[0].num_src_nodes(), blocks[0].num_edges())
    info = np.array(info)
    return info

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('--dataset', type=str, default='reddit-body') 
    parser.add_argument('--num-instance', type=int, default=1)
    args = parser.parse_args()
    dataset = args.dataset
    instance = args.num_instance
    if(dataset == 'reddit'):
        result = sample_reddit(instance)
    elif(dataset == 'products'):
        result = sample_products(instance)
    elif(dataset == 'arxiv'):
        result = sample_arxiv(instance)
    elif(dataset == 'reddit-body'):
        result = sample_reddit_body(instance)
    elif(dataset == 'AS'):
        result = sample_as(instance)

    df = pd.DataFrame(result)
    file_name = '{}_sample_info.csv'.format(dataset)
    df.to_csv(file_name)
    

    
