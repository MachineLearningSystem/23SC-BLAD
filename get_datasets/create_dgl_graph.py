# Rewrite from roland (https://github.com/snap-stanford/roland.git)
import os
os.environ['MPLCONFIGDIR'] = "/tmp"
import dgl
from graphgym.cmd_args import parse_args
from graphgym.config import (cfg, assert_cfg, dump_cfg,
                             update_out_dir)

import torch
import random
import numpy as np

import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from graphgym.loader import load_dataset

def dataset_cfg_setup_live_update(name: str):
    if name in ['reddit-body.tsv']:
        cfg.dataset.format = 'reddit_hyperlink'
        cfg.dataset.edge_dim = 88
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'W'

    elif name in ['AS-733']:
        cfg.dataset.format = 'as'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'D'

def dataset_cfg_setup_fixed_split(name: str):
    if name in ['reddit-body.tsv']:
        cfg.dataset.format = 'reddit_hyperlink'
        cfg.dataset.edge_dim = 88
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.dataset.split = [0.7, 0.1, 0.2]
        cfg.transaction.snapshot_freq = 'D'

    elif name in ['AS-733']:
        cfg.dataset.format = 'as'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'D'
        cfg.train.start_compute_mrr = 81


def create_batched_dataset(datasets):
    batch_len = 9
    len_snapshot = len(datasets)
    snapshot_window = 6
    batch_data = []
    for i in range(0, len_snapshot - snapshot_window, batch_len):
        batch_window = []
        for j in range(0, snapshot_window):
            batch_snap = []
            for k in range(0, batch_len):
                batch_snap.append(datasets[k + i + j])
            batch_snap = dgl.batch(batch_snap)
            batch_window.append(batch_snap)
        
        batch_data.append(batch_window)
    
    return batch_data


def cal_load_index(datasets, index, snapshot_len, batch_size):
    load = 0
    for i in range(0, snapshot_len):
        for j in range(0, batch_size):
            load = load + datasets[index + i + j].num_edges()
    
    return load



if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Repeat for different random seeds
    # Load config file
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    assert_cfg(cfg)

    # over-ride data path and remark if required.
    if args.override_data_dir is not None:
        cfg.dataset.dir = args.override_data_dir
    if args.override_remark is not None:
        cfg.remark = args.override_remark

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    out_dir_parent = cfg.out_dir
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    update_out_dir(out_dir_parent, args.cfg_file)
    dump_cfg(cfg)
        # Infer dataset format if required, retrieve the cfg setting associated
        # with the particular dataset.
    if cfg.dataset.format == 'infer':
        if cfg.train.mode in ['baseline', 'baseline_v2', 'live_update_fixed_split']:
            dataset_cfg_setup_fixed_split(cfg.dataset.name)
        elif cfg.train.mode == 'live_update':
            dataset_cfg_setup_live_update(cfg.dataset.name)
        elif cfg.train.mode == 'live_update_baseline':
            dataset_cfg_setup_live_update(cfg.dataset.name)

    graphs = load_dataset()

    len_graph = len(graphs)
    dataset = 'reddit-body'
    num_edges = []
    dgl_graph_snapshots = []
    for i in tqdm(range(0, len_graph)):
        file = f'../data/{dataset}/full/snapshot{i}.bin'
        num_nodes = graphs[i].num_nodes
        edge_index = graphs[i].edge_index
        node_feature = graphs[i].node_feature
        node_label_index = graphs[i].node_label_index
        dgl_graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
        dgl_graph = dgl.to_bidirected(dgl_graph)
        dgl_graph.ndata['feat'] = node_feature
        dgl_graph.ndata['label'] = node_label_index
        dgl.save_graphs(file, dgl_graph)
        num_edges.append(dgl_graph.num_edges())

        dgl_graph_snapshots.append(dgl_graph)
        
    num_edges = np.array(num_edges)
    np.savetxt('tmp.txt', num_edges)
    print(torch.max(node_label_index))

