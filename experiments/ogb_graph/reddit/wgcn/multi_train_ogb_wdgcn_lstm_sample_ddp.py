from torch.multiprocessing import set_start_method
set_start_method('spawn', force=True)

import argparse
import dgl
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from model.wdgcn_lstm import WDGCN
import time
import numpy as np
import communication_wdgcn as comm
import os
from tqdm import tqdm
import pandas as pd

from ogb.nodeproppred import DglNodePropPredDataset
from dgl import load_graphs

from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = str(12245)

inputs = {}
input_gradients = {}

def run_forward(model, time_window_size, world_size, cached_subgraph, cached_feat, cached_label, minibatch, rank, local_rank, step, communication, feat, g_list, feat_list, pattern_list):
    pipeline_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    # reddit
    # wdgcn-lstm
    #pipeline_list = [0, 0, 1, -1, 1, -1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,0,0,0,0,0]
    #pipeline_list = [0, -1, 2, -1, 2, -1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    rank_time_window_size = int((time_window_size+1)/2)
    #g_list = cached_subgraph[minibatch - time_window_size:minibatch + 1]
    if(rank == 0):
        a = 1
    if(g_list == None):
        feat_list = []
        g_list = []
        for num in range(minibatch - time_window_size, minibatch + 1):
            g_list_tmp = []
            g_list_tmp.append(cached_subgraph[num][0])
            g_list_tmp.append(cached_subgraph[num][1])
            g_list.append(g_list_tmp)
            feat_list.append(cached_feat[num + minibatch - time_window_size])
               
    else:
        tmp = g_list.pop(0)
        tmp_batch = [cached_subgraph[minibatch][0], cached_subgraph[minibatch][1]]
        g_list.append(tmp_batch)
        tmp = feat_list.pop(0)
        feat_list.append(cached_feat[minibatch])
    labels = None
    if(rank == 1):
        a = 1
    if(step == 1):
        labels = torch.ones(g_list[-1][1].num_dst_nodes(), dtype=torch.int64).to(local_rank)
        #labels = torch.ones(cached_label[minibatch].shape[0], dtype=torch.int64).to("cuda:0")
    g_index = np.arange(0, time_window_size + 1)
    if(step == 0):
        g_exe = g_index[step * rank_time_window_size:(step + 1) * rank_time_window_size + pipeline_list[minibatch-time_window_size]]
    else:
        g_exe = g_index[step * rank_time_window_size + pipeline_list[minibatch-time_window_size]:(step + 1) * rank_time_window_size]


    feat_list_exe = []
    pattern_list_exe = []
    g_list_exe = []
    for num, item in enumerate(g_list):
        if(num in g_exe):
            if(feat_list[num].device == torch.device('cpu')):
                feat_list[num] = feat_list[num].to(local_rank)
                g_list[num][0] = g_list[num][0].to(local_rank)
                g_list[num][1] = g_list[num][1].to(local_rank)
            g_list_exe.append(g_list[num])
            feat_list_exe.append(feat_list[num])
            pattern_list_exe.append(pattern_list[num])
        else:
            if(feat_list[num].device == torch.device(local_rank)):
                feat_list[num] = feat_list[num].to("cpu")
                g_list[num][0] = g_list[num][0].to("cpu")
                g_list[num][1] = g_list[num][1].to("cpu")
            g_list_exe.append(None)
            feat_list_exe.append(None)
            pattern_list_exe.append(None)
    
    
    torch.cuda.synchronize()
    dist.barrier()

    W1 = None
    W2 = None
    torch.cuda.synchronize()
    start111 = time.perf_counter()
    predictions_prevs = None
    if(step > 0):
        if(rank == 1):
            a = 1
        data, fwd_comm = communication.recv(tag=100)
        predictions_prevs = data[0]
    start222 = time.perf_counter()
    for num1 in range(0, 1):
        if(step == 0):
            predictions = model(g_list, feat_list_exe, time_window_size, g_exe, pattern_list_exe, forward=True, prevs=predictions_prevs)
        elif(step == 1):
            predictions = model(g_list, feat_list_exe, time_window_size, g_exe, pattern_list_exe, forward=False, prevs=predictions_prevs)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    if(step == 0):
        end111 = time.perf_counter()
        fwd_time = end111 - start222

    if(step == 0):
        data = [predictions]
        if(rank == 0):
            a = 1
        fwd_comm = communication.send(data, tag=100)

    else:
        #loss_class_weight = torch.Tensor([0.35, 0.65]).to("cuda:0")
        loss = F.cross_entropy(predictions, labels)#, weight=loss_class_weight)
        predictions = loss
        if(step == 1):
            end111 = time.perf_counter()
            fwd_time = end111 - start222
        def hook_wrapper(input_name):
            def hook(input_gradient):
                input_gradients[input_name] = input_gradient
                return hook
        #W1.register_hook(hook_wrapper(0))
        #W2.register_hook(hook_wrapper(1))
    return predictions, predictions_prevs, g_list, feat_list, fwd_time, start111, fwd_comm

def run_backward(input_data, out_data, world_size, rank, step, communication, group):

    torch.cuda.synchronize()
    
    start111 = time.perf_counter()
    if(step == 1):
        dist.barrier(group=group)
        torch.autograd.backward(out_data)
    else:
        data, bwd_comm = communication.recv(tag=101)
        predictions_prev = data[0]
        torch.cuda.synchronize()
        dist.barrier(group=group)
        start111 = time.perf_counter()
        torch.autograd.backward(out_data, predictions_prev)
        #torch.autograd.backward(out_data[0], W1_gradient)
        #torch.autograd.backward(out_data[1], W2_gradient)
    torch.cuda.synchronize()
    end111 = time.perf_counter()
    if(step == 1):
        input_data_gradient = [input_data.grad]
        if(rank == 0):
            a = 1
        bwd_comm = communication.send(input_data_gradient, tag=101)
    bwd_time = end111 - start111
    return bwd_time, bwd_comm

def train_model(args, rank, local_rank, world_size, fwd_shared_memory, bwd_shared_memory, pattern_list):

    dataset = DglNodePropPredDataset('ogbn-arxiv', root='../../../../dataset/')
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    all_idx = torch.cat([train_idx, val_idx, test_idx])

    data = dataset[0]
    feat = data[0].ndata['feat']
    train_feat = feat[train_idx]
    labels=data[1]
    train_labels = labels[train_idx]
    cached_subgraph = []
    cached_feat = []
    cached_label = []
    result = []
    folder_name = 'typo_data1/typo{}{}{}{}{}{}'.format(pattern_list[0],pattern_list[1],pattern_list[2],pattern_list[3],pattern_list[4],pattern_list[5])
    #x = [i for i in range(29, -1, -1)]
    x = [i for i in range(0, 30)]
    
    for i in range(0, 30):
        # we add self loop edge when we construct full graph, not here
        graph_name = '../../../../data/reddit/sample/sample1/snapshot{}.bin'.format(x[i])
        node_subgraph = load_graphs(graph_name)[0]
        blocks = [None, None]
        blocks[0] = dgl.create_block(node_subgraph[0].edges())
        blocks[1] = dgl.create_block(node_subgraph[1].edges())
        cached_feat.append(node_subgraph[0].srcdata['feat'])
        cached_label.append(node_subgraph[1].dstdata['labels'])

        #node_subgraph = dgl.node_subgraph(graph=g, nodes=node_mask_by_time[i])
        #cached_subgraph.append(node_subgraph.to("cuda:0"))
        cached_subgraph.append(blocks)
 
    #in_feat = int(feat.shape[1])
    in_feat = 602
    model = WDGCN(in_feat=in_feat,
                           hidden_feat=args.n_hidden,
                           out_feat=7,
                           class_feat=128,
                           num_layers=args.n_layers
                           )
    if(rank == 0):
        a = 1
    model = model.to(local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    group1 = dist.new_group([0, 2])
    group2 = dist.new_group([1, 3])
    if(rank == 0 or rank == 2):
        group = group1
    elif(rank == 1 or rank == 3):
        group = group2

    model = DDP(model, device_ids=[local_rank], process_group=group, find_unused_parameters=True)

    train_max_index = 29
    time_window_size = args.n_hist_steps
    loss_class_weight = [float(w) for w in args.loss_class_weight.split(',')]
    loss_class_weight = torch.Tensor(loss_class_weight).to(local_rank)
    src_rank = None
    dst_rank = None
    if(rank == 0):
        src_rank = None
        dst_rank = 1
    elif(rank == 1):
        src_rank = 0
        dst_rank = None
    elif(rank == 2):
        src_rank = None
        dst_rank = 3
    elif(rank == 3):
        src_rank = 2
        dst_rank = None
    communication_fwd = comm.communication(src_rank, dst_rank, local_rank, fwd_shared_memory)

    if(rank == 0):
        src_rank = 1
        dst_rank = None
    elif(rank == 1):
        src_rank = None
        dst_rank = 0
    elif(rank == 2):
        src_rank = 3
        dst_rank = None
    elif(rank == 3):
        src_rank = None
        dst_rank = 2
    communication_bwd = comm.communication(src_rank, dst_rank, local_rank, bwd_shared_memory) 
    
    if(rank == 1):   
        a = 1
    start = end = 0
    time_list = []
    if(rank == 0):
        a = 1

    if(rank == 0 or rank == 2):
        step = 0
    elif(rank == 1 or rank == 3):
        step = 1
    local_world_size = int(world_size/2)
    comm_sum = []
    for epoch in range(args.num_epochs):
        model.train()
        predictions_list = []
        prev_list = []
        g_list = None
        feat_list = None
        
        for i in range(time_window_size, time_window_size + local_world_size - step - 1):
            if(rank == 0):
                a = 1
            predictions, prev, g_list, feat_list, fwd_time, start111, fwd_comm = run_forward(model, time_window_size, world_size, cached_subgraph, cached_feat, cached_label, i, rank, local_rank, step, communication_fwd, feat, g_list, feat_list, pattern_list)
            if(step == 1):
                predictions_list.append(predictions)
                prev_list.append(prev)
            else:
                predictions_list.append(predictions)
                prev_list.append(prev)

        if(step == 1):
            dist.barrier()
        dist.barrier()

        for i in range(time_window_size + local_world_size - step - 1, train_max_index + 1):
            torch.cuda.synchronize()
            start = time.perf_counter()
            if(rank == 1):
                 a = 1

            predictions, prev, g_list, feat_list, fwd_time, start111, fwd_comm = run_forward(model, time_window_size, world_size, cached_subgraph, cached_feat, cached_label, i, rank, local_rank, step, communication_fwd, feat, g_list, feat_list, pattern_list)

            if(step == 1):
                predictions_list.append(predictions)
                prev_list.append(prev)

            else:
                predictions_list.append(predictions)
                prev_list.append(prev)
                #predictions_list.append(predictions)

            if(step == 1):
                predictions_back = predictions_list.pop(0)
                prev_back = prev_list.pop(0)

                bwd_time, bwd_comm = run_backward(prev_back, predictions_back, world_size, rank, step, communication_bwd, group)

            else:
                #predictions_back = predictions_list.pop(0)
                #predictions_back = predictions_list.pop(0)
                prev_back = prev_list.pop(0)
                predictions_back = predictions_list.pop(0)
                #predictions_back = predictions_list.pop(0)
                bwd_time, bwd_comm = run_backward(prev_back, predictions_back, world_size, rank, step, communication_bwd, group)

            torch.cuda.synchronize()
            dist.barrier()
            end = time.perf_counter()
            if(step == 1 and i == train_max_index):
                continue

            if(fwd_time > 1.1):
                fwd_time = fwd_time - 1.1
            elif(fwd_time > 1):
                fwd_time = fwd_time - 1
            
            if(bwd_time > 1.1):
                bwd_time = bwd_time - 1.1
            elif(bwd_time > 1):
                bwd_time = bwd_time - 1

            time_list.append([fwd_time, bwd_time, fwd_time + bwd_time, end - start111, fwd_comm + bwd_comm])
            

        if(step == 0):
            dist.barrier()
            dist.barrier()

        for i in range(train_max_index + 1, train_max_index + 1 + local_world_size - step - 1):
            #predictions_back = predictions_list.pop(0)
            prev_back = prev_list.pop(0)
            predictions_back = predictions_list.pop(0)
            bwd_time, bwd_comm = run_backward(prev_back, predictions_back, world_size, rank, step, communication_bwd, group)
        
        #end training
        dist.barrier()
            #g_list = cached_subgraph[i - time_window_size:i + 1]
            
            # get predictions which has label
            #predictions = predictions[cached_labeled_node_mask[i]]
            #labels = cached_subgraph[i].ndata['label'][cached_labeled_node_mask[i]].long()
            #loss = F.cross_entropy(predictions, labels, weight=loss_class_weight)
            #optimizer.zero_grad()
            #loss.backward()
            #ptimizer.step()
    
    if(rank == 0):
        if(os.path.exists(folder_name) == False):
            os.mkdir(folder_name)
    dist.barrier()
    file_name = '{}/rank{}.csv'.format(folder_name, rank)
    df = pd.DataFrame(time_list)
    df.to_csv(file_name)


    
def main():
    argparser = argparse.ArgumentParser("EvolveGCN")
    argparser.add_argument('--model', type=str, default='EvolveGCN-O',
                           help='We can choose EvolveGCN-O or EvolveGCN-H,'
                                'but the EvolveGCN-H performance on Elliptic dataset is not good.')
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training.")
    argparser.add_argument('--num-epochs', type=int, default=3)
    argparser.add_argument('--n-hidden', type=int, default=256)
    argparser.add_argument('--n-layers', type=int, default=2)
    argparser.add_argument('--n-hist-steps', type=int, default=5,
                           help="If it is set to 5, it means in the first batch,"
                                "we use historical data of 0-4 to predict the data of time 5.")
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--loss-class-weight', type=str, default='0.35,0.65',
                           help='Weight for loss function. Follow the official code,'
                                'we need to change it to 0.25, 0.75 when use EvolveGCN-H')
    argparser.add_argument('--patience', type=int, default=100,
                           help="Patience for early stopping.")
    
    #argparser.add_argument('--pattern', type=str, default='[0,1,3,1,2,1]')
    #argparser.add_argument('--pattern', type=str, default='[0,1,0,1,2,1]')
    argparser.add_argument('--pattern', type=str, default='[0,0,0,0,0,0]')

    args = argparser.parse_args()
    pattern_list = args.pattern[1:-1]
    pattern_list = pattern_list.split(',')
    for num in range(0, len(pattern_list)):
        pattern_list[num] = int(pattern_list[num])
    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    args.device = device

    plist = []
    fwd_shared_memory_pool = []
    bwd_shared_memory_pool = []
    local_rank = [0, 0, 1, 1]
    for i in range(0, 2):
        device = 'cuda:{}'.format(i)
        fwd_shared_memory = torch.zeros([512, 256]).to(device).share_memory_()
        bwd_shared_memory = torch.zeros([512, 256]).to(device).share_memory_()
        fwd_shared_memory_pool.append(fwd_shared_memory)
        bwd_shared_memory_pool.append(bwd_shared_memory)
    proc_len = 4
    for i in range(0, proc_len):
        plist.append(mp.Process(target=train_model, args=(args, i, local_rank[i], proc_len, fwd_shared_memory_pool[local_rank[i]], bwd_shared_memory_pool[local_rank[i]], pattern_list)))
    
    for item in plist:
        item.start()
    
    for item in plist:
        item.join()

if(__name__ == "__main__"):
    main()
