from torch.multiprocessing import set_start_method
set_start_method('spawn', force=True)

import argparse
import dgl
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

#from model.evolvegcn import EvolveGCN
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

def run_forward(model, time_window_size, world_size, cached_subgraph, cached_feat, cached_label, minibatch, rank, local_rank, step, communication, feat, g_list, feat_list, pattern_list, batch_len=1, batch_len_tmp=1):
    pipeline_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    # AS
    #pipeline_list = [0, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0]
    #pipeline_list = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    #pipeline_list = [0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
    rank_time_window_size = int((time_window_size+1)/(world_size/2))

    graph_index = [i for i in range(minibatch - time_window_size, minibatch + 1)]
    g_index = np.arange(0, time_window_size + 1)
    batch_id = int((minibatch - time_window_size)/batch_len)
    if(step == 0):
        g_exe = g_index[step * rank_time_window_size:(step + 1) * rank_time_window_size + pipeline_list[batch_id]]
        #
    else:
        g_exe = g_index[step * rank_time_window_size + pipeline_list[batch_id]:(step + 1) * rank_time_window_size]
    
    g_list = []
    feat_list = []
    pattern_list_exe = []
    if(rank == 0):
        a = 1
    for i in range(0, time_window_size + 1):
        if(i not in g_exe):
            g_list.append(None)
            feat_list.append(None)
            pattern_list_exe.append(None)
        else:
            batch_graph = []
            batch_block0 = []
            batch_block1 = []
            batch_feat = []
            batch_label = []
            index = minibatch - time_window_size + i
            batch_graph = cached_subgraph[index:index+batch_len]
            for item in batch_graph:
                batch_block0.append(item[0])
                batch_block1.append(item[1])
            batch_feat = cached_feat[index:index+batch_len]
            batch_label = cached_label[index:index+batch_len]
            batch_block0 = dgl.batch(batch_block0)
            batch_block1 = dgl.batch(batch_block1)
            batch_feat = torch.cat(batch_feat)
            batch_label = torch.cat(batch_label)
            tmp_block0 = []
            tmp_block1 = []
            tmp_feat = []
            tmp_label = []
            for j in range(0, batch_len_tmp):
                tmp_block0.append(batch_block0)
                tmp_block1.append(batch_block1)
                tmp_feat.append(batch_feat)
                tmp_label.append(batch_label)
            tmp_block0 = dgl.batch(tmp_block0)
            tmp_block1 = dgl.batch(tmp_block1)
            tmp_block0 = dgl.create_block(tmp_block0.edges(), num_src_nodes=tmp_block0.num_src_nodes(), num_dst_nodes=tmp_block0.num_dst_nodes()).to(local_rank)
            tmp_block1 = dgl.create_block(tmp_block1.edges(), num_src_nodes=tmp_block1.num_src_nodes(), num_dst_nodes=tmp_block1.num_dst_nodes()).to(local_rank)
            tmp_feat = torch.cat(tmp_feat).to(local_rank)
            tmp_label = torch.cat(tmp_label)
        
            g_list.append([tmp_block0, tmp_block1])
            feat_list.append(tmp_feat)
            pattern_list_exe.append(pattern_list[i])
        feat_list_exe = feat_list
    
    labels = tmp_label.long().to(local_rank)

    '''
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
    '''
    
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
    if(rank == 0):
        a = 1
    cached_subgraph = []
    cached_feat = []
    cached_label = []
    result = []
    folder_name = 'typo_data1/typo000000'
    #x = [i for i in range(39, -1, -1)]
    #x = [i for i in range(29, -1, -1)]
    x = [i for i in range(0, 17)]

    '''
    for num in range(13,18):
        cached_subgraph[num][0] = cached_subgraph[num][0].to(local_rank)
        cached_subgraph[num][1] = cached_subgraph[num][1].to(local_rank)

    g_list_gpu = [cached_subgraph[13], cached_subgraph[14], cached_subgraph[15], cached_subgraph[15], cached_subgraph[16], cached_subgraph[17]]
    feat_list_gpu = [cached_feat[13].to(local_rank), cached_feat[14].to(local_rank), cached_feat[15].to(local_rank), cached_feat[15].to(local_rank), cached_feat[16].to(local_rank), cached_feat[17].to(local_rank)]
    '''
    result = []

    graph_name = '../../../../dynamic_datasets/AS/snapshot{}.bin'.format(0)
    node_subgraph = load_graphs(graph_name)[0][0]

    feats = node_subgraph.ndata['feat']
    num_nodes = node_subgraph.num_nodes()
    label = torch.randint(0, 7, [num_nodes])

    # AS
    batch_len = 25

    cached_feat = []
    cached_labels = []

    batch_feats = []
    batch_label = []
    for i in range(0, 733):
        graph_name = '../../../../data/{}/sample/sample{}/snapshot{}.bin'.format('AS', 0, i)
        node_subgraph = load_graphs(graph_name)
        blocks = [None, None]
        #blocks[0] = dgl.create_block(node_subgraph[0][0].edges()).to("cuda:0")
        #blocks[1] = dgl.create_block(node_subgraph[0][1].edges()).to("cuda:0")
        blocks[0] = node_subgraph[0][0]
        blocks[1] = node_subgraph[0][1]
        cached_feat.append(node_subgraph[0][0].srcdata['feat'])
        cached_labels.append(torch.randint(0, 7, [node_subgraph[0][1].num_dst_nodes()]))
        #node_subgraph = dgl.node_subgraph(graph=g, nodes=node_mask_by_time[i])
        cached_subgraph.append(blocks)
 
    in_feat = int(feats.shape[1])
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

    # AS
    batch_len_tmp = 2

    group1 = dist.new_group([0, 2])
    group2 = dist.new_group([1, 3])
    if(rank == 0 or rank == 2):
        group = group1
    elif(rank == 1 or rank == 3):
        group = group2

    model = DDP(model, device_ids=[local_rank], process_group=group, find_unused_parameters=True)

    # AS
    train_max_index = 24

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
            minibatch = int((i - time_window_size) * batch_len + time_window_size)
            if(rank == 0):
                a = 1
            predictions, prev, g_list, feat_list, fwd_time, start111, fwd_comm = run_forward(model, time_window_size, world_size, cached_subgraph, cached_feat, cached_labels, minibatch, rank, local_rank, step, communication_fwd, feats, g_list, feat_list, pattern_list, batch_len=batch_len, batch_len_tmp=batch_len_tmp)
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
            minibatch = int((i - time_window_size) * batch_len + time_window_size)
            torch.cuda.synchronize()
            start = time.perf_counter()
            if(rank == 1):
                 a = 1

            predictions, prev, g_list, feat_list, fwd_time, start111, fwd_comm = run_forward(model, time_window_size, world_size, cached_subgraph, cached_feat, cached_labels, minibatch, rank, local_rank, step, communication_fwd, feats, g_list, feat_list, pattern_list, batch_len=batch_len, batch_len_tmp=batch_len_tmp)

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
    argparser.add_argument('--n-hidden', type=int, default=32)
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
    
    #argparser.add_argument('--pattern', type=str, default='[0,1,3,3,2,1]')
    #argparser.add_argument('--pattern', type=str, default='[0,1,0,1,2,1]')
    argparser.add_argument('--pattern', type=str, default='[0,0,0,0,0,0]')
    # AS
    #argparser.add_argument('--pattern', type=str, default='[0,1,2,0,0,1]')


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
        fwd_shared_memory = torch.zeros([50000, 256]).to(device).share_memory_()
        bwd_shared_memory = torch.zeros([50000, 256]).to(device).share_memory_()
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