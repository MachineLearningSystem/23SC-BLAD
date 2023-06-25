import torch.distributed as dist
import torch
from time import time
class communication():
    def __init__(self, src_rank, dst_rank, local_rank, src_shared_memory, backend='gpu'):
        self.src_rank = src_rank
        self.dst_rank = dst_rank
        self.shared_memory = src_shared_memory
        self.local_rank = local_rank
        self.backend = backend
        self.dst_local_rank = src_shared_memory.device

        self.tensor_data = torch.zeros(2, dtype=torch.int32)
        self.req = None
        self.num_seq = 3

    def send(self, data, tag=100, end=False):
        if(self.backend == 'gpu'):
            if(self.req != None):
                self.req.wait()
            W1 = data[0]
            if(W1 == None):
                W1 = torch.randn([2,3]).to(self.local_rank)
            data_shape1 = torch.tensor(W1.shape, dtype=torch.int32)
            self.tensor_data = data_shape1
            start = time()
            for i in range(0, self.num_seq):
                self.shared_memory[0:data_shape1[0], 0:data_shape1[1]] = W1[0:data_shape1[0], 0:data_shape1[1]]
                torch.cuda.synchronize()
            end1 = time()
            duration = end1 - start
            if(end == False):
                self.req = dist.isend(self.tensor_data, dst=self.dst_rank, tag=tag)
            if(end == True):
                dist.send(self.tensor_data, dst=self.dst_rank, tag=tag)
        elif(self.backend == 'cpu'):
            if(self.req != None):
                self.req.wait()
            W1 = data[0]
            if(W1 == None):
                W1 = torch.randn([2,3]).to(self.local_rank)
            data_shape1 = torch.tensor(W1.shape, dtype=torch.int32)
            self.tensor_data = data_shape1
            start = time()
            for i in range(0, self.num_seq):
                W1 = W1.to(self.dst_local_rank, non_blocking=True)
                self.shared_memory[0:data_shape1[0], 0:data_shape1[1]] = W1[0:data_shape1[0], 0:data_shape1[1]]
                torch.cuda.synchronize()
            end1 = time()
            duration = end1 - start
            if(end == False):
                self.req = dist.isend(self.tensor_data, dst=self.dst_rank, tag=tag)
            if(end == True):
                dist.send(self.tensor_data, dst=self.dst_rank, tag=tag)
        
        return duration

    def recv(self, tag=100):
        if(self.backend == 'gpu'):
            dist.recv(self.tensor_data, src=self.src_rank, tag=tag)
            start = time()
            for i in range(0, self.num_seq):
                W1 = self.shared_memory[0:self.tensor_data[0], 0:self.tensor_data[1]]
                torch.cuda.synchronize()
            end = time()
            duration = end - start
            W1 = W1.requires_grad_()
            data = [W1]
        elif(self.backend == 'cpu'):
            dist.recv(self.tensor_data, src=self.src_rank, tag=tag)
            start = time()
            for i in range(0, self.num_seq):
                W1 = self.shared_memory[0:self.tensor_data[0], 0:self.tensor_data[1]]
                W2 = W1.to(self.local_rank, non_blocking=False)
                torch.cuda.synchronize()
            end = time()
            duration = end - start
            W1 = W2.detach().requires_grad_()
            data = [W1]
        
        return data, duration
