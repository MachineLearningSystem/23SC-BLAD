import torch.distributed as dist
import torch
import time
class communication():
    def __init__(self, src_rank, dst_rank, local_rank, src_shared_memory, backend='gpu'):
        self.src_rank = src_rank
        self.dst_rank = dst_rank
        self.shared_memory = src_shared_memory
        self.local_rank = local_rank
        self.backend = backend
        self.dst_local_rank = src_shared_memory.device

        self.tensor_data = torch.zeros(4, dtype=torch.int32)
        self.req = None
        self.num_seq = 3

    def send(self, data, tag=100, end=False):
        if(self.backend == 'gpu'):
            if(self.req != None):
                self.req.wait()
            W1 = data[0]
            W2 = data[1]
            data_shape1 = torch.tensor(W1.shape, dtype=torch.int32)
            data_shape2 = torch.tensor(W2.shape, dtype=torch.int32)
            self.tensor_data = torch.cat([data_shape1, data_shape2], 0)
            start = time.time()
            self.shared_memory[0, 0:self.tensor_data[0], 0:self.tensor_data[1]] = W1[0:self.tensor_data[0], 0:self.tensor_data[1]]
            self.shared_memory[1, 0:self.tensor_data[2], 0:self.tensor_data[3]] = W2[0:self.tensor_data[2], 0:self.tensor_data[3]]
            torch.cuda.synchronize()
            end1 = time.time()
            duration = end1 - start
            if(end == False):
                self.req = dist.isend(self.tensor_data, dst=self.dst_rank, tag=tag)
            if(end == True):
                dist.send(self.tensor_data, dst=self.dst_rank, tag=tag)
        elif(self.backend == 'cpu'):
            if(self.req != None):
                self.req.wait()
            W1 = data[0]
            W2 = data[1]
            data_shape1 = torch.tensor(W1.shape, dtype=torch.int32)
            data_shape2 = torch.tensor(W2.shape, dtype=torch.int32)
            self.tensor_data = torch.cat([data_shape1, data_shape2], 0)
            start = time.time()
            for i in range(0, self.num_seq):
                W1 = W1.to(self.dst_local_rank, non_blocking=False)
                torch.cuda.synchronize()
                W2 = W2.to(self.dst_local_rank, non_blocking=False)
                torch.cuda.synchronize()
                self.shared_memory[0, 0:self.tensor_data[0], 0:self.tensor_data[1]] = W1[0:self.tensor_data[0], 0:self.tensor_data[1]]
                self.shared_memory[1, 0:self.tensor_data[2], 0:self.tensor_data[3]] = W2[0:self.tensor_data[2], 0:self.tensor_data[3]]
                torch.cuda.synchronize()
            end1 = time.time()
            duration = end1 - start
            if(end == False):
                self.req = dist.isend(self.tensor_data, dst=self.dst_rank, tag=tag)
            if(end == True):
                dist.send(self.tensor_data, dst=self.dst_rank, tag=tag)
        
        return duration

    def recv(self, tag=100):
        if(self.backend == 'gpu'):
            dist.recv(self.tensor_data, src=self.src_rank, tag=tag)
            start = time.time()
            W1 = self.shared_memory[0, 0:self.tensor_data[0], 0:self.tensor_data[1]]
            W2 = self.shared_memory[1, 0:self.tensor_data[2], 0:self.tensor_data[3]]
            torch.cuda.synchronize()
            end1 = time.time()
            duration = end1 - start
            W1 = W1.requires_grad_()
            W2 = W2.requires_grad_()
            data = [W1, W2]
        elif(self.backend == 'cpu'):
            dist.recv(self.tensor_data, src=self.src_rank, tag=tag)
            start = time.time()
            W1 = self.shared_memory[0, 0:self.tensor_data[0], 0:self.tensor_data[1]]
            W2 = self.shared_memory[1, 0:self.tensor_data[2], 0:self.tensor_data[3]]
            for i in range(0, self.num_seq):
                W3 = W1.to(self.local_rank, non_blocking = False)
                torch.cuda.synchronize()
                W4 = W2.to(self.local_rank, non_blocking = False)
                torch.cuda.synchronize()
            end = time.time()
            duration = end - start
            W1 = W3.requires_grad_()
            W2 = W4.requires_grad_()
            data = [W1, W2]
        
        return data, duration
