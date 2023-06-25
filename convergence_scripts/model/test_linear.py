import torch

import torch.nn as nn

import torch.optim as optim



class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(64, 8)
        self.relu2 = nn.ReLU()
    
    def forward(self, input):
        out = self.linear1(input)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        return out


model = MyModel().to("cuda:0")

input = torch.rand([10000, 64]).to("cuda:0").requires_grad_(True)
label = torch.ones([10000], dtype=torch.int64).to("cuda:0")

loss_func = nn.CrossEntropyLoss().to("cuda:0")

optimizer = optim.SGD(model.parameters(), lr=0.001)


for i in range(0, 1000):
    out = model(input)
    optimizer.zero_grad()
    loss = loss_func(out, label)
    loss.backward()
    optimizer.step()
