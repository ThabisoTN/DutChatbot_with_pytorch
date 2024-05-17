import torch
import torch.nn as nn

class NeuralNet(nn.modules):
    def __init__(self, input_size, hidden_size, num_classes):
        self.l1=nn.Linear(input_size, hidden_size)
        self.l2=nn.Linear(input_size, hidden_size)
        self.l3=nn.Linear(input_size, hidden_size)
        self.relu=nn.ReLU()

    def forward(self,x):
        out=self.l1(x)
        out=self.relu(out)
        out=self.l2(x)
        out=self.relu(out)
        out=self.l3(x)
        #mo activation and no softmax
        return out