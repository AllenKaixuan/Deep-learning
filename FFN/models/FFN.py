
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np



class FFN(nn.Module):      # inherit from nn.Module
    def __init__(self, dropp):
        super(FFN, self).__init__()
        self.dropp = dropp
        self.fc1 = nn.Linear(28*28, 256)  # weight: [28*28, 50]   bias: [50, ]
        self.fc1_drop = nn.Dropout(self.dropp)
        
        self.fc2 = nn.Linear(256, 196)
        self.fc2_drop = nn.Dropout(self.dropp)
        self.fc3 = nn.Linear(196, 128)
        self.fc3_drop = nn.Dropout(self.dropp)
        self.fc4 = nn.Linear(128, 10)
        
#         self.relu1 = nn.ReLU() 

       

    def forward(self, x):
        x = x.view(-1, 28*28)   # [32, 28*28]，对tensor进行reshape操作
        x = F.relu(self.fc1(x))  # use nn.functional to call the activation function, such as relu
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)   # [32, 10]
        x = F.relu(self.fc3(x))
        x = self.fc3_drop(x)
        return F.log_softmax(self.fc4(x), dim=1)


if __name__ == "__main__":
    Net = FFN()
    print(Net)