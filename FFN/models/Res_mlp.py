
import torch
import torch.nn as nn
import torch.nn.functional as F



class MLPBlock(nn.Module):
    """定义 MLPBlock 模块"""
    def __init__(self, in_channels, out_channels, hidden_size=2048):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_channels)

    def forward(self, x):
        out = self.fc1(x)
        out = F.gelu(out)
        out = self.fc2(out)
        return out

class ResMLPBlock(nn.Module):
    """定义 ResMLPBlock 模块"""
    def __init__(self, in_channels, out_channels, hidden_size=2048, dropout=0.2):
        super(ResMLPBlock, self).__init__()
        self.mlp1 = MLPBlock(in_channels, out_channels, hidden_size)
        self.mlp2 = MLPBlock(in_channels, out_channels, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out1 = self.mlp1(x)
        out2 = self.mlp2(self.dropout(out1))
        out = out1 + out2  # ResNet 风格的残差连接
        return out

class ResMLP(nn.Module):
    """定义 ResMLP 模型"""
    def __init__(self, input_size=28*28, num_classes=10, hidden_size=2048, depth=3, dropout=0.1):
        super(ResMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([ResMLPBlock(hidden_size, hidden_size, hidden_size, dropout) for _ in range(depth)])
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 28*28) 
        out = self.fc1(x)
        out = F.gelu(out)
        out = self.dropout(out)
        for block in self.blocks:
            out = block(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    Net = ResMLP()
    print(Net)