import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)

epochs = 10
batch_size = 128
train_dataset = datasets.MNIST('./data', 
                               train=False, 
                               download=False, 
                               transform=transforms.ToTensor())

validation_dataset = datasets.MNIST('./data', 
                                    train=False, 
                                    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=False)




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
    def __init__(self, in_channels, out_channels, hidden_size=2048, dropout=0.1):
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


def train(epoch, log_interval=200):
    # Set model to training mode
    model.train()
    
    if epoch < 8:
        optimizer = adam_optimizer
    # 切换到 SGD 优化器进行后续训练
    else:
        optimizer = sgd_optimizer
   
    

    
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()  
        
        # Update weights
        optimizer.step()    #  w - alpha * dL / dw
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            

def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))
  



model = ResMLP().to(device)
adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
sgd_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

print(model)

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)



# visialization
plt.figure(figsize=(5,3))
plt.plot(np.arange(1,epochs+1), lossv)
plt.title('validation loss')


plt.figure(figsize=(5,3))
plt.plot(np.arange(1,epochs+1), accv)
plt.title('validation accuracy')
plt.show()

