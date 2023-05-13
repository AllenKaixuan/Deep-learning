import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import argparse
import logging

from models.FFN import FFN
from models.Res_mlp import ResMLP



#Use GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    



# Options
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--name', default = 'FNN', type=str, help='name')
parser.add_argument('--epochs', default = 10, type=int, help='epochs')
parser.add_argument('--batchsize', default = 128, type=int, help='batchsize')
parser.add_argument('--drop',default = 0.1,type=float, help='drop possibility')
opt = parser.parse_args()

epochs = opt.epochs
name = opt.name
batch_size = opt.batchsize
dropp = opt.drop



#log
logging.basicConfig(filename='./log/train.log', filemode='a', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger()

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

console_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


# Load data
train_dataset = datasets.MNIST('./data', 
                               train=True, 
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


# train
def train(epoch, log_interval=200):

    logging.info('\nbatchsize:{}, epoch:{}, drop:{}'.format(batch_size,epoch,model.dropp))
    # Set model to training mode
    model.train()
    if epoch/epochs < 0.8:
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
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            

# validate
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
    accuracy_vector.append(accuracy.item())
    
    
    logging.info('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))
   


# choose net
if name == "FNN":
    model = FFN(dropp).to(device)
    
else:
    model = ResMLP().to(device)

adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
sgd_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()


# train
logging.info(model) 
lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)


# save result
target_path = './result/result.csv'
df = pd.read_csv(target_path)
save_name = f'batchsize{batch_size}_drop{dropp}_accu'
df[save_name] = accv


df.to_csv(target_path,index=False)

logging.info("Finish training!")
