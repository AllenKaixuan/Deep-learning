import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import  transforms 
import matplotlib.pyplot as plt
import numpy as np
import argparse




PATH = './model/model.pt'

loss_vector = []
epoch_vector = []
epochs = 20




if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # RGB three channels

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)  # need download,download=True
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True) 

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)  # multi process loading data

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)   # kernel_size=5, padding=2, stride=1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        
        
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def train():
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 600 == 599:    # print every 2000 mini-batches
            
            print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 600:.3f}')
            loss_vector.append("{:.3f}".format(running_loss / 600))
            running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(),PATH )



def test(net):
    
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))
    


def validate():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


def plt_para():
    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1,epochs+1), loss_vector)
    plt.title('validation loss')
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    net = Net()
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1, epochs+1):
        train()
    print(loss_vector)
    plt_para()

    # net.load_state_dict(torch.load(PATH))
    # test(net)
    # validate()

   


    

