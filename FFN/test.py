import torch
import torch.optim as optim

# 定义模型和损失函数
model = ...
criterion = ...

# 定义 Adam 和 SGD 优化器
adam_optimizer = optim.Adam(model.parameters(), lr=0.001)
sgd_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练过程
for epoch in range(num_epochs):
    # 使用 Adam 优化器进行前 n 次迭代
    if epoch > 5:
        optimizer = adam_optimizer
    # 切换到 SGD 优化器进行后续训练
    else:
        optimizer = sgd_optimizer
        # 对学习率进行衰减
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] *= 0.1
    
    # 训练一个 epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
