# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:20:37 2020

@author: panpy
"""

import os
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from net import Net


EPOCH = 40
BATCH_SIZE = 128
LR = 0.005
DOWNLOAD_MNIST = True
N_TEST = 10
torch.manual_seed(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


train_data = torchvision.datasets.MNIST(root='./dataset/', train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=DOWNLOAD_MNIST,
                                        )
trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
Net = Net().to(device)
optimizer = torch.optim.Adam(Net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.2)
loss_func = nn.MSELoss()

f, a = plt.subplots(2, N_TEST, figsize=(10, 2))
plt.ion()
idx = []
for i in range(N_TEST):
    idx.append((train_data.targets == i).nonzero()[0].item())
view_data = train_data.data[idx]

for i in range(N_TEST):
    a[0][i].imshow(view_data[i], cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())
view_data = (view_data / 255.).to(device)

# 读一个checkpoint，如果存在的话
start_epoch = 0
avg_losses = []
if os.path.exists('checkpoint'):
    if os.path.exists('checkpoint/ckpt.pth'):
        print('Checkpoint exists, reading weights from checkpoint...')
        checkpoint = torch.load('checkpoint/ckpt.pth')
        Net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        avg_losses = checkpoint['avg_losses']
else:
    os.makedirs('checkpoint')

for epoch in range(start_epoch, EPOCH):
    avg_loss = 0
    flag = 0
    for step, (data, label) in enumerate(trainloader):
        Net.train()
        data = data.squeeze().to(device)
        encoded, decoded = Net(data)
        loss = loss_func(decoded, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            Net.eval()
            with torch.no_grad():
                print('Epoch: %d | train loss: %.4f | LR: %f' % (epoch, loss.item(), optimizer.param_groups[0]['lr']))
                encoded, decoded = Net(view_data)
                if decoded.device != 'cpu':
                    decoded = decoded.to('cpu')
                for i in range(N_TEST):
                    a[1][i].clear()
                    a[1][i].imshow(decoded[i], cmap='gray')
                    a[1][i].set_xticks(())
                    a[1][i].set_yticks(())
                plt.draw()
                plt.pause(0.05)
                flag += 1
                avg_loss += loss.item()

    scheduler.step()

    avg_loss = avg_loss / flag
    print('Epoch %d average loss: %.4f' % (epoch, avg_loss))
    avg_losses.append(avg_loss)
    x1 = range(0, len(avg_losses))
    y1 = avg_losses
    plt.figure(2)
    plt.plot(x1, y1, color='brown', linestyle='-')
    # for text_x, text_y in zip(x1, y1):  # epoch多了之后会挤成一团，实际用的时候可以不显示
    #     plt.text(text_x, text_y, '%.4f' % text_y, ha='center', va='bottom', fontsize=7)
    plt.xlabel('epoch')
    plt.ylabel('average loss')
    plt.draw()
    plt.pause(0.05)

    # 每个epoch结束后保存一个checkpoint
    torch.save({'epoch': epoch,
                'model_state_dict': Net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'avg_losses': avg_losses,
                }, 'checkpoint/ckpt.pth')

plt.ioff()
plt.show()

print(encoded[i])
if not (os.path.exists('model')):
    os.makedirs('model')
torch.save(Net.state_dict(), 'model/net.pth')
print('model saved!')
