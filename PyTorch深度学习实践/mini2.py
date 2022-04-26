#coding=utf-8
import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from d2l import torch as d2l
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy

##################数据处理#################
trans_train = transforms.Compose([transforms.ToTensor(),transforms.Resize(28),transforms.Normalize((0.1307,),(0.3081)),
                            transforms.RandomHorizontalFlip()])
trans_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])


##################加载数据集################
batch_size = 32
train_load = datasets.MNIST(r"D:\wenjian\py\pytorch-master\PyTorch深度学习实践",train=True, transform=trans_train, download=True)
test_load = datasets.MNIST(r"D:\wenjian\py\pytorch-master\PyTorch深度学习实践",train=False,transform=trans_test,download=True)
train_set = DataLoader(train_load,batch_size=batch_size,shuffle=True)
test_set = DataLoader(test_load,batch_size=batch_size,shuffle=False)


###############显示图片##################

fig = plt.figure()


def show_pc(imgs,hang,lie,title):
    fig,axes = plt.subplots(hang,lie)
    axes = axes.flatten()
    for i,img in enumerate(imgs):
        if torch.is_tensor(img):
            axes[i].imshow(img.numpy())
        else:
            axes[i].imshow(img)

        axes[i].set_xticks([])  # 将 x y 坐标轴的坐标去掉
        axes[i].set_yticks([])
        axes[i].set_title("{}".format(title[i]))
    plt.show()

a1,a1label = next(iter(DataLoader(train_load,batch_size=6,shuffle=True)))
print(a1.shape)
print(a1label)
show_pc(a1.reshape(6,28,28),2,3,a1label)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# print(device)
# 网络

net = d2l.resnet18(10,1)
net = net.to(device)
optimize = torch.optim.Adam(params=net.parameters(),lr=0.05,weight_decay=0.005)

def train(net,train_set,device,optimizer):

    # 一次一轮

    net.train()
    train_loss = 0
    train_acc = 0
    for train_idx,(train_batch,train_label) in enumerate(train_set):

        train_batch, train_label = train_batch.to(device),train_label.to(device)
        optimizer.zero_grad()
        output = net(train_batch)
        train_loss_batch = F.cross_entropy(output,train_label)
        train_loss_batch.backward()
        optimizer.step()

        gailv,label = output.max(dim=1)
        train_acc = (label == train_label).sum().item()
        train_batch_acc = train_acc/len(train_batch)
        train_acc += train_batch_acc

        train_loss += train_loss_batch.item() # 每一批的loss
    return train_loss/len(train_set),train_acc/len(train_set)    # 求平均，每一批的loss

def test(net,device,test_set):
    net.eval()

    test_loss = 0
    test_acc= 0
    with torch.no_grad():
        for test_idx,(test_batch,test_lable) in enumerate(test_set):
            test_batch, test_lable = test_batch.to(device),test_lable.to(device)
            output = net(test_batch)
            # 测试机不需要反向传播
            test_loss_batch = F.cross_entropy(output,test_lable)

            test_loss += test_loss_batch.item()

            # 准确率
            gailv,place = output.max(1)

            corr_sum = (place == test_lable).sum().item()   # 每一批预测准确的个数
            test_batch_acc = corr_sum/len(test_batch)   # 每一批的个数
            test_acc += test_batch_acc  # 将每一批的准确率加起来

        return test_loss/len(test_set),test_acc/len(test_set)   # 平均准确率

train_loss = []
train_acc = []
test_loss = []
test_acc = []


for epoch in range(10):
    train_loss_epoch,train_acc_epoch = train(net,train_set,device,optimizer=optimize)
    train_loss.append(train_loss_epoch)
    train_acc.append(train_acc_epoch)
    torch.save(net,'%s.pth'%epoch)

    print()

    test_loss_epoch,test_acc_epoch = test(net,device,test_set)
    test_loss.append(test_loss_epoch)
    test_acc.append(test_acc_epoch)

plt.subplot(121)
plt.plot(range(len(train_loss)),train_loss)
plt.plot(range(len(train_loss)),test_loss)
plt.ylabel("损失")

plt.subplot(122)
plt.plot(range(len(train_loss)),train_acc)
plt.plot(range(len(train_loss)),test_acc)









