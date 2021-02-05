#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
from PIL import Image
import math
import time
import os
import numpy as np
import natsort
import csv


# # Network Setting

# In[2]:


epochSize = 100
batchSize = 120
testSize = 300
valSize = 120
learningRate = 4e-4
momentum = 0.5
printInterval = 6
outSize = 10

torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
torch.cuda.set_device(0)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))


# # Data Augumentation

# In[3]:


import numbers
inputSize = 224
    
trainAugumentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(20,translate=(0.1,0.1),scale=(0.8,1)),    
    transforms.ToTensor(),
    transforms.Resize((inputSize,inputSize))
])

testAugumentation = transforms.Compose([    
    transforms.ToTensor(),
    transforms.Resize((inputSize,inputSize))
])


# # Data Sets

# In[4]:


dataDir = "C://Users//jason//OneDrive//圖片//Machine Learning Data//CIFAR-10-images-master//"
trainDir = os.path.join(dataDir,'train//')
valDir = os.path.join(dataDir,'val//')
testDir = os.path.join(dataDir,'test//')

trainData = datasets.ImageFolder(trainDir,transform = trainAugumentation)
valData = datasets.ImageFolder(valDir,transform = trainAugumentation)
testData = datasets.ImageFolder(testDir,transform = testAugumentation)


# # Data Loader

# In[5]:


trainLoader = torch.utils.data.DataLoader(trainData,batch_size = batchSize,shuffle = True)
valLoader = torch.utils.data.DataLoader(valData,batch_size = valSize,shuffle = True)
testLoader = torch.utils.data.DataLoader(testData,batch_size = testSize,shuffle = False)

trainDataSize = len(trainLoader.dataset)
valDataSize = len(valLoader.dataset)
testDataSize = len(testLoader.dataset)


# # Check Training Data

# In[6]:


examples = enumerate(testLoader)
batch_idx, (example_data, example_targets) = next(examples)
example_data.shape

fig = plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    #plt.tight_layout()
    plt.imshow(np.transpose(example_data[i], (1, 2, 0)))
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])


# # ResNet 18 Model

# In[7]:


class ResBlock(nn.Module):
    def __init__(self,inChannel,outChannel,isFirstBlock):
        super(ResBlock,self).__init__()
        
        if isFirstBlock:
            strideNum = 1
        else:
            strideNum = 2
        
        self.conv1 = nn.Conv2d(inChannel,outChannel,kernel_size=3,padding=1,stride=strideNum)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.conv2 = nn.Conv2d(outChannel,outChannel,kernel_size=3,padding=1,stride=1)
        self.bn2 = nn.BatchNorm2d(outChannel)
        
        self.shortcut = nn.Sequential()
        if inChannel != outChannel:
            self.shortcut.add_module('conv',nn.Conv2d(inChannel,outChannel,kernel_size=1,stride=strideNum))
            self.shortcut.add_module('bn',nn.BatchNorm2d(outChannel)) 
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(x)
        
        out = F.relu(out)
        
        return out

    
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,padding=3,stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3,stride=2,padding=1)
        
        self.res1 = self._make_resBlock(64,64,'res1',True)
        self.res2 = self._make_resBlock(64,128,'res2',False)
        self.res3 = self._make_resBlock(128,256,'res3',False)
        self.res4 = self._make_resBlock(256,512,'res4',False)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)
        
    def _make_resBlock(self,inChannel,outChannel,block_name,isFirstBlock):
        resBlock = nn.Sequential()
        resBlock.add_module(block_name, ResBlock(inChannel, outChannel,isFirstBlock))
        
        return resBlock
    
    
    def forward(self,x):
        
        out = self.conv1(x)
        out = F.relu(out)
        out = self.bn(out)
        out = self.pool(out)
        
        out = self.res1(out)        
        out = self.res2(out)        
        out = self.res3(out)        
        out = self.res4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.log_softmax(out)
        
        return out


# # Create Network

# In[8]:


network = ResNet().cuda()
#network.load_state_dict(torch.load("C:\\Users\\jason\\OneDrive\\文件\\Python NN\\Pytorch Pet\\weights161120.pth"))

optimizer = optim.Adam(network.parameters(), weight_decay=0.01,lr=learningRate)
scheduler = ReduceLROnPlateau(optimizer,verbose=True,patience=8,min_lr=1e-6)

print('Memory Usage:')
print('Allocated:', torch.cuda.memory_allocated(0)/1024/1024, 'MB')
print('Cached:   ', torch.cuda.memory_allocated(0)/1024/1024, 'MB')

pytorch_total_params = sum(p.numel() for p in network.parameters())
print("\nTotal Parameters:",pytorch_total_params)


# # Learning Rate Finder

# def findLearningRate():
#     lr = 1e-7
#     
#         
#     lossArr = []
#     dLossArr = []
#     d2LossArr = []
#     lrArr = []
#     
#     for idx,(data,label) in enumerate(trainLoader):
#         
#         
#         # Put data to GPU
#         data = data.cuda()
#         label = label.cuda()
#         
#         # Computations
#         optimizer.zero_grad()
#         output = network(data)
#         loss = F.nll_loss(output,label)
#         loss.backward()
#         optimizer.step()
#         
#         print(lr,loss.item())
#         
#         lossArr.append(loss.item())
#         lrArr.append(lr)
#         lr = lr*1.1
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#     
#         if (lr > 0.1):
#             break
#     
#     return lossArr,lrArr
# 
# lossArr,lrArr = findLearningRate()
# plt.figure(figsize=(15, 9))
# plt.plot(lrArr,lossArr)
# plt.ylabel('some numbers')
# plt.xscale('log')
# plt.show()
# 
# def weight_reset(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         m.reset_parameters()
#         
# network.apply(weight_reset)
# 
# for param_group in optimizer.param_groups:
#             param_group['lr'] = 1e-4

# # Result Array

# In[9]:


trainLossArr = []
trainCounterArr = []
valLossArr = []
valCounterArr = []
testLossArr = []
testCounterArr = [(i+1)*len(trainLoader.dataset) for i in range(epochSize)]


# # Training Function

# In[9]:


def train(epoch):
    network.train()
    epoch_loss = 0
    epoch_accuracy = 0
    
    currentImg = 0
    totalImg = len(trainLoader.dataset)
    
    for idx,(data,label) in enumerate(trainLoader):
        # Put data to GPU
        data = data.cuda()
        label = label.cuda()
        
        # Computations
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output,label)
        loss.backward()
        optimizer.step()
        
        # Calculate acc and loss
        batch_accuracy = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += (output.argmax(dim=1) == label).float().sum()
        epoch_loss += loss/len(trainLoader)
        
        # Print and record every n interval
        if idx % printInterval == 0:
            
            # Print
            currentImg = idx * len(data)
            epochProgress = currentImg/totalImg
            
            val_accuracy,val_loss = val()
            
            if idx == 0:
                print('Train Epoch: {} [{}/{} (0%)]'.format(epoch,idx, len(trainLoader.dataset)))
            else:                
                print('Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.4f}  Accuracy: {:.4f}  Val_Accuracy: {:.4f}'.format(
                    epoch,currentImg,totalImg,epochProgress*100 ,loss.item(),batch_accuracy,val_accuracy))
            
            # Record
            trainLossArr.append(loss.item())
            trainCounterArr.append(
            (batch_idx*batchSize) + ((epoch-1)*len(trainLoader.dataset)))
            
            valLossArr.append(val_loss)
            valCounterArr.append(
            (batch_idx*batchSize) + ((epoch-1)*len(trainLoader.dataset)))
            
        
    
    # Final print
    print('Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.4f}  Accuracy: {:.4f}  Val_Accuracy: {:.4f}'.format(
                    epoch,totalImg,totalImg,100 ,loss.item(),batch_accuracy,val_accuracy))
    print("Train  Accuracy: {:.4f}  Loss: {:.4f}".format(epoch_accuracy/len(trainLoader.dataset),loss.item()))


# # Testing Function

# In[10]:


def test():
    network.eval()
    test_loss = 0
    test_accuracy = 0
    idx = 0
    with torch.no_grad():
        for data,label in testLoader:
            # Data to GPU and compute
            data = data.to(device)
            label = label.to(device)
            output = network(data)
            test_loss += F.nll_loss(output,label).item()
            #test_accuracy += ((output.argmax(dim=1) == label).float().mean())/len(testLoader)
            test_accuracy += (output.argmax(dim=1) == label).float().sum()
    
    testLossArr.append(test_loss)
    test_accuracy = test_accuracy/len(testLoader.dataset)
    print("Test  Accuracy: {:.4f}  Loss: {:.4f}".format(test_accuracy,test_loss))
    
    return test_loss,test_accuracy


# # Validation Function

# In[11]:


def val():
    #network.eval()
    val_loss = 0
    val_accuracy = 0
    idx = 0
    with torch.no_grad():
        for data,label in valLoader:
            data = data.to(device)
            label = label.to(device)
            output = network(data)
            val_loss += F.nll_loss(output,label).item()
            val_accuracy += ((output.argmax(dim=1) == label).float().mean())/len(valLoader)
            
    network.train()
    
    return val_accuracy,val_loss


# # Change Weight Decay

# In[13]:


def setWeightDecay(acc,wDStage):
    
    acc = acc*100
    if acc> 85 and wDecayStage == 4:
        wDStage = 5
        for param_gp in optimizer.param_groups:
            param_gp['weight_decay'] = 0
            param_gp['lr'] = 2.5e-5
        print('Reduce Weight Decay to ',param_gp['weight_decay'])
    elif acc>80 and wDecayStage == 3:
        wDStage = 4
        for param_gp in optimizer.param_groups:
            param_gp['weight_decay'] = 0.000001
            param_gp['lr'] = 5e-5
        print('Reduce Weight Decay to ',param_gp['weight_decay'])
    elif acc>75 and wDecayStage == 2:
        wDStage = 3
        for param_gp in optimizer.param_groups:
            param_gp['weight_decay'] = 0.00001
            param_gp['lr'] = 1e-4
        print('Reduce Weight Decay to ',param_gp['weight_decay'])
    elif acc>65 and wDecayStage == 1:
        wDStage = 2
        for param_gp in optimizer.param_groups:
            param_gp['weight_decay'] = 0.0001
        print('Reduce Weight Decay to ',param_gp['weight_decay'])
    elif acc>50 and wDecayStage == 0:
        wDStage = 1
        for param_gp in optimizer.param_groups:
            param_gp['weight_decay'] = 0.001
        print('Reduce Weight Decay to ',param_gp['weight_decay'])
    
    return wDStage


# # Start training

# In[14]:


wDecayStage = 0
maxAcc = 0
for epoch in range(1,epochSize+1):
        
    startEpochTime = time.time()
    train(epoch)
    
    loss,acc = test()
    scheduler.step(loss)
    wDecayStage = setWeightDecay(acc,wDecayStage)
    
    
    if acc > maxAcc:
        maxAcc = acc
        torch.save(network.state_dict(), "C:\\Users\\jason\\OneDrive\\文件\\Python NN\\Pytorch ResNet\\weight211120.pth")
        print("Model Saved")
    
    print("Time: {:.4f}".format(time.time()-startEpochTime),end='\n\n')


# # Plot Training Record

# In[ ]:


bufferSize = 10

trainCounterArr = []
for i in range(len(trainLossArr)-bufferSize):
    trainCounterArr.append(i)

trainLossArr_ = []
valLossArr_ = []

for i in range(len(trainLossArr)-bufferSize):
    trainLossArr_.append(np.average(trainLossArr[i:i+bufferSize]))
    valLossArr_.append(np.average(valLossArr[i:i+bufferSize]))

fig = plt.figure()
plt.plot(trainCounterArr, trainLossArr_, color='blue')
plt.plot(trainCounterArr, valLossArr_, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')


# # Load Saved Weights

# In[12]:


network.load_state_dict(torch.load("C:\\Users\\jason\\OneDrive\\文件\\Python NN\\Pytorch ResNet\\weight241120_85.pth"))


# # Final Test

# In[13]:


testData = datasets.ImageFolder(testDir,transform = testAugumentation)
testLoader = torch.utils.data.DataLoader(testData,batch_size = 50,shuffle = True)

#loss,acc = test()

examples = enumerate(testLoader)
batch_idx, (example_data, example_targets) = next(examples)
output = network(example_data.cuda())
print(output)
output = output.argmax(dim=1)

fig = plt.figure(figsize=(18, 18))
label = "Airplanes", "Cars", "Birds", "Cats", "Deer", "Dogs", "Frogs", "Horses", "Ships", "Trucks"
for i in range(49):
    plt.subplot(7,7,i+1)
    #plt.tight_layout()
    plt.imshow(np.transpose(example_data[i], (1, 2, 0)))
    title = label[output[i]]+"   "+str((output[i]==example_targets[i]).item())
    plt.title(title)#"Ground Truth: {}".format(example_targets[i])
    plt.xticks([])
    plt.yticks([])


# # Load Result

# In[ ]:


testDataSize = 10000
testSize = 50

output_epoch = np.zeros([testDataSize,outSize])
ans = np.zeros([testDataSize])
out = np.zeros([testDataSize,2])

for i, (x, y) in enumerate(testLoader):
    x = x.cuda()
    output_epoch[i*testSize:(i+1)*testSize,:] = network(x).cpu().detach().numpy()
    ans[i*testSize:(i+1)*testSize] = y.cpu()
    
    output_batch = F.softmax(network(x), dim=1)
    for j in range(np.shape(output_batch)[0]):
        out[i*testSize+j,0] = output_batch[j].argmax()
        out[i*testSize+j,1] = output_batch[j].max().detach().cpu().numpy()

print(out)
# out: [label,prop]


# 

# # Find Best 5 and Worst 5

# In[ ]:


class_max = np.zeros([outSize,10])
# class_max: [1st pos_dataset,...,5th pos_dataset , 1st val,...,5th val]

for i in range(testDataSize):
    idx = out[i,0]
    val = out[i,1]
    if val > class_max[int(idx),5]:
        class_max[int(idx),6:9] = class_max[int(idx),5:8]
        class_max[int(idx),5] = val
        class_max[int(idx),1:4] = class_max[int(idx),0:3]
        class_max[int(idx),0] = i
    elif val > class_max[int(idx),6]:
        class_max[int(idx),7:9] = class_max[int(idx),6:8]
        class_max[int(idx),6] = val
        class_max[int(idx),2:4] = class_max[int(idx),1:3]
        class_max[int(idx),1] = i
    elif val > class_max[int(idx),7]:
        class_max[int(idx),8:9] = class_max[int(idx),7:8]
        class_max[int(idx),7] = val
        class_max[int(idx),3:4] = class_max[int(idx),2:3]
        class_max[int(idx),2] = i
    elif val > class_max[int(idx),8]:
        class_max[int(idx),9] = class_max[int(idx),8]
        class_max[int(idx),8] = val
        class_max[int(idx),4] = class_max[int(idx),3]
        class_max[int(idx),3] = i
    elif val > class_max[int(idx),9]:
        class_max[int(idx),9] = val
        class_max[int(idx),4] = i
        
class_min = np.zeros([outSize,10])+1000

for i in range(testDataSize):
    idx = out[i,0]
    val = out[i,1]
    if val < class_min[int(idx),5]:
        class_min[int(idx),6:9] = class_min[int(idx),5:8]
        class_min[int(idx),5] = val
        class_min[int(idx),1:4] = class_min[int(idx),0:3]
        class_min[int(idx),0] = i
    elif val < class_min[int(idx),6]:
        class_min[int(idx),7:9] = class_min[int(idx),6:8]
        class_min[int(idx),6] = val
        class_min[int(idx),2:4] = class_min[int(idx),1:3]
        class_min[int(idx),1] = i
    elif val < class_min[int(idx),7]:
        class_min[int(idx),8:9] = class_min[int(idx),7:8]
        class_min[int(idx),7] = val
        class_min[int(idx),3:4] = class_min[int(idx),2:3]
        class_min[int(idx),2] = i
    elif val < class_min[int(idx),8]:
        class_min[int(idx),9] = class_min[int(idx),8]
        class_min[int(idx),8] = val
        class_min[int(idx),4] = class_min[int(idx),3]
        class_min[int(idx),3] = i
    elif val < class_min[int(idx),9]:
        class_min[int(idx),9] = val
        class_min[int(idx),4] = i


# In[ ]:


print('MAX')
for i in range(outSize):
    for j in range(5):
        img = testData.__getitem__(int(class_max[i,j]))[0]
        img = np.transpose(img, (1, 2, 0))
        plt.subplot(1,5,j+1)
        plt.imshow(img)
        plt.title("Idx:{}".format(int(class_max[i,j])+11500))
    plt.show()

print('MIN')
for i in range(outSize):
    for j in range(5):
        img = testData.__getitem__(int(class_min[i,j]))[0]
        img = np.transpose(img, (1, 2, 0))
        plt.subplot(1,5,j+1)
        plt.imshow(img)
        plt.title("Idx:{}".format(int(class_min[i,j])+11500))
    plt.show()


# In[ ]:


ConfusionMatrix = np.zeros([outSize,outSize])

for i in range(testDataSize):
    ConfusionMatrix[int(ans[i]),int(out[i,0])] = ConfusionMatrix[int(ans[i]),int(out[i,0])]+1
    
print(ConfusionMatrix)

positive = np.zeros([outSize])
negative = np.zeros([outSize])

for i in range(outSize):
    positive[i] = ConfusionMatrix[i,i]/np.sum(ConfusionMatrix[:,i])
    negative[i] = (np.sum(ConfusionMatrix[:i,:i])+np.sum(ConfusionMatrix[(i+1):,:i])+np.sum(ConfusionMatrix[:i,(i+1):])+np.sum(ConfusionMatrix[(i+1):,(i+1):]))/(np.sum(ConfusionMatrix[:,:i])+np.sum(ConfusionMatrix[:,(i+1):]))

print("\n Airplane   Automobile Bird       Cat        Deer       Dog\n Frog       Horse      Ship       Truck")
print(positive)
print(negative)


# In[ ]:


print('Memory Usage:')
print('Allocated:', torch.cuda.memory_allocated(0)/1024/1204, 'MB')
print('Cached:   ', torch.cuda.memory_allocated(0)/1024/1024, 'MB')


# In[ ]:




