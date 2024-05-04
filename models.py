# 实现两个网络，Lenet和VGG，后者比前者复杂
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self,input_channels = 1,num_classes = 10):
        super(LeNet,self).__init__()
        self.conv_one = nn.Conv2d(input_channels,6,5,padding=2)
        self.conv_two = nn.Conv2d(6,16,5)
        self.fc_one = nn.Linear(16*6*6,120)
        self.fc_two = nn.Linear(120,84)
        self.fc_three = nn.Linear(84,num_classes)
        self.agg_data = 0 # how many data samples have been aggregated in the global model
    def forward(self,x):
        out = F.max_pool2d(F.relu(self.conv_one(x)),(2,2))
        out = F.max_pool2d(F.relu(self.conv_two(out)),(2,2))
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc_one(out))
        out = F.relu(self.fc_two(out))
        out = self.fc_three(out)
        return out

class VGG9(nn.Module):
    def __init__(self,in_channles=1,num_cls=10):
        super(VGG9,self).__init__()
        self.agg_data = 0
        self.conv_one = nn.Conv2d(in_channles,32,3,padding = 1)
        self.conv_two = nn.Conv2d(32,64,3,padding = 1)
        self.conv_three = nn.Conv2d(64,128,3,padding = 1)
        self.conv_four = nn.Conv2d(128,128,3,padding = 1)
        self.conv_five = nn.Conv2d(128,256,3,padding = 1)
        self.conv_six = nn.Conv2d(256,256,3,padding = 1)
        self.fc_one = nn.Linear(4*4*256,512)
        self.fc_two = nn.Linear(512,512)
        self.fc_three = nn.Linear(512,num_cls)
        
    def forward(self,x):
        out = F.relu(self.conv_one(x))
        out = F.relu(self.conv_two(out))
        out = F.max_pool2d(out,2,2)
        out = F.relu(self.conv_three(out))
        out = F.relu(self.conv_four(out))
        out = F.max_pool2d(out,2,2)
        out = F.relu(self.conv_five(out))
        out = F.relu(self.conv_six(out))
        out = F.max_pool2d(out,2,2)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc_one(out))
        out = F.relu(self.fc_two(out))
        out = self.fc_three(out)
        return out