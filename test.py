import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten



class Model(nn.Module):
 def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(11, 11), stride=(2, 2), bias=False)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), bias=False)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), bias=False)
        self.relu3 = ReLU()
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), bias=False)
        self.relu4 = ReLU()
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), bias=False)
        self.relu5 = ReLU()
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=15488, out_features=2048)
        self.relu6 = ReLU()
        self.fc2 = nn.Linear(in_features=2048, out_features=2048)
        self.relu7 = ReLU()
        #final regression linear layer
        self.fc3 = nn.Linear(in_features=2048, out_features=2)



 def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        x = self.relu7(x)
        #final layer is regression linear, binary limits 0->1
        output = self.fc3(x)
        print(output)

        return output


model = Model()
x = torch.randn(1, 1, 256, 256)
out = model(x)
