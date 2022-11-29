# Import libraries
from __future__ import print_function, division
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
warnings.filterwarnings("ignore")
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
from typing import Iterator, Dict
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch.utils.data.sampler import SubsetRandomSampler

 
df=pd.read_csv('../DataCollection-REDfirst/OGP_dataset_collection_RED.csv', names=['image_name', 'x', 'y', 'z','w','X', 'Y', 'Z'], header=None)

#FINDING MAX AND MIN
max_x=df['x'].max()
min_x=df['x'].min()

max_y=df['y'].max()
min_y=df['y'].min()

max_z=df['z'].max()
min_z=df['z'].min()

max_w=df['w'].max()
min_w=df['w'].min()

max_X=df['X'].max()
min_X=df['X'].min()

max_Y=df['Y'].max()
min_Y=df['Y'].min()

max_Z=df['Z'].max()
min_Z=df['Z'].min()





class ClothDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations of OGP.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.OGP = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.OGP)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.OGP.iloc[idx, 0])
        image = io.imread(img_name)
        #print(self.OGP.iloc[idx, 0])
        image = image.astype('float32')/255.0
        image = np.array([image])
        OGP_pose = self.OGP.iloc[idx, 1:]
        OGP_pose = np.array([OGP_pose])
        #normalizing values between 0-1 using the max and min of each label column in csv
        #rather use a vector to do normalization, don't do one by one normalization here
        OGP_pose[0,0]=(OGP_pose[0,0]-min_x)/(max_x-min_x)
        OGP_pose[0,1]=(OGP_pose[0,1]-min_y)/(max_y-min_y)
        OGP_pose[0,2]=(OGP_pose[0,2]-min_z)/(max_z-min_z)
        OGP_pose[0,3]=(OGP_pose[0,3]-min_w)/(max_w-min_w)
        OGP_pose[0,4]=(OGP_pose[0,4]-min_X)/(max_X-min_X)
        OGP_pose[0,5]=(OGP_pose[0,5]-min_Y)/(max_Y-min_Y)
        OGP_pose[0,6]=(OGP_pose[0,6]-min_Z)/(max_Z-min_Z)
        OGP_pose = OGP_pose.astype('float').flatten()
        # print(OGP_pose)
        sample = {'image': image, 'OGP_pose': OGP_pose}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
 
      
cloth_dataset = ClothDataset(csv_file='../DataCollection-REDfirst/OGP_dataset_collection_RED.csv',
                                    root_dir='../DataCollection-REDfirst/')
                                    

validation_split = .2
shuffle_dataset = True
random_seed= 42   
                                
dataset_size = len(cloth_dataset)            
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))                      
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]          
#print( 'train_indices', train_indices)         
#print( 'val_indices', val_indices)  
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices) 
             
                                   
training_loader = torch.utils.data.DataLoader(cloth_dataset, batch_size=16,  num_workers=2, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(cloth_dataset, batch_size=16, num_workers=2, sampler=valid_sampler)                                   
                                   
                                                          
#without validation when only training was done                                    
#training_loader = torch.utils.data.DataLoader(cloth_dataset, batch_size=16, shuffle=True, num_workers=2)                                            
        



class GraspEstimationModel(nn.Module):


    def __init__(self):
        super(GraspEstimationModel, self).__init__()
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
        #this number 15488 is determined by running blender-3.2.1-linux-x64/test.py 
        self.fc1 = nn.Linear(in_features=138880, out_features=2048)
        self.relu6 = ReLU()
        self.fc2 = nn.Linear(in_features=2048, out_features=2048)
        self.relu7 = ReLU()
        #final regression linear layer
        self.fc3 = nn.Linear(in_features=2048, out_features=7)       



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

        return output

model = GraspEstimationModel()


loss_fn = torch.nn.MSELoss()
#loss_fn = torch.nn.SmoothL1Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train_one_epoch(epoch_index):
    running_loss = 0.000
    last_loss = 0.000
    running_tloss = 0.0
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        images=data['image']
        labels=data['OGP_pose'].float()
        #print('images ',images.shape)
        #print('labels ', labels)
        #print('labels.view ',labels.view(-1, 1).shape)
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        #print(labels)
        outputs = model(images)
        #print('....')
        #print('image 0', images[0])
        #print('image 1', images[1])
        #img0 = images[0]
        #img1 = images[1]
        #import matplotlib.pyplot as plt
  
        #from IPython import embed;embed()
        #plt.imshow(img0[0])
        #plt.imshow(img1[0])
        #plt.show()

        
        #print('....')
        #print(outputs)

        # Compute the loss and its gradients
        #print('outputs model(images) ',outputs)
        loss = loss_fn(outputs, labels)
        running_tloss += loss
        print(loss)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        
        # Gather data and report
        running_loss += loss.item()
        if i % 50 == 49:
            last_loss = running_loss / 50 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
        
    avg_loss = running_tloss / (i + 1)
            

    return avg_loss,last_loss




epoch_number = 0

EPOCHS = 10
#code is from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html 
# Make sure gradient tracking is on, and do a pass over the data

model.train()
#train_one_epoch(epoch_number)

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_tloss,last_tloss = train_one_epoch(epoch_number)

    # We don't need gradients on to do reporting
    model.train(False)   
    
    print('validation')
    #validation:
    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vimages=vdata['image']
        vlabels=vdata['OGP_pose'].float()
        voutputs = model(vimages)        
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss
        print (vloss)
    
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train: avg {} and last {}  valid: {}'.format(avg_tloss,last_tloss, avg_vloss))

    
    epoch_number += 1   
        
        
        
        
        
        
        
        
        
        
