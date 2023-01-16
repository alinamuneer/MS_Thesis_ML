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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime





################TRAINING CSV FILE###############
#in csv data is stored as quarternions x y z w then euler z y x and position x y z
df=pd.read_csv('../DataCollection-RED-Largedataset-Training/OGP_dataset_collection_RED.csv', names=['image_name', 'quarternion_x', 'quarternion_y', 'quarternion_z','quarternion_w','euler_z','euler_y','euler_x','position_X', 'position_Y', 'position_Z'], header=None)

#FINDING MAX AND MIN
max_euler_x=df['euler_x'].max()
min_euler_x=df['euler_x'].min()

max_euler_y=df['euler_y'].max()
min_euler_y=df['euler_y'].min()

max_position_X=df['position_X'].max()
min_position_X=df['position_X'].min()

max_position_Y=df['position_Y'].max()
min_position_Y=df['position_Y'].min()



##############Testing SIMULATION csv file#################
#in csv data is stored as quarternions x y z w then euler z y x and position x y z
test_df=pd.read_csv('../DataCollection-RED-Largedataset-Testing/OGP_dataset_collection_RED.csv', names=['image_name', 'quarternion_x', 'quarternion_y', 'quarternion_z','quarternion_w','euler_z','euler_y','euler_x','position_X', 'position_Y', 'position_Z'], header=None)

#FINDING MAX AND MIN
test_max_euler_x=test_df['euler_x'].max()
test_min_euler_x=test_df['euler_x'].min()

test_max_euler_y=test_df['euler_y'].max()
test_min_euler_y=test_df['euler_y'].min()

test_max_position_X=test_df['position_X'].max()
test_min_position_X=test_df['position_X'].min()

test_max_position_Y=test_df['position_Y'].max()
test_min_position_Y=test_df['position_Y'].min()




#############TESTING REAL CSV FILE#################

#in csv data is stored as quarternions x y z w then euler z y x and position x y z
Rtest_df=pd.read_csv('../DataCollection-Real-Pr2/ogp_real_world_Euler.csv', names=['image_name', 'euler_z','euler_y','euler_x','position_X', 'position_Y', 'position_Z'], header=None)

#FINDING MAX AND MIN
Rtest_max_euler_x=Rtest_df['euler_x'].max()
Rtest_min_euler_x=Rtest_df['euler_x'].min()

Rtest_max_euler_y=Rtest_df['euler_y'].max()
Rtest_min_euler_y=Rtest_df['euler_y'].min()

Rtest_max_position_X=Rtest_df['position_X'].max()
Rtest_min_position_X=Rtest_df['position_X'].min()

Rtest_max_position_Y=Rtest_df['position_Y'].max()
Rtest_min_position_Y=Rtest_df['position_Y'].min()















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
        OGP_pose = self.OGP.iloc[idx, 6:10]
        OGP_pose = np.array([OGP_pose])
        #print(OGP_pose)
        #normalizing values between 0-1 using the max and min of each label column in csv
        #rather use a vector to do normalization, don't do one by one normalization here

        OGP_pose[0,0]=(OGP_pose[0,0]-min_euler_y)/(max_euler_y-min_euler_y)
        OGP_pose[0,1]=(OGP_pose[0,1]-min_euler_x)/(max_euler_x-min_euler_x)

        #OGP_pose[0,0]=math.cos(OGP_pose[0,0])
        #OGP_pose[0,1]=math.cos(OGP_pose[0,1])
        OGP_pose[0,2]=(OGP_pose[0,2]-min_position_X)/(max_position_X-min_position_X)
        OGP_pose[0,3]=(OGP_pose[0,3]-min_position_Y)/(max_position_Y-min_position_Y)

        OGP_pose = OGP_pose.astype('float').flatten()



        sample = {'image': image, 'OGP_pose': OGP_pose}

        if self.transform:
            sample = self.transform(sample)

        return sample



class SimTestDataset(Dataset):

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
        OGP_pose = self.OGP.iloc[idx, 6:10]
        OGP_pose = np.array([OGP_pose])
        #print(OGP_pose)
        #normalizing values between 0-1 using the max and min of each label column in csv
        #rather use a vector to do normalization, don't do one by one normalization here
        #OGP_pose[0,0]=math.cos(OGP_pose[0,0])
        #OGP_pose[0,1]=math.cos(OGP_pose[0,1])
        OGP_pose[0,0]=(OGP_pose[0,0]-test_min_euler_y)/(test_max_euler_y-test_min_euler_y)
        OGP_pose[0,1]=(OGP_pose[0,1]-test_min_euler_x)/(test_max_euler_x-test_min_euler_x)
        OGP_pose[0,2]=(OGP_pose[0,2]-test_min_position_X)/(test_max_position_X-test_min_position_X)
        OGP_pose[0,3]=(OGP_pose[0,3]-test_min_position_Y)/(test_max_position_Y-test_min_position_Y)

        OGP_pose = OGP_pose.astype('float').flatten()
        #print(OGP_pose)


        sample = {'image': image, 'OGP_pose': OGP_pose}

        if self.transform:
            sample = self.transform(sample)

        return sample



 
 
class RealTestDataset(Dataset):

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
        img = io.imread(img_name)
        #print(self.OGP.iloc[idx, 0])  
        img[np.where(img > 920)] = 0
        img = img/3.6078
        img[np.where(img > 60)] = 255
        img[np.where(img <= 60)] = 0
        image = 1-img.astype('float32')/255.0
        image = np.array([image])
        #print(image)
        
        OGP_pose = self.OGP.iloc[idx, 2:6]
        OGP_pose = np.array([OGP_pose])
        #print(OGP_pose)
        #normalizing values between 0-1 using the max and min of each label column in csv
        #rather use a vector to do normalization, don't do one by one normalization here
        #OGP_pose[0,0]=math.cos(OGP_pose[0,0])
        #OGP_pose[0,1]=math.cos(OGP_pose[0,1])
        OGP_pose[0,0]=(OGP_pose[0,0]-Rtest_min_euler_y)/(Rtest_max_euler_y-Rtest_min_euler_y)
        OGP_pose[0,1]=(OGP_pose[0,1]-Rtest_min_euler_x)/(Rtest_max_euler_x-Rtest_min_euler_x)
        OGP_pose[0,2]=(OGP_pose[0,2]-Rtest_min_position_X)/(Rtest_max_position_X-Rtest_min_position_X)
        OGP_pose[0,3]=(OGP_pose[0,3]-Rtest_min_position_Y)/(Rtest_max_position_Y-Rtest_min_position_Y)

        OGP_pose = OGP_pose.astype('float').flatten()
        #print(OGP_pose)
        #print(image.shape)

        sample = {'image': image, 'OGP_pose': OGP_pose}

        if self.transform:
            sample = self.transform(sample)

        return sample













cloth_dataset = ClothDataset(csv_file='../DataCollection-RED-Largedataset-Training/OGP_dataset_collection_RED.csv',
                                    root_dir='../DataCollection-RED-Largedataset-Training/')


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



#############TRAINING SIMULATION DATASET###########
training_loader = torch.utils.data.DataLoader(cloth_dataset, batch_size=16,  num_workers=2, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(cloth_dataset, batch_size=16, num_workers=2, sampler=valid_sampler)



##########TESTING SIMULATION DATASET############
#test_sim_dataset = SimTestDataset(csv_file='../DataCollection-RED-Largedataset-Testing/OGP_dataset_collection_RED.csv',root_dir='../DataCollection-RED-Largedataset-Testing/')
#testing_loader = torch.utils.data.DataLoader(test_sim_dataset, batch_size=16,shuffle=False, num_workers=2)



##########TESTING REAL DATASET#################
test_real_dataset = RealTestDataset(csv_file='../DataCollection-Real-Pr2/ogp_real_world_Euler.csv',root_dir='../DataCollection-Real-Pr2/')
testing_loader = torch.utils.data.DataLoader(test_real_dataset, batch_size=16,shuffle=False, num_workers=2)









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
        #this number 138880 is determined by running blender-3.2.1-linux-x64/test.py
        self.fc1 = nn.Linear(in_features=138880, out_features=2048)
        self.relu6 = ReLU()
        self.fc2 = nn.Linear(in_features=2048, out_features=2048)
        self.relu7 = ReLU()
        #final regression linear layer
        self.fc3 = nn.Linear(in_features=2048, out_features=4)



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

if torch.cuda.is_available():
    model = model.cuda()







loss_fn = torch.nn.MSELoss()
#loss_fn = torch.nn.SmoothL1Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)




epoch_number = 0

EPOCHS = 20



timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('Final_graphs_full/OGP_trainer_0.001LR_RealTest{}_time_{}'.format(EPOCHS,timestamp))


#code is from https://www.geeksforgeeks.org/training-neural-networks-with-validation-using-pytorch/
min_valid_loss = np.inf

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)

    train_loss = 0.000

    for i, data in enumerate(training_loader):
        images=data['image']
        labels=data['OGP_pose'].float()
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        #print(labels)
        outputs = model(images)
        loss = loss_fn(outputs, labels)


        print(loss)

        loss.backward()
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        train_loss += loss.item()








    # We don't need gradients on to do reporting
    model.train(False)

    print('validation')
    #validation:
    valid_loss = 0.0
    model.eval()
    for i, vdata in enumerate(validation_loader):
        vimages=vdata['image']
        vlabels=vdata['OGP_pose'].float()
        if torch.cuda.is_available():
            vimages, vlabels = vimages.cuda(), vlabels.cuda()
        voutputs = model(vimages)
        vloss = loss_fn(voutputs, vlabels)
        valid_loss += vloss.item()

        #print(i)
        #print('vlabels: ',vlabels)
        #print('voutputs: ',voutputs)

        #print ('vloss: ',vloss)

    #print('Epoch',(e+1) \t\t Training Loss: {\train_loss / len(trainloader)} \t\t Validation Loss: {\valid_loss / len(validloader)}')
    #print('Loss train:{} and Loss valid: {}'.format(train_loss/len(training_loader),valid_loss/len(validation_loader))    )



    #writer.add_scalars('Training vs. Validation Loss', { 'Training' : train_loss/len(training_loader), 'Validation' : valid_loss/len(validation_loader) },epoch_number + 1)
    #writer.flush()


    ###########
    test__loss = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(testing_loader):
            test_images=data['image']
            test_labels=data['OGP_pose'].float()
            if torch.cuda.is_available():
                test_images, test_labels = test_images.cuda(), test_labels.cuda()
            # calculate outputs by running images through the network
            test_outputs = model(test_images)
            testloss = loss_fn(test_outputs, test_labels)
            test__loss += testloss.item()
    		#print(i)
    		#print('Test_outputs: ',test_outputs)
    		#print('test_labels: ',test_labels)
    		#print(testloss)


    print('Loss train:{} and Loss valid: {} Loss Test: {} '.format(train_loss/len(training_loader),valid_loss/len(validation_loader),test__loss/len(testing_loader))    )
    writer.add_scalars('Training vs. Validation vs Test Loss', { 'Training' : train_loss/len(training_loader), 'Validation' : valid_loss/len(validation_loader) ,'Testing' : test__loss/len(testing_loader)},epoch_number + 1)
    writer.flush()

    ##########



    if min_valid_loss > valid_loss:
        print('Validation Loss Decreased')
        min_valid_loss = valid_loss
        # Saving State Dict
        #torch.save(model.state_dict(), 'OGP_saved_model_Full_0.001LR.pth')
        torch.save(model.state_dict(), 'OGP_saved_model_Full_0.001LR_RealTest.pth')






    print('training length:{} and valid length: {} and Test length: {}'.format(len(training_loader),len(validation_loader),len(testing_loader))    )
    epoch_number += 1






#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-analysis-train-model
