# Import libraries
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.preprocessing import MinMaxScaler
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import pandas as pd


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
        self.OGP = pd.read_csv(csv_file,encoding='utf-8')
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
        #normalizing image by 255
        image = image.astype('float32')/255.0
        image = np.array([image])
        OGP_pose = self.OGP.iloc[idx, 1:] 
        OGP_pose = np.array([OGP_pose])
        #normalizing values between 0-1 using the max and min of each label column in csv
        OGP_pose[0,0]=(OGP_pose[0,0]-min_x)/(max_x-min_x)
        OGP_pose[0,1]=(OGP_pose[0,1]-min_y)/(max_y-min_y)
        OGP_pose[0,2]=(OGP_pose[0,2]-min_z)/(max_z-min_z)
        OGP_pose[0,3]=(OGP_pose[0,3]-min_w)/(max_w-min_w)
        OGP_pose[0,4]=(OGP_pose[0,4]-min_X)/(max_X-min_X)
        OGP_pose[0,5]=(OGP_pose[0,5]-min_Y)/(max_Y-min_Y)
        OGP_pose[0,6]=(OGP_pose[0,6]-min_Z)/(max_Z-min_Z)
        #print(OGP_pose)
        OGP_pose = OGP_pose.astype('float').flatten()
        sample = {'image': image, 'OGP_pose': OGP_pose}
        
     
        if self.transform:
            sample = self.transform(sample)

        return sample
        
        
        
cloth_dataset = ClothDataset(csv_file='../DataCollection-REDfirst/OGP_dataset_collection_RED.csv',
                                    root_dir='../DataCollection-REDfirst/')

#for i in range(len(cloth_dataset)):
    #sample = cloth_dataset[i]
    #print('targets ', sample['OGP_pose'].shape)        
        
training_loader = torch.utils.data.DataLoader(cloth_dataset, batch_size=8, shuffle=False
, num_workers=1)
        
        
for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        print('labels ',data['OGP_pose'])
        #print('here')
        
          
