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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

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
        #image/=255.0
        image = image.astype('float32')
        image/=255.0
        OGP_pose = self.OGP.iloc[idx, 1:]
        OGP_pose = np.array([OGP_pose])
        OGP_pose = OGP_pose.astype('float').reshape(-1, 1)
        sample = {'image': image, 'OGP_pose': OGP_pose}

        if self.transform:
            sample = self.transform(sample)

        return sample
        
        
        
cloth_dataset = ClothDataset(csv_file='./DataCollection-REDfirst/OGP_dataset_collection_RED.csv',
                                    root_dir='./DataCollection-REDfirst/')

for i in range(len(cloth_dataset)):
    sample = cloth_dataset[i]
    print(i, sample['image'].shape, sample['OGP_pose'].shape)        
        
        
