# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os

import pdb
import torch.utils.data as data
from torchvision.transforms import ToTensor, Resize

class WholeDataLoader(Dataset):
    def __init__(self,option):
        self.data_split = option.data_split
        self.biased_data_dir = option.data_dir
        self.test_data_dir = '/media/doh/ECECE209ECE1CDC0/research/cats_and_dogs/dataset/dogs_and_cats/sampled/test/'
        self.cnd_data = os.listdir(self.biased_data_dir)
        light_txt = '/media/doh/ECECE209ECE1CDC0/research/cats_and_dogs/dataset/dogs_and_cats/list_bright.txt'
        dark_txt = '/media/doh/ECECE209ECE1CDC0/research/cats_and_dogs/dataset/dogs_and_cats/list_dark.txt'

        with open(light_txt, 'r') as f:
            bright = [line.strip() for line in f]

        with open(dark_txt, 'r') as f:
            dark = [line.strip() for line in f]

        if self.data_split == 'train':

            # cat dog label ==> [0,1]
            self.label = np.array([], dtype = np.int8)
            for i in range(len(self.cnd_data)):
                if 'cat' == self.cnd_data[i][0:3]:
                    self.label = np.append(self.label, [0], axis=0)
                else :
                    self.label = np.append(self.label, [1], axis=0)

            # color
            
            self.color_label = np.array([], dtype = np.int8)
                # bright / dark label ==> [0,1]
            for i in range(len(self.cnd_data)):
                
                if self.cnd_data[i] in bright:
                    self.color_label = np.append(self.color_label, [0], axis=0)
            
                if self.cnd_data[i] in dark:
                    self.color_label = np.append(self.color_label, [1], axis=0)
                    
            # pdb.set_trace()
            self.transform_data = transforms.Compose([
                transforms.Resize(256),
                transforms.ColorJitter(),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(128),
                transforms.ToTensor()
            ])
          
            # length 
            self.dataset_length = len(self.cnd_data)

# ---------------------------------Test----------------------------------------------------------------------------------
        elif self.data_split == 'test': 

            self.test_data = os.listdir(self.test_data_dir)
            # cat dog label ==> [0,1]
            self.label = np.array([], dtype = np.int8)
            for i in range(len(self.test_data)):
                if 'cat' == self.test_data[i][0:3]:
                    self.label = np.append(self.label, [0], axis=0)
                else :
                    self.label = np.append(self.label, [1], axis=0)

            # color
            
            self.color_label = np.array([], dtype = np.int8)
                # bright / dark label ==> [0,1]
            for i in range(len(self.test_data)):
                if self.test_data[i] in bright:
                    self.color_label = np.append(self.color_label, [0], axis=0)
                if self.test_data[i] in dark:
                    self.color_label = np.append(self.color_label, [1], axis=0)

            self.transform_data = transforms.Compose([
                transforms.Resize(256),
                transforms.ColorJitter(),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(128),
                transforms.ToTensor()
            ])
           
            # length 
            self.dataset_length = len(self.test_data)

    def __getitem__(self,index):
        if self.data_split == 'train':
            images = Image.open(os.path.join(self.biased_data_dir,self.cnd_data[index]))
            images = self.transform_data(images)
            color = self.color_label[index]
            label = self.label[index]
        # print(images.shape)
        elif self.data_split == 'test': 
            images = Image.open(os.path.join(self.test_data_dir,self.test_data[index]))
            images = self.transform_data(images)
            color = self.color_label[index]
            label = self.label[index]
        return  images, color, label.astype(np.float)
  
        

    def __len__(self):
        return self.dataset_length



