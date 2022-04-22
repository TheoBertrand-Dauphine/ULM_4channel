# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:45:02 2021

@author: theot
"""
# import random
import csv
from PIL import Image
import torchvision
from random import sample
# import torch
# import math
# import torchgeometry.image.gaussian

import matplotlib.pyplot as plt

from scipy.io import loadmat, savemat
import numpy as np

images_numbering = [2,3,5,8,9,10,12,13,15,16,17,20,21,22,24,28,30,34,36,37,38,39,40,48]

nb_training_samples = 14

training_samples = sample(images_numbering, nb_training_samples)
validation_samples = list(set(images_numbering)-set(training_samples))

patches_per_image = 2

pil_to_tensor = torchvision.transforms.ToTensor()
tensor_to_pil = torchvision.transforms.ToPILImage()

side_size = 256
# size_tain = 40
# size_val = 20

Crop = torchvision.transforms.RandomCrop(side_size)

#%% Making training set
for p, index in enumerate(training_samples):

    if index>=10:
        f0 = loadmat('./data_IOSTAR/IOSTAR_datapoints/IOSTAR_points_{}.mat'.format(index))
        I = Image.open('./data_IOSTAR/IOSTAR_images/IOSTAR_image_{}.jpg'.format(index))
    else:
        f0 = loadmat('./data_IOSTAR/IOSTAR_datapoints/IOSTAR_points_0{}.mat'.format(index))
        I = Image.open('./data_IOSTAR/IOSTAR_images/IOSTAR_image_0{}.jpg'.format(index))

    I_tensor = pil_to_tensor(I)

    for k in range(patches_per_image):

        params = Crop.get_params(I_tensor, output_size = (side_size,side_size))
        
        Icropped = torchvision.transforms.functional.crop(I_tensor,*params)
        plt.imshow(Icropped.squeeze().permute([1,2,0]))
        
        image_to_save = tensor_to_pil(Icropped)
        
        # print(patches_per_image*p+k+1)
        image_to_save.save('./data_IOSTAR/train_images/images_IOSTAR/training_IOSTAR_{}.png'.format(patches_per_image*p + k+1))
        
        [j,i,h,w] = params

        # if p==0:
        #     print(f0)

        f1 = f0.copy()

        #Filtering out points located ouside the cropped region

        f1['EndpointPos'] = f0['EndpointPos'][(f0['EndpointPos'][:,1]>i) & (f0['EndpointPos'][:,1]<i+h) & (f0['EndpointPos'][:,0]>j) & (f0['EndpointPos'][:,0]<j+w)] - np.array([[j,i]])
        f1['CrossPos'] = f0['CrossPos'][(f0['CrossPos'][:,1]>i) & (f0['CrossPos'][:,1]<i+h) & (f0['CrossPos'][:,0]>j) & (f0['CrossPos'][:,0]<j+w)] - np.array([[j,i]])
        f1['BiffPos'] = f0['BiffPos'][(f0['BiffPos'][:,1]>i) & (f0['BiffPos'][:,1]<i+h) & (f0['BiffPos'][:,0]>j) & (f0['BiffPos'][:,0]<j+w)] - np.array([[j,i]])
        
        # Saving in data folder

        savemat('./data_IOSTAR/train_images/IOSTAR_points/IOSTAR_points_{}.csv'.format(patches_per_image*p + k+1), f1)
    

#%% Making validation set

for p, index in enumerate(validation_samples):

    if index>=10:
        f0 = loadmat('./data_IOSTAR/IOSTAR_datapoints/IOSTAR_points_{}.mat'.format(index))
        I = Image.open('./data_IOSTAR/IOSTAR_images/IOSTAR_image_{}.jpg'.format(index))
    else:
        f0 = loadmat('./data_IOSTAR/IOSTAR_datapoints/IOSTAR_points_0{}.mat'.format(index))
        I = Image.open('./data_IOSTAR/IOSTAR_images/IOSTAR_image_0{}.jpg'.format(index))

    I_tensor = pil_to_tensor(I)

    for k in range(patches_per_image):

        params = Crop.get_params(I_tensor, output_size = (side_size,side_size))
        
        Icropped = torchvision.transforms.functional.crop(I_tensor,*params)
        plt.imshow(Icropped.squeeze().permute([1,2,0]))
        
        image_to_save = tensor_to_pil(Icropped)
        
        # print(patches_per_image*p+k+1)
        image_to_save.save('./data_IOSTAR/val_images/images_IOSTAR/validation_IOSTAR_{}.png'.format(patches_per_image*p + k+1))
        
        [j,i,h,w] = params

        f1 = f0.copy()

        #Filtering out points located ouside the cropped region

        f1['EndpointPos'] = f0['EndpointPos'][(f0['EndpointPos'][:,1]>i) & (f0['EndpointPos'][:,1]<i+h) & (f0['EndpointPos'][:,0]>j) & (f0['EndpointPos'][:,0]<j+w)] - np.array([[j,i]])
        f1['CrossPos'] = f0['CrossPos'][(f0['CrossPos'][:,1]>i) & (f0['CrossPos'][:,1]<i+h) & (f0['CrossPos'][:,0]>j) & (f0['CrossPos'][:,0]<j+w)] - np.array([[j,i]])
        f1['BiffPos'] = f0['BiffPos'][(f0['BiffPos'][:,1]>i) & (f0['BiffPos'][:,1]<i+h) & (f0['BiffPos'][:,0]>j) & (f0['BiffPos'][:,0]<j+w)] - np.array([[j,i]])

        # print(f1['EndpointPos'].max())
        
        # Saving in data folder

        savemat('./data_IOSTAR/val_images/IOSTAR_points/IOSTAR_points_{}.csv'.format(patches_per_image*p + k+1), f1)


