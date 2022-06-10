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

images_numbering = [2,3,8,9,10,12,13,15,16,20,21,22,24,28,30,36,37,38,39,40,48]

# nb_training_samples = 12

# training_samples = sample(images_numbering, nb_training_samples)
# validation_samples = list(set(images_numbering)-set(training_samples))

patches_per_image = 1

pil_to_tensor = torchvision.transforms.ToTensor()
tensor_to_pil = torchvision.transforms.ToPILImage()

side_size = 512
# size_tain = 40
# size_val = 20

# Crop = torchvision.transforms.RandomCrop(side_size)

Crop = torchvision.transforms.CenterCrop(side_size)

#%% Making DRIVE training set
for p in range(20):
    f0 = loadmat('./data_IOSTAR/DRIVE_datapoints/{}_training_JunctionsPos.mat'.format(p+21))
    I = Image.open('./data_IOSTAR/DRIVE_images/{}_training.tif'.format(p+21))

    I_tensor = pil_to_tensor(I)

    for k in range(patches_per_image):

        # params = Crop.get_params(I_tensor, output_size = (side_size,side_size))
        
        # Icropped = torchvision.transforms.functional.crop(I_tensor,*params)
        Icropped = Crop(I_tensor)
        plt.imshow(Icropped.squeeze().permute([1,2,0]))
        
        image_to_save = tensor_to_pil(Icropped)
        
        # print(patches_per_image*p+k+1)
        image_to_save.save('./data_IOSTAR/train_images/images_IOSTAR/training_IOSTAR_{}.png'.format(patches_per_image*p + k+1))
        
        [j,i,h,w] = [int((I_tensor.shape[1]-side_size)/2),int((I_tensor.shape[2]-side_size)/2), side_size, side_size]

        f1 = f0.copy()

        #Filtering out points located ouside the cropped region

        f1['EndpointPos'] = f0['EndpointPos'][(f0['EndpointPos'][:,1]>i) & (f0['EndpointPos'][:,1]<i+h) & (f0['EndpointPos'][:,0]>j) & (f0['EndpointPos'][:,0]<j+w)] - np.array([[j,i]])
        f1['CrossPos'] = f0['CrossPos'][(f0['CrossPos'][:,1]>i) & (f0['CrossPos'][:,1]<i+h) & (f0['CrossPos'][:,0]>j) & (f0['CrossPos'][:,0]<j+w)] - np.array([[j,i]])
        f1['BiffPos'] = f0['BiffPos'][(f0['BiffPos'][:,1]>i) & (f0['BiffPos'][:,1]<i+h) & (f0['BiffPos'][:,0]>j) & (f0['BiffPos'][:,0]<j+w)] - np.array([[j,i]])
        
        # Saving in data folder
        plt.figure(0)
        plt.imshow(image_to_save)
        plt.scatter(f1['EndpointPos'][:,1], f1['EndpointPos'][:,0], c='r')
        plt.scatter(f1['CrossPos'][:,1], f1['CrossPos'][:,0], c='b')
        plt.scatter(f1['BiffPos'][:,1], f1['BiffPos'][:,0], c='g')
        plt.savefig('./data_IOSTAR/train_images/visualization_train/train_im_scatter{}.png'.format(patches_per_image*p + k+1))
        plt.clf()
        

        savemat('./data_IOSTAR/train_images/IOSTAR_points/IOSTAR_points_{}.mat'.format(patches_per_image*p + k+1), f1)

# #%% Making training set
# for p, index in enumerate(training_samples):

#     if index>=10:
#         f0 = loadmat('./data_IOSTAR/IOSTAR_datapoints/IOSTAR_points_{}.mat'.format(index))
#         I = Image.open('./data_IOSTAR/IOSTAR_images/IOSTAR_image_{}.jpg'.format(index))
#     else:
#         f0 = loadmat('./data_IOSTAR/IOSTAR_datapoints/IOSTAR_points_0{}.mat'.format(index))
#         I = Image.open('./data_IOSTAR/IOSTAR_images/IOSTAR_image_0{}.jpg'.format(index))

#     I_tensor = pil_to_tensor(I)

#     for k in range(patches_per_image):

#         params = Crop.get_params(I_tensor, output_size = (side_size,side_size))
        
#         Icropped = torchvision.transforms.functional.crop(I_tensor,*params)
#         plt.imshow(Icropped.squeeze().permute([1,2,0]))
        
#         image_to_save = tensor_to_pil(Icropped)
        
#         # print(patches_per_image*p+k+1)
#         image_to_save.save('./data_IOSTAR/train_images/images_IOSTAR/training_IOSTAR_{}.png'.format(patches_per_image*p + k+1))
        
#         [j,i,h,w] = params

#         # if p==0:
#         #     print(f0)

#         f1 = f0.copy()

#         #Filtering out points located ouside the cropped region

#         f1['EndpointPos'] = f0['EndpointPos'][(f0['EndpointPos'][:,1]>i) & (f0['EndpointPos'][:,1]<i+h) & (f0['EndpointPos'][:,0]>j) & (f0['EndpointPos'][:,0]<j+w)] - np.array([[j,i]])
#         f1['CrossPos'] = f0['CrossPos'][(f0['CrossPos'][:,1]>i) & (f0['CrossPos'][:,1]<i+h) & (f0['CrossPos'][:,0]>j) & (f0['CrossPos'][:,0]<j+w)] - np.array([[j,i]])
#         f1['BiffPos'] = f0['BiffPos'][(f0['BiffPos'][:,1]>i) & (f0['BiffPos'][:,1]<i+h) & (f0['BiffPos'][:,0]>j) & (f0['BiffPos'][:,0]<j+w)] - np.array([[j,i]])
        
#         # Saving in data folder
#         plt.figure(0)
#         plt.imshow(image_to_save)
#         plt.scatter(f1['EndpointPos'][:,1], f1['EndpointPos'][:,0], c='r')
#         plt.scatter(f1['CrossPos'][:,1], f1['CrossPos'][:,0], c='b')
#         plt.scatter(f1['BiffPos'][:,1], f1['BiffPos'][:,0], c='g')
#         plt.savefig('./data_IOSTAR/train_images/visualization_train/train_im_scatter{}.png'.format(patches_per_image*p + k+1))
#         plt.clf()
        

#         savemat('./data_IOSTAR/train_images/IOSTAR_points/IOSTAR_points_{}.mat'.format(patches_per_image*p + k+1), f1)

#%% Making DRIVE test set
for p in range(20):
    if p+1>=10:
        f0 = loadmat('./data_IOSTAR/DRIVE_datapoints/{}_test_JunctionsPos.mat'.format(p+1))
        I = Image.open('./data_IOSTAR/DRIVE_images/{}_test.tif'.format(p+1))
    else:
        f0 = loadmat('./data_IOSTAR/DRIVE_datapoints/0{}_test_JunctionsPos.mat'.format(p+1))
        I = Image.open('./data_IOSTAR/DRIVE_images/0{}_test.tif'.format(p+1))

    I_tensor = pil_to_tensor(I)

    for k in range(patches_per_image):

        # params = Crop.get_params(I_tensor, output_size = (side_size,side_size))
        
        # Icropped = torchvision.transforms.functional.crop(I_tensor,*params)
        Icropped = Crop(I_tensor)
        plt.imshow(Icropped.squeeze().permute([1,2,0]))
        
        image_to_save = tensor_to_pil(Icropped)
        
        # print(patches_per_image*p+k+1)
        image_to_save.save('./data_IOSTAR/val_images/images_IOSTAR/validation_IOSTAR_{}.png'.format(patches_per_image*p + k+1))
        
        [j,i,h,w] = [int((I_tensor.shape[1]-side_size)/2),int((I_tensor.shape[2]-side_size)/2), side_size, side_size]

        f1 = f0.copy()

        #Filtering out points located ouside the cropped region

        f1['EndpointPos'] = f0['EndpointPos'][(f0['EndpointPos'][:,1]>i) & (f0['EndpointPos'][:,1]<i+h) & (f0['EndpointPos'][:,0]>j) & (f0['EndpointPos'][:,0]<j+w)] - np.array([[j,i]])
        f1['CrossPos'] = f0['CrossPos'][(f0['CrossPos'][:,1]>i) & (f0['CrossPos'][:,1]<i+h) & (f0['CrossPos'][:,0]>j) & (f0['CrossPos'][:,0]<j+w)] - np.array([[j,i]])
        f1['BiffPos'] = f0['BiffPos'][(f0['BiffPos'][:,1]>i) & (f0['BiffPos'][:,1]<i+h) & (f0['BiffPos'][:,0]>j) & (f0['BiffPos'][:,0]<j+w)] - np.array([[j,i]])
        
        # Saving in data folder
        plt.figure(0)
        plt.imshow(image_to_save)
        plt.scatter(f1['EndpointPos'][:,1], f1['EndpointPos'][:,0], c='r')
        plt.scatter(f1['CrossPos'][:,1], f1['CrossPos'][:,0], c='b')
        plt.scatter(f1['BiffPos'][:,1], f1['BiffPos'][:,0], c='g')
        plt.savefig('./data_IOSTAR/val_images/visualization_val/val_im_scatter{}.png'.format(patches_per_image*p + k+1))
        plt.clf()
        

        savemat('./data_IOSTAR/val_images/IOSTAR_points/IOSTAR_points_{}.mat'.format(patches_per_image*p + k+1), f1)
    

#%% Making IOSTAR validation set

for p, index in enumerate(images_numbering):

    if index>=10:
        f0 = loadmat('./data_IOSTAR/IOSTAR_datapoints/IOSTAR_points_{}.mat'.format(index))
        I = Image.open('./data_IOSTAR/IOSTAR_images/IOSTAR_image_{}.jpg'.format(index))
    else:
        f0 = loadmat('./data_IOSTAR/IOSTAR_datapoints/IOSTAR_points_0{}.mat'.format(index))
        I = Image.open('./data_IOSTAR/IOSTAR_images/IOSTAR_image_0{}.jpg'.format(index))

    I_tensor = pil_to_tensor(I)

    for k in range(patches_per_image):

        # params = Crop.get_params(I_tensor, output_size = (side_size,side_size))
        
        # Icropped = torchvision.transforms.functional.crop(I_tensor,*params)
        Icropped = Crop(I_tensor)
        plt.imshow(Icropped.squeeze().permute([1,2,0]))
        
        image_to_save = tensor_to_pil(Icropped)
        
        # print(patches_per_image*p+k+1)
        image_to_save.save('./data_IOSTAR/val_images/images_IOSTAR/validation_IOSTAR_{}.png'.format(patches_per_image*p + k + 1 + patches_per_image*20))
        
        [j,i,h,w] = [int((I_tensor.shape[1]-side_size)/2), int((I_tensor.shape[2]-side_size)/2), side_size, side_size]

        f1 = f0.copy()

        #Filtering out points located ouside the cropped region

        f1['EndpointPos'] = f0['EndpointPos'][(f0['EndpointPos'][:,1]>i) & (f0['EndpointPos'][:,1]<i+h) & (f0['EndpointPos'][:,0]>j) & (f0['EndpointPos'][:,0]<j+w)] - np.array([[j,i]])
        f1['CrossPos'] = f0['CrossPos'][(f0['CrossPos'][:,1]>i) & (f0['CrossPos'][:,1]<i+h) & (f0['CrossPos'][:,0]>j) & (f0['CrossPos'][:,0]<j+w)] - np.array([[j,i]])
        f1['BiffPos'] = f0['BiffPos'][(f0['BiffPos'][:,1]>i) & (f0['BiffPos'][:,1]<i+h) & (f0['BiffPos'][:,0]>j) & (f0['BiffPos'][:,0]<j+w)] - np.array([[j,i]])

        # print(f1['EndpointPos'].max())
        plt.figure(0)
        plt.imshow(image_to_save)
        plt.scatter(f1['EndpointPos'][:,1], f1['EndpointPos'][:,0], c='r')
        plt.scatter(f1['CrossPos'][:,1], f1['CrossPos'][:,0], c='b')
        plt.scatter(f1['BiffPos'][:,1], f1['BiffPos'][:,0], c='g')
        plt.savefig('./data_IOSTAR/val_images/visualization_val/val_im_scatter{}.png'.format(patches_per_image*p + k + 1 + patches_per_image*20))
        plt.clf()
        
        # Saving in data folder

        savemat('./data_IOSTAR/val_images/IOSTAR_points/IOSTAR_points_{}.mat'.format(patches_per_image*p + k + 1 + patches_per_image*20), f1)


