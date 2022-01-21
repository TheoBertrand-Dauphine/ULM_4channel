# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:45:02 2021

@author: theot
"""
# import random
import csv
from PIL import Image
import torchvision
# import torch
import math
# import torchgeometry.image.gaussian

import matplotlib.pyplot as plt


#%% Making training set

f0 = open('./data_synthetic/csv/training_synthetic_point.csv',newline='\n')

list_of_points = []

for row in f0.readlines():
    p = row.split(',')
    
    list_of_points.append([int(p[1]),int(p[2]),p[0]])

I = Image.open('./data_synthetic/training_synthetic_image.png')
pil_to_tensor = torchvision.transforms.ToTensor()
tensor_to_pil = torchvision.transforms.ToPILImage()

I_tensor = pil_to_tensor(I)

side_size = 256
size_tain = 40
size_val = 20

A = torchvision.transforms.RandomCrop(side_size)

for k in range(math.floor(size_tain)):
    while True:
        params = A.get_params(I_tensor,output_size = (side_size,side_size))
        
        Icropped = torchvision.transforms.functional.crop(I_tensor,*params)
        plt.imshow(Icropped.squeeze().permute([1,2,0]))
        
        image_to_save = tensor_to_pil(Icropped)
        
        image_to_save.save('./data_synthetic/train_images/images_ULM/training_ULM_{}.png'.format(k+1))
        
        [i,j,h,w] = params

        cropped_list = []
        
        for x in list_of_points:
            if x[1]>i and x[1]<i+h and x[0]>j and x[0]<j+w:
                cropped_list.append([x[1]-i,x[0]-j,x[2]])
        
        with open("./data_synthetic/train_images/ULM_points/point_list_{}.csv".format(k+1),"w") as f:
            wr = csv.writer(f)
            wr.writerows(cropped_list)
        if len(cropped_list)>0:
            break
    

#%% Making validation set

f0 = open('./data_synthetic/csv/validation_synthetic_points.csv',newline='\n')

list_of_points = []

for row in f0.readlines():
    p = row.split(',')
    
    list_of_points.append([int(p[1]),int(p[2]),p[0]])

I = Image.open('./data_synthetic/validation_synthetic_image.png')
pil_to_tensor = torchvision.transforms.ToTensor()
tensor_to_pil = torchvision.transforms.ToPILImage()

I_tensor = pil_to_tensor(I)

A = torchvision.transforms.RandomCrop(side_size)

for k in range(math.floor(size_val)):
    while True:
        params = A.get_params(I_tensor,output_size = (side_size,side_size))
        
        Icropped = torchvision.transforms.functional.crop(I_tensor,*params)
        
        image_to_save = tensor_to_pil(Icropped)
        
        image_to_save.save('./data_synthetic/val_images/images_ULM/validation_ULM_{}.png'.format(k+1))
        
        [i,j,h,w] = params

        cropped_list = []
        
        for x in list_of_points:
            if x[1]>i and x[1]<i+h and x[0]>j and x[0]<j+w:
                cropped_list.append([x[1]-i,x[0]-j,x[2]])
        
        plt.imshow(Icropped.squeeze().permute([1,2,0]))
        for i in range(len(cropped_list)):
            plt.scatter(cropped_list[i][1], cropped_list[i][0])
        plt.show()
        with open("./data_synthetic/val_images/ULM_points/point_list_{}.csv".format(k+1),"w") as f:
            wr = csv.writer(f)
            wr.writerows(cropped_list)
        if len(cropped_list)>0:
            break

