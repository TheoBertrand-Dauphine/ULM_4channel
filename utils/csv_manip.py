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

f = open('./data/csv/ULM_points_halfright_label_16_07.csv',newline='\n')

list_of_points = []

for row in f.readlines():
    p = row.split(',')
    
    list_of_points.append([int(p[1]),int(p[2]),p[0]])

I = Image.open('./data/rat_brain_invivo_halfright_sqrt1_16_07.png')
pil_to_tensor = torchvision.transforms.ToTensor()
tensor_to_pil = torchvision.transforms.ToPILImage()

I_tensor = pil_to_tensor(I)

side_size = 256
size_tain = 40
size_val = 20

A = torchvision.transforms.RandomCrop(side_size)

for k in range(math.floor(size_tain/2)):
    while True:
        params = A.get_params(I_tensor,output_size = (side_size,side_size))
        
        Icropped = torchvision.transforms.functional.crop(I_tensor,*params)
        plt.imshow(Icropped.squeeze())
        
        image_to_save = tensor_to_pil(Icropped)
        
        image_to_save.save('./data/train_images/images_ULM/training_ULM_{}.png'.format(k+1))
        
        [i,j,h,w] = params

        cropped_list = []
        
        for x in list_of_points:
            if x[1]>i and x[1]<i+h and x[0]>j and x[0]<j+w:
                cropped_list.append([x[1]-i,x[0]-j,x[2]])
        
        with open("./data/train_images/ULM_points/point_list_{}.csv".format(k+1),"w") as f:
            wr = csv.writer(f)
            wr.writerows(cropped_list)
        if len(cropped_list)>0:
            break
    
    
    
# #%%

# B = torch.zeros([side_size,side_size])
# gaussian_blur = torchgeometry.image.gaussian.GaussianBlur((17,17),(1,1)).cuda()


# for x in cropped_list:
#     B[x[0],x[1]] = 1

# B_blurr = gaussian_blur(B.unsqueeze(0).unsqueeze(0))

# plt.imshow(B_blurr.squeeze().squeeze())

# plt.show()

# plt.imshow(image_to_save)

# #%%

# B = torch.zeros(I_tensor.size())
# gaussian_blur = torchgeometry.image.gaussian.GaussianBlur((17,17),(5,5)).cuda()


# for x in list_of_points:
#     B[0,x[1],x[0]] = 1

# B_blurr = gaussian_blur(B.unsqueeze(0))

# plt.imshow(B_blurr.squeeze().squeeze())
    

#%% Left part of hires brain goes to validation set

f = open('./data/csv/ULM_points_halfleft_label_16_07.csv',newline='\n')

list_of_points = []

for row in f.readlines():
    p = row.split(',')
    
    list_of_points.append([int(p[1]),int(p[2]),p[0]])

I = Image.open('./data/rat_brain_invivo_halfleft_sqrt1_16_07.png')
pil_to_tensor = torchvision.transforms.ToTensor()
tensor_to_pil = torchvision.transforms.ToPILImage()

I_tensor = pil_to_tensor(I)

A = torchvision.transforms.RandomCrop(side_size)

for k in range(math.floor(size_val/2)):
    while True:
        params = A.get_params(I_tensor,output_size = (side_size,side_size))
        
        Icropped = torchvision.transforms.functional.crop(I_tensor,*params)
        plt.imshow(Icropped.squeeze())
        
        image_to_save = tensor_to_pil(Icropped)
        
        image_to_save.save('./data/val_images/images_ULM/validation_ULM_{}.png'.format(k+1))
        
        [i,j,h,w] = params

        cropped_list = []
        
        for x in list_of_points:
            if x[1]>i and x[1]<i+h and x[0]>j and x[0]<j+w:
                cropped_list.append([x[1]-i,x[0]-j,x[2]])
        
        with open("./data/val_images/ULM_points/point_list_{}.csv".format(k+1),"w") as f:
            wr = csv.writer(f)
            wr.writerows(cropped_list)
        if len(cropped_list)>0:
            break


#%%
        
I_tensor = pil_to_tensor(I)

plt.figure(0)
plt.clf()
plt.imshow(I_tensor.squeeze())
for x in list_of_points:
    plt.scatter(x[0],x[1],c='r')
    


#%% Left half of bolus image goes to training set

f = open('./data/csv/ULM_points_full_bolus_label_16_07.csv', newline='\n')

list_of_points = []

for row in f.readlines():
    p = row.split(',')
    
    list_of_points.append([int(p[1]),int(p[2]),p[0]])

I = Image.open('./data/rat_brain_bolus_full_sqrt1_16_07.png')
pil_to_tensor = torchvision.transforms.ToTensor()
tensor_to_pil = torchvision.transforms.ToPILImage()

I_tensor = pil_to_tensor(I)

I_tensor = I_tensor[:,:,0:round(I_tensor.shape[2]/2)]


for k in range(math.floor(size_tain/2)):
    while True:
        params = A.get_params(I_tensor,output_size = (side_size,side_size))
        
        Icropped = torchvision.transforms.functional.crop(I_tensor,*params)
        plt.imshow(Icropped.squeeze())
        
        image_to_save = tensor_to_pil(Icropped)
        
        # image_to_save.save('./validation_ULM_single/validation_ULM_{}.png'.format(k+1))
        image_to_save.save('./data/train_images/images_ULM/training_ULM_{}.png'.format(k+21))
        
        [i,j,h,w] = params

        cropped_list = []
        
        for x in list_of_points:
            if x[1]>i and x[1]<i+h and x[0]>j and x[0]<j+w:
                cropped_list.append([x[1]-i,x[0]-j,x[2]])
        
        with open("./data/train_images/ULM_points/point_list_{}.csv".format(k+21),"w") as f:
            wr = csv.writer(f)
            wr.writerows(cropped_list)
        if len(cropped_list)>0:
            break

#%%
plt.figure(0)
plt.imshow(Icropped.squeeze())
for i in range(len(cropped_list)):
    plt.scatter(cropped_list[i][1],cropped_list[i][0],c='r')


#%% Right part of bolus brain image goes to validation set

I_tensor = pil_to_tensor(I)

I_tensor = I_tensor[:,:,round(I_tensor.shape[2]/2):I_tensor.shape[2]]


for k in range(math.floor(size_val/2)):
    while True:
        params = A.get_params(I_tensor,output_size = (side_size,side_size))
        
        Icropped = torchvision.transforms.functional.crop(I_tensor,*params)
        plt.imshow(Icropped.squeeze())
        
        image_to_save = tensor_to_pil(Icropped)
        
        # image_to_save.save('./validation_ULM_single/validation_ULM_{}.png'.format(k+1))
        image_to_save.save('./data/val_images/images_ULM/validation_ULM_{}.png'.format(k+11))
        
        [i,j,h,w] = params
        j = j + round(I_tensor.shape[2])

        cropped_list = []
        
        for x in list_of_points:
            if x[1]>i and x[1]<i+h and x[0]>j and x[0]<j+w:
                cropped_list.append([x[1]-i,x[0]-j,x[2]])
        
        with open("./data/val_images/ULM_points/point_list_{}.csv".format(k+11),"w") as f:
            wr = csv.writer(f)
            wr.writerows(cropped_list)
        if len(cropped_list)>0:
            break


#%%
I_tensor = pil_to_tensor(I)

plt.figure(0)
plt.clf()
plt.imshow(I_tensor.squeeze())
for x in list_of_points:
    plt.scatter(x[0],x[1],c='r')
    
