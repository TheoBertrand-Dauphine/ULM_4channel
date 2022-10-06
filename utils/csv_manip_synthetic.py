# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:45:02 2021

@author: theot
"""
# import random
import csv
from PIL import Image
import torchvision
import torch
import math
# import torchgeometry.image.gaussian

import matplotlib.pyplot as plt


# #%% Making training set

# f0 = open('./data_synthetic/csv/training_synthetic_point.csv',newline='\n')

# list_of_points = []

# for row in f0.readlines():
#     p = row.split(',')
    
#     list_of_points.append([int(p[1]),int(p[2]),p[0]])

# I = Image.open('./data_synthetic/training_synthetic_image.png')
# pil_to_tensor = torchvision.transforms.ToTensor()
# tensor_to_pil = torchvision.transforms.ToPILImage()

# I_tensor = pil_to_tensor(I)

# side_size = 256
# size_tain = 40
# size_val = 20

# A = torchvision.transforms.RandomCrop(side_size)

# for k in range(math.floor(size_tain)):
#     while True:
#         params = A.get_params(I_tensor,output_size = (side_size,side_size))
        
#         Icropped = torchvision.transforms.functional.crop(I_tensor,*params)
#         # plt.imshow(Icropped.squeeze().permute([1,2,0]))
        
#         image_to_save = tensor_to_pil(Icropped)
        
#         image_to_save.save('./data_synthetic/train_images/images_ULM/training_ULM_{}.png'.format(k+1))
        
#         [i,j,h,w] = params

#         cropped_list = []
        
#         for x in list_of_points:
#             if x[1]>i and x[1]<i+h and x[0]>j and x[0]<j+w:
#                 cropped_list.append([x[1]-i,x[0]-j,x[2]])
        
#         with open("./data_synthetic/train_images/ULM_points/point_list_{}.csv".format(k+1),"w") as f:
#             wr = csv.writer(f)
#             wr.writerows(cropped_list)
#         if len(cropped_list)>0:
#             break
    

# #%% Making validation set

# f0 = open('./data_synthetic/csv/validation_synthetic_points.csv',newline='\n')

# list_of_points = []

# for row in f0.readlines():
#     p = row.split(',')
    
#     list_of_points.append([int(p[1]),int(p[2]),p[0]])

# I = Image.open('./data_synthetic/validation_synthetic_image.png')
# pil_to_tensor = torchvision.transforms.ToTensor()
# tensor_to_pil = torchvision.transforms.ToPILImage()

# I_tensor = pil_to_tensor(I)

# A = torchvision.transforms.RandomCrop(side_size)

# for k in range(math.floor(size_val)):
#     while True:
#         params = A.get_params(I_tensor,output_size = (side_size,side_size))
        
#         Icropped = torchvision.transforms.functional.crop(I_tensor,*params)
        
#         image_to_save = tensor_to_pil(Icropped)
        
#         image_to_save.save('./data_synthetic/val_images/images_ULM/validation_ULM_{}.png'.format(k+1))
        
#         [i,j,h,w] = params

#         cropped_list = []
        
#         for x in list_of_points:
#             if x[1]>i and x[1]<i+h and x[0]>j and x[0]<j+w:
#                 cropped_list.append([x[1]-i,x[0]-j,x[2]])
        
#         # plt.imshow(Icropped.squeeze().permute([1,2,0]))
#         # for i in range(len(cropped_list)):
#         #     plt.scatter(cropped_list[i][1], cropped_list[i][0])
#         # plt.show()
#         with open("./data_synthetic/val_images/ULM_points/point_list_{}.csv".format(k+1),"w") as f:
#             wr = csv.writer(f)
#             wr.writerows(cropped_list)
#         if len(cropped_list)>0:
#             break


# #%% Making big validation test image

# im_padded = torch.nn.functional.pad(I_tensor, (24,24,12,12))[0]

# image_to_save = tensor_to_pil(im_padded)

# image_to_save.save('./data_synthetic/test_images/images_ULM/big_test_image.png')

# padded_list = []
# for x in list_of_points:
#     padded_list.append([x[1]+12,x[0]+24,x[2]])

# with open("./data_synthetic/test_images/ULM_points/point_list_test_image.csv","w") as f:
#     wr = csv.writer(f)
#     wr.writerows(padded_list)



side_size = 256

pil_to_tensor = torchvision.transforms.ToTensor()
tensor_to_pil = torchvision.transforms.ToPILImage()

import os

for mydir in ['./data_synthetic/train_images/images_ULM/', './data_synthetic/train_images/ULM_points/', './data_synthetic/val_images/images_ULM/', './data_synthetic/val_images/ULM_points/', './data_synthetic/val_images/fig_viz/', './data_synthetic/train_images/fig_viz/' ]:
    filelist = [ f for f in os.listdir(mydir) if f.endswith(".png") or f.endswith(".csv") ]
    for f in filelist:
        os.remove(os.path.join(mydir, f))

#%% training set 
f = open('./data_synthetic/csv/training_synthetic_point.csv',newline='\n')

list_of_points = []

for row in f.readlines():
    p = row.split(',')
    
    list_of_points.append([int(p[1]),int(p[2]),p[0]])

I = Image.open('./data_synthetic/training_synthetic_image.png')


I_tensor = pil_to_tensor(I).squeeze()[0]



nx = int(I_tensor.shape[0]/side_size)
ny = int(I_tensor.shape[1]/side_size)
nb_images_train = nx*ny

print(nb_images_train)

for i in range(nx):
    for j in range(ny):
        params = (i*side_size, j*side_size, side_size, side_size)
        Icropped = torchvision.transforms.functional.crop(I_tensor, *params)
        # plt.imshow(Icropped.squeeze())
        
        image_to_save = tensor_to_pil(Icropped)
        
        image_to_save.save('./data_synthetic/train_images/images_ULM/training_ULM_{}.png'.format(ny*i+j+1))
        
        [t,l,h,w] = params
    
        cropped_list = []
        
        for x in list_of_points:
            if x[1]>t and x[1]<t+h and x[0]>l and x[0]<l+w:
                cropped_list.append([x[1]-t,x[0]-l,x[2]])
        
        with open("./data_synthetic/train_images/ULM_points/point_list_{}.csv".format(ny*i+j+1),"w") as f:
            wr = csv.writer(f)
            wr.writerows(cropped_list)

        plt.figure(0)
        plt.clf()
        plt.imshow(Icropped.squeeze())
        for x in cropped_list:
            plt.scatter(x[1],x[0],c='r')
        plt.savefig('./data_synthetic/train_images/fig_viz/viz_{}.png'.format(ny*i+j+1))
        plt.close()

#%% training set left brain


f = open('./data_synthetic/csv/validation_synthetic_points.csv', newline='\n')

list_of_points = []

for row in f.readlines():
    p = row.split(',')
    
    list_of_points.append([int(p[1]),int(p[2]),p[0]])

I = Image.open('./data_synthetic/validation_synthetic_image.png')

I_tensor = pil_to_tensor(I)[0]

nx = int(I_tensor.shape[0]/side_size)
ny = int(I_tensor.shape[1]/side_size)
nb_images_val = nx*ny

print(nb_images_val)

for i in range(nx):
    for j in range(ny):
        params = (i*side_size, j*side_size, side_size, side_size)
        Icropped = torchvision.transforms.functional.crop(I_tensor, *params)
        # plt.imshow(Icropped.squeeze())
        
        image_to_save = tensor_to_pil(Icropped)
        
        image_to_save.save('./data_synthetic/val_images/images_ULM/training_ULM_{}.png'.format(ny*i+j+1))
        
        [t,l,h,w] = params
    
        cropped_list = []
        
        for x in list_of_points:
            if x[1]>t and x[1]<t+h and x[0]>l and x[0]<l+w:
                cropped_list.append([x[1]-t,x[0]-l,x[2]])
        
        with open("./data_synthetic/val_images/ULM_points/point_list_{}.csv".format(ny*i+j+1),"w") as f:
            wr = csv.writer(f)
            wr.writerows(cropped_list)

        plt.figure(0)
        plt.clf()
        plt.imshow(Icropped.squeeze())
        for x in cropped_list:
            plt.scatter(x[1],x[0],c='r')
        plt.savefig('./data_synthetic/val_images/fig_viz/viz_{}.png'.format(ny*i+j+1))
        plt.close()
            