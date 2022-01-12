import torch
import torch.nn as nn
# from torch.autograd import Variable
# import torch.nn.functional as F
import torchvision
from PIL import Image
# import re
import numpy as np
import matplotlib.pyplot as plt

from network_designs import Unet_for_ULM

import csv
import torchgeometry.image.gaussian



pil_to_tensor = torchvision.transforms.ToTensor()

net = Unet_for_ULM().cuda()

net.load_state_dict(torch.load('model_08_03_soir_noise001_20F1'))



validation_mean_loss = 0
            
precision = 0
recall = 0
F1 = 0

pixel_tol = 7

local_max_filt = nn.MaxPool2d(21,stride=1,padding=10).cuda()

gaussian_blur = torchgeometry.image.gaussian.GaussianBlur((17,17),(3,3)).cuda()

l2loss = nn.MSELoss(reduction='sum')
#%%
for i in range(20):
    with torch.no_grad():        
        im = Image.open('validation_ULM/validation_ULM_{}.png'.format(i+1))
    
            
        im_tensor = torch.unsqueeze(pil_to_tensor(im),0)
        im_tensor = (im_tensor[:,0,:,:].unsqueeze(0))
        im_tensor = (im_tensor + 0.0*torch.randn(im_tensor.size())).cuda()
        
        point_list = []
        with open('validation_ULM_points/point_list_{}.csv'.format(i+1)) as f:
            rd = csv.reader(f)
            for row in rd:
                if len(row)!=0:
                    point_list.append([int(row[0]),int(row[1]),row[2]])
        
        im_bif = torch.zeros([1,3,im_tensor.size(2),im_tensor.size(3)])
        
        for j in range(len(point_list)):
            if point_list[j][2] == 'endpoint':
                im_bif[0,0,point_list[j][0],point_list[j][1]] = 1
                point_list[j] = [0,point_list[j][0],point_list[j][1]]
            if point_list[j][2] == 'biffurcation':
                im_bif[0,1,point_list[j][0],point_list[j][1]] = 1
                point_list[j] = [1,point_list[j][0],point_list[j][1]]
            if point_list[j][2] == 'crossing':
                im_bif[0,2,point_list[j][0],point_list[j][1]] = 1
                point_list[j] = [2,point_list[j][0],point_list[j][1]]
                
        im_bif_blur = gaussian_blur(im_bif).cuda()
        im_bif_blur = im_bif_blur/(im_bif_blur.max())
        
        net_output = net(im_tensor)
        
        loss = l2loss(net_output,im_bif_blur)
        validation_mean_loss +=loss/20
        
        max_filtered_output = (local_max_filt(net_output)).double()
        im_points = ((max_filtered_output==net_output)*(net_output>0.05)).double().squeeze()
        list_points_detected = im_points[:,:].nonzero()
        
        nb_detected=0
        
        point_array = torch.tensor(point_list)
        
        for p in range(list_points_detected.size(0)):
            if ((list_points_detected[p,1:].cpu()-point_array[:,1:])**2).sum(dim=1).min()<pixel_tol**2: 
                nb_detected+=1
                
        precision += min(len(list_points_detected),1)*nb_detected/(max(len(list_points_detected),1))/20
        recall += nb_detected/(max(len(point_list),1))/20
        
        F1 += 2*nb_detected/max(len(point_list) + len(list_points_detected),1)/20
        
        # print(min(len(list_points_detected),1)*nb_detected/(max(len(list_points_detected),1)))

print('MEAN precision: {} '.format(precision))        
print('MEAN recall: {} '.format(recall))  
print('MEAN F1-score: {} '.format(F1))  

print('mean validation loss is {}'.format(validation_mean_loss))

# #%% Precision-recall diagram
# n=200
# a=np.linspace(0.0,1,n)


# list_precision = np.zeros(n)
# list_recall = np.zeros(n)
# list_F1 = np.zeros(n)


# precision=0
# recall=0
# F1 = 0
# for i in range(20):
#     with torch.no_grad():        
#         im = Image.open('validation_ULM/validation_ULM_{}.png'.format(i+1))
    
            
#         im_tensor = torch.unsqueeze(pil_to_tensor(im),0)
#         im_tensor = im_tensor[:,0,:,:].unsqueeze(0)
#         im_tensor = (im_tensor+0.01*torch.randn(im_tensor.size())).cuda()
        
#         point_list = []
#         with open('validation_ULM_points/point_list_{}.csv'.format(i+1)) as f:
#             rd = csv.reader(f)
#             for row in rd:
#                 if len(row)!=0:
#                     point_list.append([int(row[0]),int(row[1]),row[2]])
        
#         im_bif = torch.zeros([1,3,im_tensor.size(2),im_tensor.size(3)])
        
#         for j in range(len(point_list)):
#             if point_list[j][2] == 'endpoint':
#                 im_bif[0,0,point_list[j][0],point_list[j][1]] = 1
#                 point_list[j] = [0,point_list[j][0],point_list[j][1]]
#             if point_list[j][2] == 'biffurcation':
#                 im_bif[0,1,point_list[j][0],point_list[j][1]] = 1
#                 point_list[j] = [1,point_list[j][0],point_list[j][1]]
#             if point_list[j][2] == 'crossing':
#                 im_bif[0,2,point_list[j][0],point_list[j][1]] = 1
#                 point_list[j] = [2,point_list[j][0],point_list[j][1]]
                
#         im_bif_blur = gaussian_blur(im_bif).cuda()
#         im_bif_blur = im_bif_blur/(im_bif_blur.max())
        
#         net_output = net(im_tensor)
        
#         loss = l2loss(net_output,im_bif_blur)
#         validation_mean_loss +=loss/20
        
#         max_filtered_output = (local_max_filt(net_output)).double()
        
#         for index in range(len(a)):
#             im_points = ((max_filtered_output==net_output)*(net_output>(a[index]))).double().squeeze()
#             list_points_detected = im_points[:,:].nonzero()
            
#             nb_detected=0
            
#             point_array = torch.tensor(point_list)
            
#             for p in range(list_points_detected.size(0)):
#                 if ((list_points_detected[p,1:].cpu()-point_array[:,1:])**2).sum(dim=1).min()<pixel_tol**2: 
#                     nb_detected+=1
                    
#             list_precision[index] += min(len(list_points_detected),1)*nb_detected/(max(len(list_points_detected),1))/20
#             list_recall[index] += min(nb_detected/(max(len(point_list),1)),1)/20
#             list_F1[index] += 2*nb_detected/max(len(point_list) + len(list_points_detected),1)/20

            
#             # print(min(len(list_points_detected),1)*nb_detected/(max(len(list_points_detected),1)))

# plt.figure(100)
# plt.plot(list_precision,list_recall)

# plt.figure(101)
# plt.plot(a,list_F1)


#%%
for i in range(20):
    with torch.no_grad():        
        im = Image.open('validation_ULM/validation_ULM_{}.png'.format(i+1))
                
        point_list = []
        with open('validation_ULM_points/point_list_{}.csv'.format(i+1)) as f:
            rd = csv.reader(f)
            for row in rd:
                if len(row)!=0:
                    point_list.append([int(row[0]),int(row[1]),row[2]])
        
        im_tensor = torch.unsqueeze(pil_to_tensor(im),0)
        im_tensor = im_tensor[:,0,:,:].unsqueeze(0)
        im_tensor = (im_tensor+0.01*torch.randn(im_tensor.size())).cuda()
        
        im_bif = torch.zeros([1,3,im_tensor.size(2),im_tensor.size(3)])
        
        for j in range(len(point_list)):
            if point_list[j][2] == 'endpoint':
                im_bif[0,0,point_list[j][0],point_list[j][1]] = 1
                point_list[j] = [0,point_list[j][0],point_list[j][1]]
            if point_list[j][2] == 'biffurcation':
                im_bif[0,1,point_list[j][0],point_list[j][1]] = 1
                point_list[j] = [1,point_list[j][0],point_list[j][1]]
            if point_list[j][2] == 'crossing':
                im_bif[0,2,point_list[j][0],point_list[j][1]] = 1
                point_list[j] = [2,point_list[j][0],point_list[j][1]]
                
        im_bif_blur = gaussian_blur(im_bif).cuda()
        im_bif_blur = im_bif_blur/(im_bif_blur.max())
        
        point_array = np.array(point_list)
        
        net_output = net(im_tensor)
        
        loss = l2loss(net_output,im_bif_blur)
        
        max_filtered_output = (local_max_filt(net_output)).double()
        im_points = ((max_filtered_output==net_output)*(net_output>0.05)).double().squeeze()
        
        plt.figure(i+1)
        
        plt.imshow(im_tensor.cpu().squeeze())
        
        list_points_detected = im_points[:,:].nonzero()
        array_points_detected = np.array(list_points_detected.cpu())
        plt.scatter(point_array[:,2],point_array[:,1],c='white',marker='.', alpha=1)
        plt.scatter(array_points_detected[:,2],array_points_detected[:,1],c='r',marker='.', alpha=0.5)
        
        plt.savefig('ULM_image_and_label_found/image_and_points_found_{}.png'.format(i+1))
        plt.close(i+1)
        
        # plt.figure(i+1)
        # plt.imshow(((net_output).double()).cpu().squeeze().permute([1,2,0]))

#%%
training_mean_loss = 0

for i in range(20):
    with torch.no_grad():        
        im = Image.open('training_ULM/training_ULM_{}.png'.format(i+1))
                
        point_list = []
        with open('training_ULM_points/point_list_{}.csv'.format(i+1)) as f:
            rd = csv.reader(f)
            for row in rd:
                if len(row)!=0:
                    point_list.append([int(row[0]),int(row[1]),row[2]])
        
        im_tensor = torch.unsqueeze(pil_to_tensor(im),0)
        im_tensor = im_tensor[:,0,:,:].unsqueeze(0)
        im_tensor = (im_tensor+0.01*torch.randn(im_tensor.size())).cuda()
        
        im_bif = torch.zeros([1,3,im_tensor.size(2),im_tensor.size(3)])
        
        for j in range(len(point_list)):
            if point_list[j][2] == 'endpoint':
                im_bif[0,0,point_list[j][0],point_list[j][1]] = 1
                point_list[j] = [0,point_list[j][0],point_list[j][1]]
            if point_list[j][2] == 'biffurcation':
                im_bif[0,1,point_list[j][0],point_list[j][1]] = 1
                point_list[j] = [1,point_list[j][0],point_list[j][1]]
            if point_list[j][2] == 'crossing':
                im_bif[0,2,point_list[j][0],point_list[j][1]] = 1
                point_list[j] = [2,point_list[j][0],point_list[j][1]]
                
        im_bif_blur = gaussian_blur(im_bif).cuda()
        im_bif_blur = im_bif_blur/(im_bif_blur.max())
        
        point_array = np.array(point_list)
        
        net_output = net(im_tensor)
        
        loss = l2loss(net_output,im_bif_blur)
        training_mean_loss +=loss/20
        
        max_filtered_output = (local_max_filt(net_output)).double()
        im_points = ((max_filtered_output==net_output)*(net_output>0.05)).double().squeeze()
        
        plt.figure(i+1)
        
        plt.imshow(im_tensor.cpu().squeeze())
        
        list_points_detected = im_points[:,:].nonzero()
        array_points_detected = np.array(list_points_detected.cpu())
        plt.scatter(point_array[:,2], point_array[:,1], c='white', marker='.', alpha=0.5)
        plt.scatter(array_points_detected[:,2], array_points_detected[:,1], c='r', marker='.', alpha=0.5)
        
        plt.savefig('ULM_image_and_label_found_training/image_and_points_found_{}.png'.format(i+1))
        plt.close(i+1)
        
        # plt.figure(i+1)
        # plt.imshow(((net_output).double()).cpu().squeeze().permute([1,2,0]))
print(training_mean_loss)

#%%    FLipping

# import time

with torch.no_grad():
    for i in range(20):
        im = Image.open('validation_ULM/validation_ULM_{}.png'.format(i+1))
        
        # point_list = []
        # with open('training_ULM_points/point_list_{}.csv'.format(i+1)) as f:
        #     rd = csv.reader(f)
        #     for row in rd:
        #         if len(row)!=0:
        #             point_list.append([int(row[0]),int(row[1]),row[2]])
        
        im_tensor = torch.unsqueeze(pil_to_tensor(im),0)
        im_tensor = im_tensor[:,0,:,:].unsqueeze(0)
        im_tensor = (im_tensor+0.01*torch.randn(im_tensor.size())).cuda()
        
        
        im_flipped = torch.zeros([2*im_tensor.shape[2],2*im_tensor.shape[3]])
        im_flipped[0:512,0:512] = im_tensor.squeeze()
        im_flipped[512:1024,0:512] = im_tensor.squeeze().fliplr()
        im_flipped[0:512,512:1024] = im_tensor.squeeze().flipud()
        im_flipped[512:1024,512:1024] = im_tensor.squeeze().flipud().fliplr()
        
        # plt.imshow(im_flipped)
        
        
        im_flipped_cuda = im_flipped.cuda()
        
        output = net(im_flipped_cuda.unsqueeze(0).unsqueeze(0))
        
        # plt.imshow(output.detach().cpu().squeeze().permute([1,2,0]))
        
        points = ((local_max_filt(output)==output)*(output>0.05)).squeeze().nonzero().cpu()
        
        plt.figure(i+1)
        plt.imshow(im_flipped.cpu())
        plt.scatter(points[:,2], points[:,1], c='r', s=10)
        plt.savefig('validation_flipped/validation_flipped_{}.png'.format(i+1))
        plt.close(i+1)
        
        plt.pause(0.01)

#%% Padding

import torch.nn.functional as F

with torch.no_grad():
    for i in range(20):
        im = Image.open('validation_ULM/validation_ULM_{}.png'.format(i+1))
                
        point_list = []
        with open('validation_ULM_points/point_list_{}.csv'.format(i+1)) as f:
            rd = csv.reader(f)
            for row in rd:
                if len(row)!=0:
                    point_list.append([int(row[0]),int(row[1])])
                    
        point_array = np.array(point_list)
        
        im_tensor = torch.unsqueeze(pil_to_tensor(im),0)
        im_tensor = im_tensor[:,0,:,:].unsqueeze(0)
        im_tensor = (im_tensor+0.01*torch.randn(im_tensor.size())).cuda()
        
        pad_size = 16
        im_padded = F.pad(im_tensor,(pad_size,pad_size,pad_size,pad_size))
        
        # plt.imshow(im_padded.cpu().squeeze())
        
        output = net(im_tensor)
        output_padded = net(im_padded)
        
        # plt.imshow(output.detach().cpu().squeeze().permute([1,2,0]))
        # plt.pause(0.01)
        # plt.imshow(output_padded.detach().cpu().squeeze().permute([1,2,0]))
        
        points = ((local_max_filt(output)==output)*(output>0.05)).squeeze().nonzero().cpu()
        points_padded = ((local_max_filt(output_padded)==output_padded)*(output_padded>0.05)).squeeze().nonzero().cpu()
        
        list_points_detected = points_padded
        
        nb_detected = 0
        
        point_array = torch.tensor(point_list) + torch.tensor([[pad_size, pad_size]])
        
        for p in range(list_points_detected.size(0)):
            if ((list_points_detected[p,1:].cpu()-point_array[:,1:])**2).sum(dim=1).min()<pixel_tol**2: 
                nb_detected+=1
                
        precision += min(len(list_points_detected),1)*nb_detected/(max(len(list_points_detected),1))/20
        recall += nb_detected/(max(len(point_list),1))/20
        
        F1 += 2*nb_detected/max(len(point_list) + len(list_points_detected),1)/20
        
        # plt.figure(i+1)
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(im_tensor.cpu().squeeze())
        axs[0].scatter(points[:,2], points[:,1], c='r', s=10)
        axs[0].scatter(point_array[:,1]-pad_size, point_array[:,0]-pad_size, c='w', s=10, alpha=0.5)
        
        axs[1].imshow(im_padded.cpu().squeeze())
        axs[1].scatter(points_padded[:,2], points_padded[:,1], c='r', s=10)
        axs[1].scatter(point_array[:,1], point_array[:,0], c='w', s=10, alpha=0.5)
        
print('MEAN precision: {} '.format(precision))        
print('MEAN recall: {} '.format(recall))  
print('MEAN F1-score: {} '.format(F1))  

print('mean validation loss is {}'.format(validation_mean_loss))
