"""
Code by Théo BERTRAND
Data provided by the LIB team (Thanks to O. Couture, V. Hingot and A. Chavignon)

This code takes a dataset of ULM vascular images as input for training and validation 
and attempts to predict the location of landmarks (endpoints, bifurcations and crossings) on those images.
The model we're using is a classical U-net architecture. 
The theoretical output is taken as a heat map resulting from the convolution of diracs 
at the location of desired landmarks with some gaussian kernel.
We then minimize the L^2 distance cost to our data. 
We also perform data augmentation (translations, rotations, shearing and scaling)
to avoid over-parametriztion of our model.

Data : We take highly resolved images of vascular brain grey-level images and cut 512x512 smaller images to make 
our dataset. The training dataset is composed of images taken from both halves of a first brain
image (40 images in total) and our validation data is composed of images taken similarly of 
another highly resolved brain image.
"""

# A few imports
import torch
import torch.nn as nn

import torchvision
from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

import csv

from network_designs import Unet_for_ULM_out4

import torchgeometry.image.gaussian

from random import sample

torch.manual_seed(0)


#%%
if __name__=='__main__':

    net = Unet_for_ULM_out4() # Loading model

    if torch.cuda.is_available():
        net.cuda() # Putting our model on CUDA device    
    # scaler = torch.cuda.amp.GradScaler() # This is used for using float precision and avoiding using too much memory
    
    pytorch_total_params = sum(p.numel() for p in net.parameters()) # Counting our parameters
    
    print('There are {} parameters in our CNN model'.format(pytorch_total_params))
    
 #%%   
    optimizer = torch.optim.Adam(net.parameters(), lr=1*1e-4) # Setting up optimizer and parameters to optimize

    num_steps = 2500 # Maximum number of steps for training phase
    
    batches = False
    
    if batches:
        batch_size = 4
    else:
        batch_size = 1
    
    patience = 600 # Number of steps between each validation step that will eventually diminish LR
    
    check_scores_timestep = 10 # Number of steps between each check of validation score
    
    pixel_tol = 7 # Size of the neighbour in pixels to say if a point is well detected
    
    pil_to_tensor = torchvision.transforms.ToTensor() # Turns images into torch tensors 
    
    gaussian_blur = torchgeometry.image.gaussian.GaussianBlur((17,17), (3,3)).cuda() # Setting up gaussian kernel
    local_max_filt = nn.MaxPool2d(21, stride=1, padding=10).cuda() # Max filter for local maximum detection
    random_affine = torchvision.transforms.RandomAffine(360, translate=(0.1,0.1)).cuda() # Random Affine transformations for data augmentation
    random_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5).cuda()
    
    random_transform = torchvision.transforms.Compose([random_affine,random_flip])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, min_lr=1e-8)
    
    former_validation_loss = 1e4
    
    scheduler.step(former_validation_loss) #initialize xith high validation loss
    
    # Setting up lists to record training
    training_loss_hist = []
    validation_loss_hist = []
    F1_hist = []
    precision_hist = []    
    recall_hist = []
    
    
    l2loss = nn.MSELoss(reduction='mean') # L^2 distance cost function
    
    sample_size = 5
    
#%% Storing images for training
    
    im_tensor_tot = torch.zeros([40,1,512,512])
    im_bif_tot = torch.zeros([40,3,512,512])
    
    for i in range(20): # Iterating through the training dataset
            with torch.no_grad():
                # Getting images and label...
                
                im = Image.open('training_ULM/training_ULM_{}.png'.format(i+1)) 
                
                point_list = []
                with open('training_ULM_points/point_list_{}.csv'.format(i+1)) as f:
                    rd = csv.reader(f)
                    for row in rd:
                        if len(row)!=0:
                            point_list.append([int(row[0]),int(row[1]),row[2]])
                
                im_tensor = torch.unsqueeze(pil_to_tensor(im),0).sqrt()
                im_tensor = (im_tensor[:,0,:,:].unsqueeze(0))
                
                
                im_bif = torch.zeros([1,3,im_tensor.size(2),im_tensor.size(3)])
                
                # Making heat map with different channels for diferent kinds of landmarks
                for j in range(len(point_list)):
                    if point_list[j][2] == 'endpoint':
                        im_bif[0,0,point_list[j][0],point_list[j][1]] = 1
                        point_list[j] = [0,point_list[j][0],point_list[j][1]]
                    if point_list[j][2] == 'biffurcation' or point_list[j][2] == 'bifurcation':
                        im_bif[0,1,point_list[j][0],point_list[j][1]] = 1
                        point_list[j] = [1,point_list[j][0],point_list[j][1]]
                    if point_list[j][2] == 'crossing':
                        im_bif[0,2,point_list[j][0],point_list[j][1]] = 1
                        point_list[j] = [2,point_list[j][0],point_list[j][1]]
                
                im_bif_blur = gaussian_blur(im_bif)
                if im_bif_blur.max()!=0:
                    im_bif_blur = im_bif_blur/(im_bif_blur.max())
                
                
                # Random transforms for data augmentation
                # transform_input = torch.cat([im_tensor,im_bif_blur],1)
                
                # transform_output = random_transform(transform_input)
                
                # im_bif_blur = transform_output[:,1:,:,:].cuda()
                # im_tensor = transform_output[:,0,:,:].unsqueeze(0)
                
                im_tensor_tot[i,0,:,:] = im_tensor
                im_bif_tot[i,:,:,:] = im_bif_blur
                
    im_bif_tot = torch.cat([im_bif_tot, im_bif_tot.max(dim=1,keepdim=True).values],dim=1)
    
#%% Storing images for validation
    
    im_tensor_tot_val = torch.zeros([20,1,512,512])
    im_bif_tot_val = torch.zeros([20,3,512,512])
    
    for i in range(20): # Iterating through the trainin dataset
            with torch.no_grad():
                # Getting images and label...
                
                im = Image.open('validation_ULM/validation_ULM_{}.png'.format(i+1)) 
                
                point_list = []
                with open('validation_ULM_points/point_list_{}.csv'.format(i+1)) as f:
                    rd = csv.reader(f)
                    for row in rd:
                        if len(row)!=0:
                            point_list.append([int(row[0]),int(row[1]),row[2]])
                
                im_tensor = torch.unsqueeze(pil_to_tensor(im),0).sqrt()
                im_tensor = (im_tensor[:,0,:,:].unsqueeze(0))
                
                
                im_bif = torch.zeros([1,3,im_tensor.size(2),im_tensor.size(3)])
                
                # Making heat map with different channels for diferent kinds of landmarks
                for j in range(len(point_list)):
                    if point_list[j][2] == 'endpoint':
                        im_bif[0,0,point_list[j][0],point_list[j][1]] = 1
                        point_list[j] = [0,point_list[j][0],point_list[j][1]]
                    if point_list[j][2] == 'biffurcation' or point_list[j][2] == 'bifurcation':
                        im_bif[0,1,point_list[j][0],point_list[j][1]] = 1
                        point_list[j] = [1,point_list[j][0],point_list[j][1]]
                    if point_list[j][2] == 'crossing':
                        im_bif[0,2,point_list[j][0],point_list[j][1]] = 1
                        point_list[j] = [2,point_list[j][0],point_list[j][1]]
                
                im_bif_blur = gaussian_blur(im_bif)
                if im_bif_blur.max()!=0:
                    im_bif_blur = im_bif_blur/(im_bif_blur.max())
                
                im_tensor_tot_val[i,0,:,:] = im_tensor
                im_bif_tot_val[i,:,:,:] = im_bif_blur
    
    # im_tensor_tot_val = im_tensor_tot_val.cuda() # loading data on CUDA
    im_bif_tot_val = torch.cat([im_bif_tot_val, im_bif_tot_val.max(dim=1,keepdim=True).values],dim=1)
#%%
    
    for step in range(num_steps):
        loss_mean = 0
        
        for i in range(round(20/batch_size)):
            if batches:
                batch_indices = sample(range(20), batch_size)
            else:
                batch_indices = i
            
            im_tensor = im_tensor_tot[batch_indices,:,:,:].unsqueeze(0)
            im_bif_blur = im_bif_tot[batch_indices,:,:,:].unsqueeze(0)
            
            im_tensor = (im_tensor + 0.01*torch.randn(im_tensor.size()))
            
            # Random transforms for data augmentation
            transform_input = torch.cat([im_tensor,im_bif_blur],1)
            
            transform_output = random_transform(transform_input.cuda())
            
            im_bif_blur = transform_output[:,1:,:,:]
            im_tensor = transform_output[:,0,:,:].unsqueeze(1)
    
            
            net_output = net(im_tensor) # Applying our model
            
            with torch.cuda.amp.autocast():
                # Back propagating
                optimizer.zero_grad() # zero-ing the gradient
    
                # loss = l2loss(net_output,im_bif_blur)   
                loss = l2loss(net_output[:,:4,:,:],im_bif_blur[:,:4,:,:])
                loss_mean += loss/40 # We compute the mean loss for output
                
            # scaler.scale(loss).backward()
            # scaler.step(optimizer) # Optimization step
            
            # scaler.update()
            loss.backward()
            optimizer.step()
            
        if step==0:
            first_loss = loss_mean.clone().detach().cpu()

            
        print('At step {}, mean learning loss is {}'.format(step+1,loss_mean.data.cpu()/first_loss))
        
        

            
        with torch.no_grad():
            im_tensor = im_tensor_tot_val[sample(range(20),sample_size),:,:,:].cuda()
            
            net_output = net(im_tensor)
            
            validation_mean_loss = l2loss(net_output[:,:3,:,:],im_bif_blur[:,:3,:,:])
            
                
            scheduler.step(validation_mean_loss)
            
            if ((step+1)%check_scores_timestep)==0:
                validation_mean_loss = 0
                
                precision = 0
                recall = 0
                F1 = 0        
                
                
                for i in range(20):
                    im_tensor = im_tensor_tot_val[i,:,:,:].cuda().unsqueeze(0)
                    net_output = net(im_tensor)
                    max_filtered_output = (local_max_filt(net(im_tensor))).double()
                    
                    im_points = ((max_filtered_output==net_output)*(net_output>0.05)).double()[:,:3,:,:].squeeze() # Finding local maxima
                    list_points_detected = im_points[:,:].nonzero() # Recording their coordinates
                    nb_detected = 0
                    
                    point_list = []
                    with open('validation_ULM_points/point_list_{}.csv'.format(i+1)) as f:
                        rd = csv.reader(f)
                        for row in rd:
                            if len(row)!=0:
                                point_list.append([int(row[0]),int(row[1]),row[2]])
                                
                    for j in range(len(point_list)):
                        if point_list[j][2] == 'endpoint':
                            point_list[j] = [0,point_list[j][0],point_list[j][1]]
                        if point_list[j][2] == 'biffurcation' or point_list[j][2] == 'bifurcation':
                            point_list[j] = [1,point_list[j][0],point_list[j][1]]
                        if point_list[j][2] == 'crossing':
                            point_list[j] = [2,point_list[j][0],point_list[j][1]]

                    point_array = torch.tensor(point_list)
                    
                    marker_reached_point_in_GT = torch.zeros(point_array.shape[0])
                    marker_valid_point_in_found = torch.zeros(list_points_detected.shape[0])
                    
                    for p in range(list_points_detected.size(0)):
                        distance_vector_to_GT = ((list_points_detected[p,1:].cpu()-point_array[:,1:])**2).sum(dim=1)
                        if distance_vector_to_GT.min()<pixel_tol**2: 
                            marker_reached_point_in_GT[distance_vector_to_GT==distance_vector_to_GT.min()] = 1
                            marker_valid_point_in_found[p] = 1
                    
                    # Computing Precsion, recall and F1 scores
                    recall += marker_reached_point_in_GT.mean()/20        
                    precision += marker_valid_point_in_found.mean()/20
                    F1 += (2/((1/marker_reached_point_in_GT.mean()) + (1.0/marker_valid_point_in_found.mean())))/20
                    
                if F1>0.85:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10
                    print('Nice job !')                
                
                
                print('MEAN precision: {} '.format(precision))  
                print('MEAN recall: {} '.format(recall))  
                print('MEAN F1-score: {} '.format(F1))
                
            
            
            # training_loss_hist.append(loss_mean)
            # validation_loss_hist.append(validation_mean_loss)
            
            # F1_hist.append(F1)
            # precision_hist.append(precision)
            # recall_hist.append(recall)
            

            
            print('At step TEST {}, mean validation loss is {}'.format(step+1,validation_mean_loss))
            print('learning rate is {}'.format(optimizer.param_groups[0]['lr']))
    
            
            if optimizer.param_groups[0]['lr'] < 1*1e-7:
                print('exited because learning rate is too low')
                break
            

#%%

# Saving the images given by our model on validation dataset

for i in range(20):
    with torch.no_grad():        
        im = Image.open('validation_ULM/validation_ULM_{}.png'.format(i+1))
                
        point_list = []
        with open('validation_ULM_points/point_list_{}.csv'.format(i+1)) as f:
            rd = csv.reader(f)
            for row in rd:
                if len(row)!=0:
                    point_list.append([int(row[0]),int(row[1]),row[2]])
        
        im_tensor = torch.unsqueeze(pil_to_tensor(im),0).cuda()
        im_tensor = im_tensor[:,0,:,:].unsqueeze(0)
        
        im_bif = torch.zeros([1,3,im_tensor.size(2),im_tensor.size(3)])
        
        for j in range(len(point_list)):
            if point_list[j][2] == 'endpoint':
                im_bif[0,0,point_list[j][0],point_list[j][1]] = 1
                point_list[j] = [0,point_list[j][0],point_list[j][1]]
            if point_list[j][2] == 'biffurcation' or point_list[j][2] == 'bifurcation':
                im_bif[0,1,point_list[j][0],point_list[j][1]] = 1
                point_list[j] = [1,point_list[j][0],point_list[j][1]]
            if point_list[j][2] == 'crossing':
                im_bif[0,2,point_list[j][0],point_list[j][1]] = 1
                point_list[j] = [2,point_list[j][0],point_list[j][1]]
                
        im_bif_blur = gaussian_blur(im_bif).cuda()
        im_bif_blur = im_bif_blur/(im_bif_blur.max())
        
        im_bif_blur = torch.cat([im_bif_blur, im_bif_blur.max(dim=1,keepdim=True).values],dim=1)
        
        point_array = np.array(point_list)
        
        net_output = net(im_tensor)
        
        loss = l2loss(net_output,im_bif_blur)
        
        max_filtered_output = (local_max_filt(net_output)).double()
        im_points = ((max_filtered_output==net_output)*(net_output>0.05)).double().squeeze()
        
        plt.figure(i+1)
        
        plt.imshow(im_tensor.cpu().squeeze())
        
        list_points_detected = im_points[:,:].nonzero()
        array_points_detected = np.array(list_points_detected.cpu())
        
        plt.scatter(array_points_detected[:,2],array_points_detected[:,1],c='r',marker='.')
        plt.scatter(point_array[:,2],point_array[:,1],c='white',marker='.', alpha = 0.5)
        plt.savefig('ULM_image_and_label_found/image_and_points_found_{}.png'.format(i+1))
        plt.close(i+1)
        
        
#%%
# for i in range(40):
#     with torch.no_grad():        
#         im = Image.open('training_synthetic/training_synthetic_{}.png'.format(i+1))
                
#         point_list = []
#         with open('training_synthetic_points/point_list_{}.csv'.format(i+1)) as f:
#             rd = csv.reader(f)
#             for row in rd:
#                 if len(row)!=0:
#                     point_list.append([int(row[0]),int(row[1]),row[2]])
        
#         im_tensor = torch.unsqueeze(pil_to_tensor(im),0).cuda()
        
#         im_bif = torch.zeros([1,3,im_tensor.size(2),im_tensor.size(3)])
        
#         for j in range(len(point_list)):
#             if point_list[j][2] == 'endpoint':
#                 im_bif[0,0,point_list[j][0],point_list[j][1]] = 1
#             if point_list[j][2] == 'biffurcation':
#                 im_bif[0,1,point_list[j][0],point_list[j][1]] = 1
#             if point_list[j][2] == 'crossing':
#                 im_bif[0,2,point_list[j][0],point_list[j][1]] = 1
        
#         im_bif_blur = gaussian_blur(im_bif).cuda()
#         im_bif_blur = im_bif_blur/(im_bif_blur.max())
        
#         im_tensor = torch.unsqueeze(pil_to_tensor(im),0).cuda()
        
#         net_output = net(im_tensor)
        
#         loss = l2loss(net_output,im_bif_blur)
        
#         max_filtered_output = (local_max_filt(net_output)).double()
#         im_points = ((max_filtered_output==net_output)*(net_output>0.01)).double().squeeze()
        
#         plt.figure(i+1)
#         plt.imshow(im_tensor.cpu().squeeze())
#         list_points_detected = im_points[:,:].nonzero()
#         array_points_detected = np.array(list_points_detected.cpu())
#         plt.scatter(array_points_detected[:,2],array_points_detected[:,1],c='r',marker='.')
#         plt.scatter(point_array[:,2],point_array[:,1],c='g',marker='.')
#         plt.savefig('synthetic_image_and_label_found_training/image_and_points_found_training_{}.png'.format(i+1))
#         plt.close(i+1)

#%% Saving our model
import time 
todaystr = time.strftime("%d_%m")
torch.save(net.state_dict(),'./model_state_dict_3chan' + todaystr + ".pt")
