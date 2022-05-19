

from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from skimage import io, transform

from scipy.io import loadmat


import torchgeometry
import matplotlib.pyplot as plt



try:
    from utils.transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image, ColorJitter, GlobalContrastNormalization, RandomAffine
except:
    from transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image, ColorJitter, GlobalContrastNormalization, RandomAffine

n=88

landmarks_stack = np.zeros([n,200,3])

heat_map_stack = torch.zeros([n,3,256,256])

for i in range(n):
    # print(i)

    img_name = './data_IOSTAR/val_images/images_IOSTAR/validation_IOSTAR_{}.png'.format(i+1)
    image = io.imread(img_name)

    # points_folder = sorted([name for name in os.listdir(self.root_dir + '/IOSTAR_points/') if os.path.isfile(self.root_dir + '/IOSTAR_points/' + name)])

    landmarks_frame = loadmat('./data_IOSTAR/val_images/IOSTAR_points/IOSTAR_points_{}.mat'.format(i+1))

    landmarks = np.vstack([np.hstack([landmarks_frame['EndpointPos']-1.,0*np.ones([landmarks_frame['EndpointPos'].shape[0],1])]),
        np.hstack([landmarks_frame['BiffPos']-1.,np.ones([landmarks_frame['BiffPos'].shape[0],1])]),
        np.hstack([landmarks_frame['CrossPos']-1.,2.*np.ones([landmarks_frame['CrossPos'].shape[0],1])])])

    # landmarks = np.vstack([np.hstack([landmarks_frame['BiffPos']-1.,np.ones([landmarks_frame['BiffPos'].shape[0],1])]),
        # np.hstack([landmarks_frame['CrossPos']-1.,2.*np.ones([landmarks_frame['CrossPos'].shape[0],1])])])



    # classes = np.empty(shape = (landmarks.shape[0],),dtype = "S12")
    # classes[landmarks[:,2]==0.] = 'endpoint'
    # classes[landmarks[:,2]==1.] = 'biffurcation'
    # classes[landmarks[:,2]==2.] = 'crossing'

    # if image.ndim==3:
    #     image = np.transpose(image,(2,0,1))

    landmarks_array = np.zeros([200,3])
    landmarks_array[:landmarks.shape[0],:] = landmarks
    landmarks_stack[i,:,:] = landmarks_array

    # print(landmarks_array)

    gaussian_blur = torchgeometry.image.gaussian.GaussianBlur((9,9), (3,3))

    heat_map = torch.zeros(1, 3, image.shape[0], image.shape[1])

    heat_map[0,landmarks_array[landmarks_array[:,1]**2+landmarks_array[:,0]**2 > 0,2].astype(int),landmarks_array[landmarks_array[:,1]**2+landmarks_array[:,0]**2 > 0,0].astype(int),landmarks_array[landmarks_array[:,1]**2+landmarks_array[:,0]**2 > 0,1].astype(int)] = 1.

    heat_map = gaussian_blur(heat_map)
    heat_map = heat_map / heat_map.max()

    heat_map_stack[i,:,:,:] = heat_map

y_hat = heat_map_stack
local_max_filt = nn.MaxPool2d(9, stride=1, padding=4)

threshold = 0.05
dist_tol = 5

max_output = local_max_filt(y_hat)
detected_points = ((max_output==y_hat)*(y_hat>threshold)).nonzero()

# ax = plt.subplot(1,3,1) 
# plt.imshow(max_output.squeeze().permute([1,2,0]))

# ax = plt.subplot(1,3,2) 
# plt.imshow(y_hat.squeeze().permute([1,2,0]))

# ax = plt.subplot(1,3,3) 
# plt.imshow(((max_output==y_hat)*(y_hat>threshold)).double().squeeze().permute([1,2,0]))

# plt.show()

# plt.pause(0.1)
# plt.close()

points_coordinates = torch.tensor(landmarks_stack)

print(points_coordinates.shape)
print(landmarks_stack.shape)

nb_points = ((points_coordinates**2).sum(dim=-1) > 0).sum(dim=-1)

# print(nb_points)
# print(landmarks.shape[0],nb_points)

F1 = torch.tensor(0.)
precision_cum = torch.tensor(0.)
recall_cum = torch.tensor(0.)

avg_points_detected = detected_points.shape[0]



for i in range(n-4):
    points = detected_points[detected_points[:,0]==i,1:][:,[1,2,0]]

    # print(points.shape)
    # print(points_coordinates[i].shape)
    distance = ((torch.tensor([[[1, 1, dist_tol]]])*(points.unsqueeze(1) - points_coordinates[i,:nb_points[i],:].unsqueeze(0)))**2).sum(dim=-1)

    # print(distance.shape)


    # plt.imshow(distance==distance.min(dim=1, keepdim=True).values)
    # plt.show()

    distance_min = distance*(distance==distance.min(dim=1, keepdim=True).values) + 1e8*(distance!=distance.min(dim=1, keepdim=True).values)
    # distance_min[distance!=distance.min(dim=1, keepdim=True).values]=1e8

    found_matrix = distance_min < dist_tol**2

    # plt.imshow(found_matrix)
    # plt.show()

    TP = found_matrix.max(dim=1).values.sum()

    # print(TP,points.shape[0],nb_points,found_matrix.shape[1])

    precision = TP/max(points.shape[0],1) #nb of points well classified/nb of points in the class
    recall = TP/max(nb_points[i],1) #nb of points well classified/nb of points labeled in the class

    print(nb_points[i],TP)
    print(points)
    print(points_coordinates[i,:nb_points[i],:])
    recall_cum += recall/n
    precision_cum += precision/n

    if precision!=0 and recall!=0:
        F1 += 2/((1/recall)+(1/precision))/n

plt.figure(0)
plt.scatter(points[:,1],points[:,0], c='r', alpha=0.5)
plt.scatter(points_coordinates[i,:nb_points[i],1],points_coordinates[i,:nb_points[i],0], c='b', alpha=0.5)

plt.figure(1)
plt.imshow(distance_min)

plt.show()
print('cumulated value F1 {}, precision {}, recall {}'.format(F1,recall_cum,precision_cum))