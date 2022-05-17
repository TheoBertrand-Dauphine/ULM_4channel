

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

for i in range(min(n,1)):

    print(i)

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

    # print(landmarks_array)

    gaussian_blur = torchgeometry.image.gaussian.GaussianBlur((9,9), (3,3))

    heat_map = torch.zeros(1, 3, image.shape[0], image.shape[1])

    heat_map[0,landmarks_array[landmarks_array[:,1]**2+landmarks_array[:,0]**2 > 0,2].astype(int),landmarks_array[landmarks_array[:,1]**2+landmarks_array[:,0]**2 > 0,0].astype(int),landmarks_array[landmarks_array[:,1]**2+landmarks_array[:,0]**2 > 0,1].astype(int)] = 1.

    heat_map = gaussian_blur(heat_map)
    heat_map = heat_map / heat_map.max()

    y_hat = heat_map
    local_max_filt = nn.MaxPool2d(5, stride=1, padding=2)

    threshold = 0.05
    dist_tol = 7

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

    points_coordinates = torch.tensor(landmarks_array)

    nb_points = ((points_coordinates[:,:2]**2).sum(dim=-1) > 0).sum()
    # print(landmarks.shape[0],nb_points)

    F1 = torch.tensor(0.)
    precision_cum = torch.tensor(0.)
    recall_cum = torch.tensor(0.)

    avg_points_detected = detected_points.shape[0]

    points = detected_points[:,1:][:,[1,2,0]]

    distance = ((torch.tensor([[[1, 1, dist_tol]]])*(points.unsqueeze(1) - points_coordinates[:nb_points,:].unsqueeze(0)))**2).sum(dim=-1)


    # plt.imshow(distance==distance.min(dim=1, keepdim=True).values)
    # plt.show()

    distance_min = distance*(distance==distance.min(dim=1, keepdim=True).values)
    distance_min[distance!=distance.min(dim=1, keepdim=True).values]=1e8

    found_matrix = distance_min < dist_tol**2

    # plt.imshow(found_matrix)
    # plt.show()

    TP = found_matrix.max(dim=1).values.sum()

    print(TP,points.shape[0],nb_points,found_matrix.shape[1])

    precision = TP/max(found_matrix.shape[1],1) #nb of points well classified/nb of points in the class
    recall = TP/max(nb_points,1) #nb of points well classified/nb of points labeled in the class

    print(precision, recall, 2/((1/recall)+(1/precision)))

    recall_cum += recall
    precision_cum += precision

    if precision!=0 and recall!=0:
        F1 += 2/((1/recall)+(1/precision))