from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from skimage import io, transform

from torchvision import transforms

import torchvision

from scipy.io import loadmat


import torchgeometry
import matplotlib.pyplot as plt


import sys

sys.path.insert(0,'./nn')
# from .nn.ulm_unet import ULM_UNet
from ulm_unet import ULM_UNet

from dataset import ULMDataset, IOSTARDataset
from torch.utils.data import DataLoader

import pytorch_lightning
from pytorch_lightning import Trainer



try:
    from utils.transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image, ColorJitter, GlobalContrastNormalization, RandomAffine
except:
    from transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image, ColorJitter, GlobalContrastNormalization, RandomAffine

model = ULM_UNet(in_channels=3, init_features=48, threshold = 0.05, out_channels = 4)
model.load_from_checkpoint(checkpoint_path = './weights_import/ulm_net_IOSTAR_epochs_2000_batch_1_out_channels_4_20_5.ckpt',in_channels=3, init_features=48, threshold = 0.05, out_channels = 4)

# n=44

# # landmarks_stack = np.zeros([n,200,3])

# # heat_map_stack = torch.zeros([n,3,256,256])
# F1 = torch.tensor(0.)
# precision_cum = torch.tensor(0.)
# recall_cum = torch.tensor(0.)

# pil_to_tensor = torchvision.transforms.ToTensor()
# tensor_to_pil = torchvision.transforms.ToPILImage()

# trfrm=transforms.Compose([Rescale(512), GlobalContrastNormalization(), HeatMap(s=9, alpha=3, out_channels = 4), ToTensor()])

# for i in range(min(1,n)):

#     img_name = './data_IOSTAR/val_images/images_IOSTAR/validation_IOSTAR_{}.png'.format(i+1)
#     image = io.imread(img_name)
#     # im_tensor = pil_to_tensor(image)

#     # plt.imshow(image.permute([1,2,0]))
#     # plt.show()
#     # print(image)
#     # points_folder = sorted([name for name in os.listdir(self.root_dir + '/IOSTAR_points/') if os.path.isfile(self.root_dir + '/IOSTAR_points/' + name)])

#     landmarks_frame = loadmat('./data_IOSTAR/val_images/IOSTAR_points/IOSTAR_points_{}.mat'.format(i+1))

#     # landmarks = np.vstack([np.hstack([landmarks_frame['EndpointPos']-1.,0*np.ones([landmarks_frame['EndpointPos'].shape[0],1])]),
#     #     np.hstack([landmarks_frame['BiffPos']-1.,np.ones([landmarks_frame['BiffPos'].shape[0],1])]),
#     #     np.hstack([landmarks_frame['CrossPos']-1.,2.*np.ones([landmarks_frame['CrossPos'].shape[0],1])])])

#     landmarks = np.vstack([np.hstack([landmarks_frame['BiffPos']-1.,np.ones([landmarks_frame['BiffPos'].shape[0],1])]),
#         np.hstack([landmarks_frame['CrossPos']-1.,2.*np.ones([landmarks_frame['CrossPos'].shape[0],1])])])

#     classes = np.empty(shape = (landmarks.shape[0],),dtype = "S12")
#     classes[landmarks[:,2]==0.] = 'endpoint'
#     classes[landmarks[:,2]==1.] = 'biffurcation'
#     classes[landmarks[:,2]==2.] = 'crossing'

#     # if image.ndim==3:
#     #     image = np.transpose(image,(2,0,1))

#     landmarks_array = np.zeros([400,3])
#     landmarks_array[:landmarks.shape[0],:] = landmarks
#     # landmarks_stack[i,:,:] = landmarks_array

#     # print(np.array(image).transpose((2,0,1)).shape)

#     sample = {'image': np.array(image).transpose((2,0,1)), 'classes': classes, 'landmarks': landmarks_array}

#     trfrm_sample = trfrm(sample)

#     plt.imshow(trfrm_sample['image'].permute([1,2,0]))
#     plt.show()

#     y = model(trfrm_sample['image'].unsqueeze(0)).squeeze()

    
#     print(y.shape)

#     plt.imshow(y[:3,:,:].detach().squeeze().permute([1,2,0]))
#     plt.show()

#     # print(landmarks_array)

#     gaussian_blur = torchgeometry.image.gaussian.GaussianBlur((9,9), (3,3))

#     heat_map = torch.zeros(1, 4, image.shape[0], image.shape[1])

#     print(heat_map.shape)

#     heat_map[0,landmarks_array[landmarks_array[:,1]**2+landmarks_array[:,0]**2 > 0,2].astype(int),landmarks_array[landmarks_array[:,1]**2+landmarks_array[:,0]**2 > 0,0].astype(int),landmarks_array[landmarks_array[:,1]**2+landmarks_array[:,0]**2 > 0,1].astype(int)] = 1.

#     heat_map = gaussian_blur(heat_map)
#     heat_map = heat_map / heat_map.max()
#     heat_map[0,3,:,:] = heat_map[0,:3,:,:].max(dim=0).values

#     print(((y-heat_map)**2).mean())

#     # heat_map_stack[i,:,:,:] = heat_map

#     y_hat = heat_map
#     local_max_filt = nn.MaxPool2d(17, stride=1, padding=8)

#     threshold = 0.05
#     dist_tol = 7

#     max_output = local_max_filt(y.unsqueeze(0))
#     detected_points = ((max_output==y.unsqueeze(0))*(y>threshold)).nonzero()

#     # print(detected_points)

#     # ax = plt.subplot(1,3,1) 
#     # plt.imshow(max_output.squeeze().permute([1,2,0]))

#     # ax = plt.subplot(1,3,2) 
#     # plt.imshow(y_hat.squeeze().permute([1,2,0]))

#     # ax = plt.subplot(1,3,3) 
#     # plt.imshow(((max_output==y_hat)*(y_hat>threshold)).double().squeeze().permute([1,2,0]))

#     # plt.show()

#     # plt.pause(0.1)
#     # plt.close()

#     points_coordinates = torch.tensor(landmarks_array)

#     # print(points_coordinates.shape)
#     # print(landmarks_stack.shape)

#     nb_points = ((points_coordinates**2).sum(dim=-1) > 0).sum(dim=-1)

#     # print(nb_points)
#     # print(landmarks.shape[0],nb_points)

#     avg_points_detected = detected_points.shape[0]



#     points = detected_points[:,1:][:,[1,2,0]]

#     if (points[:,2]==3).sum()!=0:
#         points = points[(points[:,2]!=3),:]

#     # print(((points_coordinates[:nb_points])))
#     # print(points_coordinates[i].shape)
#     distance = ((torch.tensor([[[1, 1, dist_tol]]])*(points.unsqueeze(1) - points_coordinates[:nb_points,:].unsqueeze(0)))**2).sum(dim=-1)

#     # print(distance.shape)


#     # plt.imshow(distance==distance.min(dim=1, keepdim=True).values)
#     # plt.show()

#     distance_min = distance*(distance==distance.min(dim=1, keepdim=True).values) + 1e8*(distance!=distance.min(dim=1, keepdim=True).values)
#     # distance_min[distance!=distance.min(dim=1, keepdim=True).values]=1e8

#     found_matrix = distance_min < dist_tol**2

#     # plt.imshow(found_matrix)
#     # plt.show()

#     TP = found_matrix.max(dim=1).values.sum()

#     # print(TP,points.shape[0],nb_points,found_matrix.shape[1])

#     precision = TP/max(points.shape[0],1) #nb of points well classified/nb of points in the class
#     recall = TP/max(nb_points,1) #nb of points well classified/nb of points labeled in the class

#     print(nb_points,TP)
#     # print(points)
#     # print(points_coordinates[:nb_points,:])
#     recall_cum += recall/n
#     precision_cum += precision/n

#     print(recall, precision, 2/((1/recall)+(1/precision)))
#     if precision!=0 and recall!=0:
#         F1 += 2/((1/recall)+(1/precision))/n

# plt.figure(0)
# plt.scatter(points[:,1],points[:,0], c='r', alpha=0.5)
# plt.scatter(points_coordinates[:nb_points,1],points_coordinates[:nb_points,0], c='b', alpha=0.5)

# plt.figure(2)

# plt.imshow(heat_map.squeeze().permute([1,2,0]))

# plt.figure(3)
# plt.imshow(((max_output==y.unsqueeze(0))*(y>threshold)).squeeze().double().detach()[:3].permute([1,2,0]))
# plt.show()
# print('cumulated value F1 {}, precision {}, recall {}'.format(F1,recall_cum,precision_cum))



data_dir = './data_IOSTAR/'
validation_dataset = IOSTARDataset(root_dir = data_dir + 'val_images', transform=transforms.Compose([Rescale(512), GlobalContrastNormalization(), HeatMap(s=9, alpha=3, out_channels = 4), ToTensor()]))
valloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=8)


# sample = validation_dataset[0]

# print(sample['image'].shape)
# plt.imshow(sample['image'].permute([1,2,0]))

# plt.show()
# model.eval()

# q = model.validation_step(valloader[0:2], [0,1], log=False)

model.eval()
with torch.no_grad():
    for val_batch_idx, val_batch in enumerate(valloader):
        print(val_batch_idx)
        val_out = model.validation_step(val_batch, val_batch_idx, log=False)