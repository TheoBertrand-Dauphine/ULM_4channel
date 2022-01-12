# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 10:47:45 2021

@author: theot
"""

import torch
import torch.nn as nn
# from torch.autograd import Variable
# import torch.nn.functional as F
import torchvision
from PIL import Image
# import re
import numpy as np
import matplotlib.pyplot as plt

from DRIVE_supervised_Unet.network_designs import Unet_for_ULM

import csv


pil_to_tensor = torchvision.transforms.ToTensor()

net = Unet_for_ULM().cuda()

net.load_state_dict(torch.load('model_multiple_channel_ULM_11_02'))

im_tensor = torch.unsqueeze(pil_to_tensor(Image.open('input_image.png')),0).cuda()

net_output = net(im_tensor)

# gaussian_blur = torchgeometry.image.gaussian.GaussianBlur((17,17),(1,1)).cuda()
local_max_filt = nn.MaxPool2d(21,stride=1,padding=10).cuda()

max_filtered_output = (local_max_filt(net_output)).double()
im_points = ((max_filtered_output==net_output)*(net_output>0.005)).double().squeeze()

# plt.imshow(net_output.cpu().squeeze().detach())
# plt.imshow(im_points.cpu().squeeze().detach())

list_points_detected = im_points[:,:].nonzero()

plt.figure(1)

plt.imshow(im_tensor.cpu().squeeze())
array_points_detected = np.array(list_points_detected.cpu())
plt.scatter(array_points_detected[:,1],array_points_detected[:,0],c='r',marker='.')

with open("output_point_list.csv","w") as f:
    wr = csv.writer(f)
    wr.writerows(array_points_detected)