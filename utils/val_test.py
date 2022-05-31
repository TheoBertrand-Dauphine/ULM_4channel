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

from PIL import Image

import pandas as pd




try:
    from utils.transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image, ColorJitter, GlobalContrastNormalization, RandomAffine, Padding
except:
    from transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image, ColorJitter, GlobalContrastNormalization, RandomAffine, Padding


o_channel = 3

model = ULM_UNet(in_channels=1, init_features=48, threshold = 0.5, out_channels = o_channel)
# model.load_from_checkpoint(checkpoint_path = './weights_import/ulm_net_IOSTAR_epochs_2000_batch_1_out_channels_4_20_5.ckpt',in_channels=3, init_features=48, threshold = 0.05, out_channels = 4)
model.load_state_dict(torch.load('./weights/ulm_net_synthetic_epochs_2000_batch_1_out_channels_3_30_5.pt'))
# n=44

# # landmarks_stack = np.zeros([n,200,3])

# # heat_map_stack = torch.zeros([n,3,256,256])
# F1 = torch.tensor(0.)
# precision_cum = torch.tensor(0.)
# recall_cum = torch.tensor(0.)

pil_to_tensor = torchvision.transforms.ToTensor()
tensor_to_pil = torchvision.transforms.ToPILImage()




data_dir = './data_synthetic/'
# validation_dataset = ULMDataset(root_dir = data_dir + 'val_images', transform=transforms.Compose([Rescale(256), GlobalContrastNormalization(), HeatMap(s=9, alpha=3, out_channels = o_channel), ToTensor(), Padding(32)]))
# valloader = DataLoader(validation_dataset, batch_size=20, shuffle=False, num_workers=16)


## Testing on whole big image

# if data_dir=='./data_synthetic/':
#     I = Image.open('./data_synthetic/validation_synthetic_image.png')
#     im_tensor = pil_to_tensor(I)

#     im_padded = nn.functional.pad(im_tensor,(24,24,12,12))[0]
#     y = model(im_padded.unsqueeze(0).unsqueeze(0))





validation_dataset = ULMDataset(root_dir = data_dir + 'test_images', transform=transforms.Compose([GlobalContrastNormalization(), HeatMap(s=9, alpha=3, out_channels = o_channel), ToTensor(), Padding(32)]))
valloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=16)

model.eval()
with torch.no_grad():
    for val_batch_idx, val_batch in enumerate(valloader):
        print(val_batch_idx)
        val_out = model.validation_step(val_batch, val_batch_idx, log=False)

        y = model(val_batch['image'].unsqueeze(0))

        local_max_filt = nn.MaxPool2d(9, stride=1, padding=4)

        threshold = 0.1
        max_output = local_max_filt(y)
        detected_points = ((max_output==y)*(y>threshold)).nonzero()[:,1:]

        plt.imshow(val_batch['image'].squeeze(), cmap='gray')

        plt.scatter(detected_points[detected_points[:,0]==0,2], detected_points[detected_points[:,0]==0,1], c='r', alpha=0.7)
        plt.scatter(detected_points[detected_points[:,0]==1,2], detected_points[detected_points[:,0]==1,1], c='g', alpha=0.7)
        plt.scatter(detected_points[detected_points[:,0]==2,2], detected_points[detected_points[:,0]==2,1], c='b', alpha=0.7)
        plt.scatter(detected_points[detected_points[:,0]==3,2], detected_points[detected_points[:,0]==3,1], c='w', alpha=0.4)
        plt.show()       