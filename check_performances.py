
#%%
import numpy as np

# from PIL import Image

import torch
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt

from utils.OrientationScore import gaussian_OS
from utils.dataset import ULMDataset, IOSTARDataset

from nn.ulm_unet import ULM_UNet

from tqdm import tqdm

from datetime import datetime

from skimage.filters import frangi

try:
    from utils.transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image, ColorJitter, GlobalContrastNormalization, RandomAffine, Padding, CenterCrop, ToArray
except:
    from transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image, ColorJitter, GlobalContrastNormalization, RandomAffine, Padding, CenterCrop, ToArray
    
import networkx as nx

import sys
sys.path.append("./../ULM_data")

from make_ulm_images import making_ULM_halfleft_rat_brain2D_and_orientation, making_ULM_bolus_full_rat_brain2D_and_orientation

validation_dataset = ULMDataset(root_dir =  './data/test_images', transform=transforms.Compose([ RandomCrop(512), HeatMap(s=9, alpha=3, out_channels = 4), ToTensor(), Padding(32)])) 
batch = validation_dataset[1]


model = ULM_UNet(in_channels=1, init_features=48, threshold = 0.1, out_channels = 3)
model.load_state_dict(torch.load('./weights/ulm_net_ULM_epochs_1500_batch_1_out_channels_3_7_6.pt'))
Nt = 64

if not('p' in locals()):
    p = making_ULM_bolus_full_rat_brain2D_and_orientation(N_t = Nt, scaling_effect = 0.4)


output = model(batch['image'].unsqueeze(0).unsqueeze(0)).squeeze()

local_max_filt = torch.nn.MaxPool2d(17, stride=1, padding=8)

#%%
max_output = local_max_filt(output)
detected_points = ((max_output==output)*(output>.05)).nonzero()[:,1:]
    

plt.figure()
plt.imshow((output.permute([1,2,0]).detach()>.05)+0.)

plt.figure()
plt.imshow(batch['heat_map'].squeeze().permute([1,2,0])[:,:,:3].detach())

plt.figure()
plt.imshow(batch['image'].detach(), cmap='gray')
plt.scatter(batch['landmarks'][:,1], batch['landmarks'][:,0], c = 'r', s = 1)
plt.scatter(detected_points[:,1], detected_points[:,0], c = 'b', s = 1)

# %%
