import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchgeometry

import matplotlib.pyplot as plt
from skimage import io, transform
import pandas as pd
import numpy as np

from scipy.io import loadmat

try:
    from utils.transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image, ColorJitter, GlobalContrastNormalization, RandomAffine, RandomFlip
except:
    from transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image, ColorJitter, GlobalContrastNormalization, RandomAffine, RandomFlip

def show_landmarks(image, landmarks, classes, heat_map):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 1], landmarks[:, 0], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

class ULMDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        size = len([name for name in os.listdir(self.root_dir + '/images_ULM/') if os.path.isfile(self.root_dir + '/images_ULM/' + name)])
        return size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_folder = sorted([name for name in os.listdir(self.root_dir + '/images_ULM/') if os.path.isfile(self.root_dir + '/images_ULM/' + name)])
        img_name = os.path.join(self.root_dir, 'images_ULM', data_folder[idx])
        #img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)

        points_folder = sorted([name for name in os.listdir(self.root_dir + '/ULM_points/') if os.path.isfile(self.root_dir + '/ULM_points/' + name)])

        try: #Catching empty list of points exception
            landmarks_frame = pd.read_csv(self.root_dir + '/ULM_points/' + points_folder[idx], header=None)
            landmarks = landmarks_frame.iloc[:, :2]
            classes = landmarks_frame.iloc[:,2]

            landmarks.loc[classes == 'endpoint',2] = 0
            landmarks.loc[(classes == 'biffurcation') | (classes == 'bifurcation'),2 ] = 1
            landmarks.loc[classes == 'crossing',2] = 2

            landmarks_array = np.zeros([400,3]) # Put it in a fixed size array !!!!!!!!!!!!!!

            landmarks = np.array(landmarks)

            landmarks_array[:landmarks.shape[0],:] = landmarks
        except:
            landmarks_array = np.zeros([400,3])

        if image.ndim==3:
            image = image[:,:,0]
      
        sample = {'image': np.sqrt(image), 'landmarks': landmarks_array}

        if self.transform:
            sample = self.transform(sample)

        return sample

class IOSTARDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, no_endpoints=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.no_endpoints = no_endpoints

    def __len__(self):
        size = len([name for name in os.listdir(self.root_dir + '/images_IOSTAR/') if os.path.isfile(self.root_dir + '/images_IOSTAR/' + name)])
        return size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_folder = sorted([name for name in os.listdir(self.root_dir + '/images_IOSTAR/') if os.path.isfile(self.root_dir + '/images_IOSTAR/' + name)])
        img_name = os.path.join(self.root_dir, 'images_IOSTAR', data_folder[idx])
        #img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)

        points_folder = sorted([name for name in os.listdir(self.root_dir + '/IOSTAR_points/') if os.path.isfile(self.root_dir + '/IOSTAR_points/' + name)])

        landmarks_frame = loadmat(self.root_dir + '/IOSTAR_points/' + points_folder[idx])
        
        if self.no_endpoints:
            landmarks = np.vstack([np.hstack([landmarks_frame['BiffPos']-1.,np.ones([landmarks_frame['BiffPos'].shape[0],1])]),
                np.hstack([landmarks_frame['CrossPos']-1.,2.*np.ones([landmarks_frame['CrossPos'].shape[0],1])])])

        else:
            landmarks = np.vstack([np.hstack([landmarks_frame['EndpointPos']-1.,0*np.ones([landmarks_frame['EndpointPos'].shape[0],1])]),
                np.hstack([landmarks_frame['BiffPos']-1.,np.ones([landmarks_frame['BiffPos'].shape[0],1])]),
                np.hstack([landmarks_frame['CrossPos']-1.,2.*np.ones([landmarks_frame['CrossPos'].shape[0],1])])])

        if image.ndim==3:
            image = np.transpose(image,(2,0,1))

        landmarks_array = np.zeros([400,3])
        landmarks_array[:landmarks.shape[0],:] = landmarks
      
        sample = {'image': image, 'landmarks': landmarks_array}

        
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':

    IOSTAR_dataset = IOSTARDataset(root_dir='./data_IOSTAR/val_images')
    # ULM_dataset = ULMDataset(root_dir='./data/val_images')
    # crop = transforms.Compose([GlobalContrastNormalization(), ColorJitter()])
    # heat = HeatMap(s=5,alpha=3, out_channels=4)

    # composed = transforms.Compose([Rescale(256), GlobalContrastNormalization(), ColorJitter(2), HeatMap(s=15, alpha=2, out_channels = 3), ToTensor(), RandomAffine(360, 0.05)])
    composed = transforms.Compose([RandomFlip(p=.5), HeatMap(s=25, alpha=3, out_channels = 3), ToTensor()])

    fig = plt.figure(1)

    # sample = ULM_dataset[20]
    # scale = Rescale_image(sample['image'].shape[1:])

    # for i, trfrm in enumerate([scale, crop, heat, composed]):
    #     print(i)
    #     trfrm_sample = trfrm(sample)

    #     ax = plt.subplot(1,4, i+1)
    #     plt.tight_layout()
    #     ax.set_title(type(trfrm).__name__)
    #     ax.axis('off')

    #     if trfrm_sample['image'].ndim==3:
    #         img = trfrm_sample['image'].transpose((1,2,0))
    #     else:
    #         img = trfrm_sample['image']
        
    #     if trfrm==scale:
    #         plt.imshow(img)
    #         plt.scatter(trfrm_sample['landmarks'][:,1],trfrm_sample['landmarks'][:,0], s=10, marker='.', c='red')
    #     if trfrm==crop:
    #         plt.imshow((img-img.min())/(img.max()-img.min()))
    #         plt.scatter(trfrm_sample['landmarks'][:,1],trfrm_sample['landmarks'][:,0], s=10, marker='.', c='red')
    #     if trfrm == heat:
    #         print(trfrm_sample['heat_map'].shape)
    #         plt.imshow(trfrm_sample['heat_map'][:,:3,:,:].squeeze().permute([1,2,0]))
    #     elif trfrm == composed:
    #         plt.imshow(trfrm_sample['heat_map'].squeeze().permute([1,2,0]))

    # plt.show()
    n = 8

    # for i in range(start, start + n):
    sample = IOSTAR_dataset[n]

    trfrm_sample = composed(sample)

    if trfrm_sample['image'].ndim==3:
        img = trfrm_sample['image'].permute((1,2,0))
    else:
        img = trfrm_sample['image']

    # ax = plt.subplot(min(n,2),int(n/2), i-start + 1)
    # plt.tight_layout()
    # ax.set_title('image {}'.format(i))
    # ax.axis('off')

    
    # print(((img-img.min())/(img.max()-img.min())).unsqueeze(2).repeat([1,1,3]).shape)

    
    indic = torch.max(1.*(trfrm_sample['heat_map'].squeeze()>0.1).permute([1,2,0]), dim=2, keepdims=True).values
    print(indic.shape)
    print(((1.-indic)*((img-img.min())/(img.max()-img.min())).squeeze()).shape)
    print(trfrm_sample['heat_map'].shape)

    # plt.imshow((1.-1.*indic)*((img-img.min())/(img.max()-img.min())).squeeze().unsqueeze(2).repeat(1,1,3) + indic*trfrm_sample['heat_map'].squeeze().permute([1,2,0]))
    plt.imshow((1.-1.*indic)*((img-img.min())/(img.max()-img.min())).squeeze() + indic*trfrm_sample['heat_map'].squeeze().permute([1,2,0]))

    # plt.scatter(trfrm_sample['landmarks'][:,1],trfrm_sample['landmarks'][:,0], s=10, marker='.', c='red')
    plt.axis('off')
    plt.tight_layout()
    plt.show()