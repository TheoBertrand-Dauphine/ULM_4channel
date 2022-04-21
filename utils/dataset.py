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
    from utils.transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image
except:
    from transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image


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

        landmarks_frame = pd.read_csv(self.root_dir + '/ULM_points/' + points_folder[idx], header=None)
        landmarks = landmarks_frame.iloc[:, :2]
        classes = landmarks_frame.iloc[:,2]

        landmarks.loc[classes == 'endpoint',2] = 0
        landmarks.loc[(classes == 'biffurcation') | (classes == 'bifurcation'),2 ] = 1
        landmarks.loc[classes == 'crossing',2] = 2

        landmarks_array = np.zeros([40,3]) # Put it in a fixed size array !!!!!!!!!!!!!!

        landmarks = np.array(landmarks)
        # print(landmarks)

        # print(idx)

        # print(data_folder)
        # print(points_folder)
        # plt.imshow(image)
        # plt.scatter(landmarks[:, 1], landmarks[:, 0], s=10, marker='.', c='blue')
        # plt.show()

        landmarks_array[:landmarks.shape[0],:] = landmarks

        # print(landmarks_array)

        if image.ndim==3:
            image = image[:,:,0]
      
        sample = {'image': np.sqrt(image), 'classes': classes, 'landmarks': landmarks_array}

        
        if self.transform:
            sample = self.transform(sample)
        return sample

class IOSTARDataset(Dataset):
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

        landmarks = np.vstack([np.hstack([landmarks_frame['EndpointPos']-1.,0*np.ones([landmarks_frame['EndpointPos'].shape[0],1])]),
            np.hstack([landmarks_frame['BiffPos']-1.,np.ones([landmarks_frame['BiffPos'].shape[0],1])]),
            np.hstack([landmarks_frame['CrossPos']-1.,2.*np.ones([landmarks_frame['CrossPos'].shape[0],1])])])


        classes = np.empty(shape = (landmarks.shape[0],),dtype = "S12")
        classes[landmarks[:,2]==0.] = 'endpoint'
        classes[landmarks[:,2]==1.] = 'biffurcation'
        classes[landmarks[:,2]==2.] = 'crossing'

        if image.ndim==3:
            image = image[:,:,1]

        landmarks_array = np.zeros([100,3])
        landmarks_array[:landmarks.shape[0],:] = landmarks
      
        sample = {'image': image.astype(np.float), 'classes': classes, 'landmarks': landmarks_array.astype(np.float)}

        
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':

    IOSTAR_dataset = IOSTARDataset(root_dir='./data_IOSTAR/train_images')
    crop = RandomCrop(100)
    heat = HeatMap()

    composed = transforms.Compose([Rescale_image(256), HeatMap()])

    fig = plt.figure(1)

    sample = IOSTAR_dataset[14]
    scale = Rescale_image(sample['image'].shape)

    for i, trfrm in enumerate([scale, crop, heat, composed]):
        print(i)
        trfrm_sample = trfrm(sample)

        ax = plt.subplot(1,4, i+1)
        plt.tight_layout()
        ax.set_title(type(trfrm).__name__)
        ax.axis('off')

        if trfrm==scale:
            plt.imshow(trfrm_sample['image'])
            plt.scatter(trfrm_sample['landmarks'][:,1],trfrm_sample['landmarks'][:,0], s=10, marker='.', c='red')
        if trfrm==crop:
            plt.imshow(trfrm_sample['image'])
        if trfrm == heat:
            plt.imshow(trfrm_sample['heat_map'].squeeze().permute([1,2,0]))
        elif trfrm == composed:
            plt.imshow(trfrm_sample['heat_map'].squeeze().permute([1,2,0]))

    plt.show()