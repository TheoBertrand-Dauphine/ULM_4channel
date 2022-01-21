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

try:
    from utils.transforms import Rescale, RandomCrop, ToTensor, HeatMap
except:
    from transforms import Rescale, RandomCrop, ToTensor, HeatMap


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

        landmarks_array = np.zeros([40,3])

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


if __name__ == '__main__':

    ULM_dataset = ULMDataset(root_dir='./data_synthetic/val_images')
    scale = Rescale(256)
    crop = RandomCrop(128)
    heat = HeatMap()

    composed = transforms.Compose([Rescale(256), HeatMap()])
    fig = plt.figure()

    sample = ULM_dataset[3]
    for i, trfrm in enumerate([scale, crop, heat, composed]):
        print(i)
        trfrm_sample = trfrm(sample)
        
        #print(i, sample['image'].shape, sample['landmarks'].shape, sample['classes'].shape)
        
        ax = plt.subplot(1,4, i+1)
        plt.tight_layout()
        ax.set_title(type(trfrm).__name__)
        ax.axis('off')

        if trfrm == heat:
            plt.imshow(trfrm_sample['heat_map'].squeeze().permute([1,2,0]))
            plt.scatter(trfrm_sample['landmarks'][:, 1], trfrm_sample['landmarks'][:, 0], s=10, marker='.', c='blue')
        elif trfrm == composed:
            plt.imshow(trfrm_sample['image'])
            plt.scatter(trfrm_sample['landmarks'][:, 1], trfrm_sample['landmarks'][:, 0], s=10, marker='.', c='blue')

    plt.show()