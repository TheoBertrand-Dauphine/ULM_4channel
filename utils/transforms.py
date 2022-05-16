import torch 
from torchvision import transforms, utils
from skimage import transform
import numpy as np
import torchgeometry
from PIL import Image 

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int,tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, classes = sample['image'], sample['landmarks'], sample['classes']

        if image.ndim==3:
            h, w = image.shape[1:]
        else:
            h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w :
                new_h, new_w = self.output_size * h  / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h

        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        if image.ndim ==3: #with C number of channels, image is assumed to have shape [C,ny,nx]
            img = transform.resize(image, (image.shape[0], new_h, new_w))
        else:
            img = transform.resize(image, (new_h, new_w))

        landmarks = landmarks * [new_w / w, new_h / h, 1]

        return {'image':img, 'landmarks': landmarks, 'classes':classes}

class Rescale_image(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int,tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, classes = sample['image'], sample['landmarks'], sample['classes']

        if image.ndim==3:
            h, w = image.shape[1:]
        else:
            h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w :
                new_h, new_w = self.output_size * h  / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h

        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        if image.ndim ==3: #with C number of channels, image is assumed to have shape [C,ny,nx]
            img = transform.resize(image, (image.shape[0], new_h, new_w))
        else:
            img = transform.resize(image, (new_h, new_w))
        
        return {'image':img, 'landmarks': landmarks, 'classes':classes}

class RandomAffine(object):

    def __init__(self, angle, t_value):

        # assert isinstance(t_value, (double,tuple))
        self.t_value = t_value
        self.angle = angle
        self.transform = transforms.RandomAffine(self.angle, translate=(self.t_value, self.t_value))

    def __call__(self, sample):
        image, landmarks, heat_map = sample['image'], sample['landmarks'], sample['heat_map']

        if image.ndim==3:
            transform_output = self.transform(torch.cat([image.unsqueeze(0), heat_map],1))
            return {'image': transform_output[:,:image.shape[0],:,:].squeeze(), 'landmarks': landmarks, 'heat_map': transform_output[:,image.shape[0]:,:,:].squeeze()}
        else:
            transform_output = self.transform(torch.cat([image.unsqueeze(0).unsqueeze(0), heat_map],1))
            return {'image': transform_output[:,:1,:,:].squeeze(), 'landmarks': landmarks, 'heat_map': transform_output[:,1:4,:,:].squeeze()}

class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2 
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, classes = sample['image'], sample['landmarks'], sample['classes']

        h, w = image.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        if image.ndim==3:
            image = image[:,top: top + new_h, left: left + new_w]
        else:
            image = image[top: top + new_h, left: left + new_w]

        landmarks = landmarks - [top, left, 0]

        return {'image':image, 'landmarks': landmarks, 'classes':classes}


class ToTensor(object):

    def __call__(self,sample):
        image, landmarks, classes, heat_map = sample['image'], sample['landmarks'], sample['classes'], sample['heat_map']
        return {'image':torch.from_numpy(image).float(), 'heat_map': heat_map, 'landmarks': torch.from_numpy(landmarks)}

class HeatMap(object):

    def __init__(self, s=9, alpha = 3., out_channels=3):
        self.size = s 
        self.alpha = alpha
        self.out_channels = out_channels
        self.gaussian_blur = torchgeometry.image.gaussian.GaussianBlur((self.size, self.size), (self.alpha, self.alpha))


    def __call__(self, sample):
        image, landmarks, classes = sample['image'], sample['landmarks'], sample['classes']

        if image.ndim==3:
            heat_map = torch.zeros(1, self.out_channels, image.shape[1], image.shape[2])
        else:
            heat_map = torch.zeros(1, self.out_channels, image.shape[0], image.shape[1])

        # for rows in landmarks[landmarks[:,1]**2+landmarks[:,0]**2 > 0,:]:
        #     heat_map[0,int(rows[2]),int(rows[0]),int(rows[1])] = 1
        heat_map[0,landmarks[landmarks[:,1]**2+landmarks[:,0]**2 > 0,2].astype(int),landmarks[landmarks[:,1]**2+landmarks[:,0]**2 > 0,0].astype(int),landmarks[landmarks[:,1]**2+landmarks[:,0]**2 > 0,1].astype(int)] = 1.

        heat_map = self.gaussian_blur(heat_map)
        heat_map = heat_map / heat_map.max()

        if self.out_channels==4:
            heat_map[0,3,:,:] = heat_map[0,:3,:,:].max(dim=0).values

        return {'image':image, 'heat_map': heat_map, 'landmarks': landmarks, 'classes':classes}

class ColorJitter(object):
    def __init__(self, power = 4):
        self.power = power


    def __call__(self, sample):
        image, landmarks, classes = sample['image'], sample['landmarks'], sample['classes']
        random_power = np.exp((2*np.log(self.power))*np.random.rand() - np.log(self.power))
        # print(random_power)
        return {'image': np.sign(image)*np.power(np.abs(image),random_power), 'landmarks': landmarks, 'classes': classes}

class GlobalContrastNormalization(object):
    def __init__(self, bias = 0.1):
        self.bias = bias

    def __call__(self, sample):
        image, landmarks, classes = sample['image'], sample['landmarks'], sample['classes']
        image_out = (image - image.mean(axis=(-1,-2), keepdims=True))/(np.sqrt(self.bias + ((image - image.mean(axis=(-1,-2), keepdims=True))**2).mean(axis=(-1,-2), keepdims=True)))
        return {'image': (image_out-image_out.min(axis=(-1,-2), keepdims=True))/(image_out.max(axis=(-1,-2), keepdims=True)-image_out.min(axis=(-1,-2), keepdims=True)), 'landmarks': landmarks, 'classes': classes}

# class ZCAtransform(object):
#     def __init__(self):

#     def __call__(self, sample):
#         image, landmarks, classes = sample['image'], sample['landmarks'], sample['classes']
#         im_vec = image.reshape([-1,1])
#         COV = np.cov(X_norm, rowvar=False)
#         return {'image': (image_out-image_out.min())/(image_out.max()-image_out.min()), 'landmarks': landmarks, 'classes': classes}