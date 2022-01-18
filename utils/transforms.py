import torch 
from torchvision import transforms, utils
from skimage import transform
import numpy as np
import torchgeometry

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int,tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, classes = sample['image'], sample['landmarks'], sample['classes']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w :
                new_h, new_w = self.output_size * h  / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h

        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        landmarks = landmarks * [new_w / w, new_h / h, 1]

        return {'image':img, 'landmarks': landmarks, 'classes':classes}

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

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        heat_map = heat_map[top: top + new_h, left: left + new_w]

        landmarks = landmarks - [top, left, 0]

        return {'image':image, 'landmarks': landmarks, 'classes':classes}


class ToTensor(object):

    def __call__(self,sample):
        image, landmarks, classes, heat_map = sample['image'], sample['landmarks'], sample['classes'], sample['heat_map']
        return {'image':torch.from_numpy(image), 'heat_map': heat_map, 'landmarks': torch.from_numpy(landmarks)}

class HeatMap(object):

    def __call__(self, sample):
        image, landmarks, classes = sample['image'], sample['landmarks'], sample['classes']

        heat_map = torch.zeros(1,3,image.shape[0], image.shape[1])
        for rows in landmarks:
            heat_map[0,int(rows[2]),int(rows[0]),int(rows[1])] = 1

        gaussian_blur = torchgeometry.image.gaussian.GaussianBlur((17,17), (3,3))
        heat_map = gaussian_blur(heat_map)
        heat_map = heat_map / heat_map.max()

        return {'image':image, 'heat_map': heat_map, 'landmarks': landmarks, 'classes':classes}