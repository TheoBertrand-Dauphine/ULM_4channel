
import torch
import csv
from PIL import Image
import torchvision

f = open('./data/csv/ULM_points_full_bolus_label_16_07.csv',newline='\n')

list_of_points = []

for row in f.readlines():
    p = row.split(',')
    
    list_of_points.append([int(p[1]),int(p[2]),p[0]])

I = Image.open('./data/rat_brain_bolus_full_sqrt1_16_07.png')
pil_to_tensor = torchvision.transforms.ToTensor()
tensor_to_pil = torchvision.transforms.ToPILImage()

I_tensor = pil_to_tensor(I)

# A = torchvision.transforms.RandomCrop(side_size)

Icropped = I_tensor[:,:,int(I_tensor.shape[2]/2):]

image_to_save = tensor_to_pil(Icropped)

image_to_save.save('./data/test_images/images_ULM/test_ULM1.png')


cropped_list = []

for x in list_of_points:
    
    if x[0]>int(I_tensor.shape[2]/2):
        cropped_list.append([x[1],x[0]-int(I_tensor.shape[2]/2),x[2]])

with open("./data/test_images/ULM_points/point_list_test1.csv","w") as f:
    wr = csv.writer(f)
    wr.writerows(cropped_list)
