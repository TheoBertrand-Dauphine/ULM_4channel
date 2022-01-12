# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:08:44 2020

Set of different network designs for computer vision tasks.

@author: theot


"""
import torch
import torch.nn as nn
# from torch.autograd import Variable
import torch.nn.functional as F


#%% Fully convolutional

#define an encoder decoder network with convolution transpose upsampling.
class deconv_hourglass(nn.Module):
    def __init__(self):
        super(deconv_hourglass, self).__init__()
        self.d_conv_1 = nn.Conv2d(3, 8, 7, stride=2, padding=3)
        self.d_bn_1 = nn.BatchNorm2d(8)

        self.d_conv_2 = nn.Conv2d(8, 16, 7, stride=2, padding=3)
        self.d_bn_2 = nn.BatchNorm2d(16)

        self.d_conv_3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.d_bn_3 = nn.BatchNorm2d(32)

        self.d_conv_4 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.d_bn_4 = nn.BatchNorm2d(64)

        self.d_conv_5 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.d_bn_5 = nn.BatchNorm2d(128)

        self.d_conv_6 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.d_bn_6 = nn.BatchNorm2d(256)

        self.u_deconv_5 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.u_bn_5 = nn.BatchNorm2d(128)

        self.u_deconv_4 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.u_bn_4 = nn.BatchNorm2d(64)

        self.u_deconv_3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.u_bn_3 = nn.BatchNorm2d(32)

        self.u_deconv_2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.u_bn_2 = nn.BatchNorm2d(16)

        self.u_deconv_1 = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)
        self.u_bn_1 = nn.BatchNorm2d(8)

        self.out_deconv = nn.ConvTranspose2d(8, 1, 4, stride=2, padding=2)        
        self.out_bn = nn.BatchNorm2d(1)
        

        
    def forward(self, noise):
        # print(noise.shape)
        
        down_1 = self.d_conv_1(noise)
        down_1 = self.d_bn_1(down_1)
        down_1 = F.leaky_relu(down_1)
        
        # print(down_1.shape)
        
        down_2 = self.d_conv_2(down_1)
        down_2 = self.d_bn_2(down_2)
        down_2 = F.leaky_relu(down_2)
        # print(down_2.shape)

        down_3 = self.d_conv_3(down_2)
        down_3 = self.d_bn_3(down_3)
        down_3 = F.leaky_relu(down_3)
        # skip_3 = self.s_conv_3(down_3)
        # print(down_3.shape)


        down_4 = self.d_conv_4(down_3)
        down_4 = self.d_bn_4(down_4)
        down_4 = F.leaky_relu(down_4)
        # skip_4 = self.s_conv_4(down_4)
        # print(down_4.shape)

        down_5 = self.d_conv_5(down_4)
        down_5 = self.d_bn_5(down_5)
        down_5 = F.leaky_relu(down_5)
        # skip_5 = self.s_conv_5(down_5)
        # print(down_5.shape)


        down_6 = self.d_conv_6(down_5)
        down_6 = self.d_bn_6(down_6)
        down_6 = F.leaky_relu(down_6)
        # print(down_6.shape)

        up_5 = self.u_deconv_5(down_6)
        # up_5 = torch.cat([up_5, skip_5], 1)
        up_5 = self.u_bn_5(up_5)
        up_5 = F.leaky_relu(up_5)
        # print(up_5.shape)

        up_4 = self.u_deconv_4(up_5)
        # up_4 = torch.cat([up_4, skip_4], 1)
        up_4 = self.u_bn_4(up_4)
        up_4 = F.leaky_relu(up_4)
        # print(up_4.shape)

        up_3 = self.u_deconv_3(up_4)
        # up_3 = torch.cat([up_3, skip_3], 1)
        up_3 = self.u_bn_3(up_3)
        up_3 = F.leaky_relu(up_3)
        # print(up_3.shape)

        up_2 = self.u_deconv_2(up_3)
        up_2 = self.u_bn_2(up_2)
        up_2 = F.leaky_relu(up_2)
        # print(up_2.shape)

        up_1 = self.u_deconv_1(up_2)
        up_1 = self.u_bn_1(up_1)
        up_1 = F.leaky_relu(up_1)
        # print(up_1.shape)

        out = self.out_deconv(up_1)
        out = self.out_bn(out)
        out = torch.sigmoid(out)
        
        out = F.pad(out,(-5,-4,-27,-27))

        
        return out

#%% Unet

class Unet_bif(nn.Module):
    def __init__(self):
        super(Unet_bif, self).__init__()
        self.d_conv1_layer1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.d_conv2_layer1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        
        self.s_conv_layer1 = nn.Conv2d(32, 8, 3, stride=1, padding=1)
        
        self.d_maxpool_layer2 = nn.MaxPool2d(2,stride=2,padding=0)
        self.d_conv1_layer2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.d_conv2_layer2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        self.s_conv_layer2 = nn.Conv2d(64, 16, 3, stride=1, padding=1)
        
        self.d_maxpool_layer3 = nn.MaxPool2d(2,stride=2,padding=0)
        self.d_conv1_layer3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.d_conv2_layer3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        self.s_conv_layer3 = nn.Conv2d(128, 32, 3, stride=1, padding=0)
        
        self.d_maxpool_layer4 = nn.MaxPool2d(2,stride=2,padding=0)
        self.d_conv1_layer4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.d_conv2_layer4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        
        self.s_conv_layer4 = nn.Conv2d(256, 64, 3, stride=1, padding=0)
        
        self.d_maxpool_layer5 = nn.MaxPool2d(2,stride=2,padding=0)
        self.d_conv1_layer5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.d_conv2_layer5 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        
        self.u_trans_deconv_layer4 = nn.ConvTranspose2d(512, 448, 2, stride=2, padding=0)
        self.u_conv1_layer4 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.u_conv2_layer4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        
        self.u_trans_deconv_layer3 = nn.ConvTranspose2d(256, 224, 2, stride=2, padding=0)
        self.u_conv1_layer3 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.u_conv2_layer3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        self.u_trans_deconv_layer2 = nn.ConvTranspose2d(128, 112, 2, stride=2, padding=0)
        self.u_conv1_layer2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.u_conv2_layer2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        self.u_trans_deconv_layer1 = nn.ConvTranspose2d(64, 56, 2, stride=2, padding=0)
        self.u_conv1_layer1 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.u_conv2_layer1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.out_conv = nn.Conv2d(32, 3, 1, stride=1, padding=2)
        

        
    def forward(self, x):        
        down_1 = F.leaky_relu(self.d_conv1_layer1(x))
        down_1 = F.leaky_relu(self.d_conv2_layer1(down_1))
        
        skip_1 = self.s_conv_layer1(down_1)
        
        down_2 = self.d_maxpool_layer2(down_1)
        down_2 = F.leaky_relu(self.d_conv1_layer2(down_2))
        down_2 = F.leaky_relu(self.d_conv2_layer2(down_2))
        
        skip_2 = self.s_conv_layer2(down_2)

        down_3 = self.d_maxpool_layer3(down_2)
        down_3 = F.leaky_relu(self.d_conv1_layer3(down_3))
        down_3 = F.leaky_relu(self.d_conv2_layer3(down_3))
        
        skip_3 = self.s_conv_layer3(down_3)
        
        down_4 = self.d_maxpool_layer4(down_3)
        down_4 = F.leaky_relu(self.d_conv1_layer4(down_4))
        down_4 = F.leaky_relu(self.d_conv2_layer4(down_4))
        
        skip_4 = self.s_conv_layer4(down_4)

        down_5 = self.d_maxpool_layer5(down_4)
        down_5 = F.leaky_relu(self.d_conv1_layer5(down_5))
        down_5 = F.leaky_relu(self.d_conv2_layer5(down_5))
        
        up_4 = self.u_trans_deconv_layer4(down_5)
        
        up_4 = torch.cat([F.pad(skip_4,(1,1,0,1)),up_4],1)
        
        up_4 = F.leaky_relu(self.u_conv1_layer4(up_4))
        up_4 = F.leaky_relu(self.u_conv2_layer4(up_4))

        up_3 = self.u_trans_deconv_layer3(up_4)
        
        up_3 = torch.cat([F.pad(skip_3,(1,0,0,0)),up_3],1)
        
        up_3 = F.leaky_relu(self.u_conv1_layer3(up_3))
        up_3 = F.leaky_relu(self.u_conv2_layer3(up_3))

        up_2 = self.u_trans_deconv_layer2(up_3)
        
        up_2 = torch.cat([F.pad(skip_2,(-1,-1,-2,-2)),up_2],1)
        
        up_2 = F.leaky_relu(self.u_conv1_layer2(up_2))
        up_2 = F.leaky_relu(self.u_conv2_layer2(up_2))
        
        up_1 = self.u_trans_deconv_layer1(up_2)
        
        up_1 = torch.cat([F.pad(skip_1,(-3,-2,-4,-4)),up_1],1)
        
        up_1 = F.leaky_relu(self.u_conv1_layer1(up_1))
        up_1 = F.leaky_relu(self.u_conv2_layer1(up_1))

        out = self.out_conv(up_1)
        
        out = F.pad(out,(0,1,2,2))        
        
        return out
    
#%% Simple net


class simple_net(nn.Module):   
    def __init__(self):
        super(simple_net, self).__init__()
        
        self.conv_layer1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2,dilation=1,bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv_layer2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2,dilation=1,bias=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv_layer3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2,dilation=1,bias=True)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv_layer4 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2,dilation=1,bias=True)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv_layer5 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2,dilation=1,bias=True)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_layer6 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2,dilation=1,bias=True)
        self.bn5 = nn.BatchNorm2d(32)
        
        
        self.lin_comb_layer1 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0,dilation=1,bias=True)
        # self.lin_comb_layer2 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0,dilation=1,bias=True)
        



    # Defining the forward pass    
    def forward(self, x):
        out1 = self.conv_layer1(x)
        out1 = self.bn1(out1)
        out1 = F.leaky_relu(out1)
        
        out2 = self.conv_layer2(out1)
        out2 = self.bn2(out2)
        out2 = F.leaky_relu(out2)
        
        out3 = self.conv_layer3(out2)
        out3 = self.bn3(out3)
        out3 = F.leaky_relu(out3)
        
        out4 = self.conv_layer4(out3)
        out4 = self.bn4(out4)
        out4 = F.leaky_relu(out4)
        
        out5 = self.conv_layer4(out4)
        out5 = self.bn5(out5)
        out5 = F.leaky_relu(out5)
        
        out6 = self.conv_layer4(out5)

        
        
        
        # out_fin = torch.cat([out1,out2,out3,out4,out5,out6],1)
        
        out_fin = out6
        
        # out_fin = self.lin_comb_layer1(out_fin)
        # F.leaky_relu(out_fin)
        # out_fin = self.lin_comb_layer2(out_fin)
        
        out_fin = torch.sigmoid(out_fin)
        
        out_fin = out_fin.max(1).values
        return out_fin
    
#%%

class Unet_for_ULM(nn.Module):
    def __init__(self):
        super(Unet_for_ULM, self).__init__()
        self.d_conv1_layer1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.d_conv2_layer1 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        
        self.s_conv_layer1 = nn.Conv2d(16, 4, 3, stride=1, padding=1)
        
        self.d_maxpool_layer2 = nn.MaxPool2d(2,stride=2,padding=0)
        self.d_conv1_layer2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.d_conv2_layer2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        
        self.s_conv_layer2 = nn.Conv2d(32, 8, 3, stride=1, padding=1)
        
        self.d_maxpool_layer3 = nn.MaxPool2d(2,stride=2,padding=0)
        self.d_conv1_layer3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.d_conv2_layer3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        self.s_conv_layer3 = nn.Conv2d(64, 16, 3, stride=1, padding=1)
        
        self.d_maxpool_layer4 = nn.MaxPool2d(2,stride=2,padding=0)
        self.d_conv1_layer4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.d_conv2_layer4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        self.s_conv_layer4 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        
        self.d_maxpool_layer5 = nn.MaxPool2d(2,stride=2,padding=0)
        self.d_conv1_layer5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.d_conv2_layer5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        
        self.u_trans_deconv_layer4 = nn.ConvTranspose2d(256, 224, 2, stride=2, padding=0)
        self.u_conv1_layer4 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.u_conv2_layer4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        self.u_trans_deconv_layer3 = nn.ConvTranspose2d(128, 112, 2, stride=2, padding=0)
        self.u_conv1_layer3 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.u_conv2_layer3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        self.u_trans_deconv_layer2 = nn.ConvTranspose2d(64, 56, 2, stride=2, padding=0)
        self.u_conv1_layer2 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.u_conv2_layer2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        
        self.u_trans_deconv_layer1 = nn.ConvTranspose2d(32, 28, 2, stride=2, padding=0)
        self.u_conv1_layer1 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.u_conv2_layer1 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.out_conv = nn.Conv2d(16, 3, 1, stride=1, padding=0)
        

        
    def forward(self, x):        
        down = F.leaky_relu(self.d_conv1_layer1(x))
        down = F.leaky_relu(self.d_conv2_layer1(down))
        
        skip_1 = self.s_conv_layer1(down)
        
        down = self.d_maxpool_layer2(down)
        down = F.leaky_relu(self.d_conv1_layer2(down))
        down = F.leaky_relu(self.d_conv2_layer2(down))
        
        skip_2 = self.s_conv_layer2(down)

        down = self.d_maxpool_layer3(down)
        down = F.leaky_relu(self.d_conv1_layer3(down))
        down = F.leaky_relu(self.d_conv2_layer3(down))
        
        skip_3 = self.s_conv_layer3(down)
        
        down = self.d_maxpool_layer4(down)
        down = F.leaky_relu(self.d_conv1_layer4(down))
        down = F.leaky_relu(self.d_conv2_layer4(down))
        
        skip_4 = self.s_conv_layer4(down)

        down = self.d_maxpool_layer5(down)
        down = F.leaky_relu(self.d_conv1_layer5(down))
        down = F.leaky_relu(self.d_conv2_layer5(down))
        
        # print(down.shape)
        
        up = self.u_trans_deconv_layer4(down)
        
        up = torch.cat([skip_4,up],1)
        
        up = F.leaky_relu(self.u_conv1_layer4(up))
        up = F.leaky_relu(self.u_conv2_layer4(up))

        up = self.u_trans_deconv_layer3(up)
        
        up = torch.cat([skip_3,up],1)
        
        up = F.leaky_relu(self.u_conv1_layer3(up))
        up = F.leaky_relu(self.u_conv2_layer3(up))

        up = self.u_trans_deconv_layer2(up)
        
        up = torch.cat([skip_2,up],1)
        
        up = F.leaky_relu(self.u_conv1_layer2(up))
        up = F.leaky_relu(self.u_conv2_layer2(up))
        
        up = self.u_trans_deconv_layer1(up)
        
        up = torch.cat([skip_1,up],1)
        
        up = F.leaky_relu(self.u_conv1_layer1(up))
        up = F.leaky_relu(self.u_conv2_layer1(up))

        up = self.out_conv(up)
        
        return up
    
#%%

class Unet_for_ULM_big(nn.Module):
    def __init__(self):
        super(Unet_for_ULM_big, self).__init__()
        self.d_conv1_layer1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.d_conv2_layer1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        
        self.s_conv_layer1 = nn.Conv2d(32, 8, 3, stride=1, padding=1)
        
        self.d_maxpool_layer2 = nn.MaxPool2d(2,stride=2,padding=0)
        self.d_conv1_layer2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.d_conv2_layer2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        self.s_conv_layer2 = nn.Conv2d(64, 16, 3, stride=1, padding=1)
        
        self.d_maxpool_layer3 = nn.MaxPool2d(2,stride=2,padding=0)
        self.d_conv1_layer3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.d_conv2_layer3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        self.s_conv_layer3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        
        self.d_maxpool_layer4 = nn.MaxPool2d(2,stride=2,padding=0)
        self.d_conv1_layer4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.d_conv2_layer4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        
        self.s_conv_layer4 = nn.Conv2d(256, 64, 3, stride=1, padding=1)
        
        self.d_maxpool_layer5 = nn.MaxPool2d(2,stride=2,padding=0)
        self.d_conv1_layer5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.d_conv2_layer5 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        
        self.u_trans_deconv_layer4 = nn.ConvTranspose2d(512, 448, 2, stride=2, padding=0)
        self.u_conv1_layer4 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.u_conv2_layer4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        
        self.u_trans_deconv_layer3 = nn.ConvTranspose2d(256, 224, 2, stride=2, padding=0)
        self.u_conv1_layer3 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.u_conv2_layer3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        self.u_trans_deconv_layer2 = nn.ConvTranspose2d(128, 112, 2, stride=2, padding=0)
        self.u_conv1_layer2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.u_conv2_layer2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        self.u_trans_deconv_layer1 = nn.ConvTranspose2d(64, 56, 2, stride=2, padding=0)
        self.u_conv1_layer1 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.u_conv2_layer1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.out_conv = nn.Conv2d(32, 3, 1, stride=1, padding=0)
        

        
    def forward(self, x):        
        down_1 = F.leaky_relu(self.d_conv1_layer1(x))
        down_1 = F.leaky_relu(self.d_conv2_layer1(down_1))
        
        skip_1 = self.s_conv_layer1(down_1)
        
        down_2 = self.d_maxpool_layer2(down_1)
        down_2 = F.leaky_relu(self.d_conv1_layer2(down_2))
        down_2 = F.leaky_relu(self.d_conv2_layer2(down_2))
        
        skip_2 = self.s_conv_layer2(down_2)

        down_3 = self.d_maxpool_layer3(down_2)
        down_3 = F.leaky_relu(self.d_conv1_layer3(down_3))
        down_3 = F.leaky_relu(self.d_conv2_layer3(down_3))
        
        skip_3 = self.s_conv_layer3(down_3)
        
        down_4 = self.d_maxpool_layer4(down_3)
        down_4 = F.leaky_relu(self.d_conv1_layer4(down_4))
        down_4 = F.leaky_relu(self.d_conv2_layer4(down_4))
        
        skip_4 = self.s_conv_layer4(down_4)

        down_5 = self.d_maxpool_layer5(down_4)
        down_5 = F.leaky_relu(self.d_conv1_layer5(down_5))
        down_5 = F.leaky_relu(self.d_conv2_layer5(down_5))
        
        up_4 = self.u_trans_deconv_layer4(down_5)
        
        up_4 = torch.cat([F.pad(skip_4,(0,0,0,0)),up_4],1)
        
        up_4 = F.leaky_relu(self.u_conv1_layer4(up_4))
        up_4 = F.leaky_relu(self.u_conv2_layer4(up_4))

        up_3 = self.u_trans_deconv_layer3(up_4)
        
        up_3 = torch.cat([F.pad(skip_3,(0,0,0,0)),up_3],1)
        
        up_3 = F.leaky_relu(self.u_conv1_layer3(up_3))
        up_3 = F.leaky_relu(self.u_conv2_layer3(up_3))

        up_2 = self.u_trans_deconv_layer2(up_3)
        
        up_2 = torch.cat([F.pad(skip_2,(0,0,0,0)),up_2],1)
        
        up_2 = F.leaky_relu(self.u_conv1_layer2(up_2))
        up_2 = F.leaky_relu(self.u_conv2_layer2(up_2))
        
        up_1 = self.u_trans_deconv_layer1(up_2)
        
        up_1 = torch.cat([F.pad(skip_1,(0,0,0,0)),up_1],1)
        
        up_1 = F.leaky_relu(self.u_conv1_layer1(up_1))
        up_1 = F.leaky_relu(self.u_conv2_layer1(up_1))

        out = self.out_conv(up_1)
        
        return out
    
#%%

class Unet_for_ULM_out4(nn.Module):
    
    def __init__(self):
        super(Unet_for_ULM_out4, self).__init__()
        self.d_conv1_layer1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.d_conv2_layer1 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        
        self.s_conv_layer1 = nn.Conv2d(16, 4, 3, stride=1, padding=1)
        
        self.d_maxpool_layer2 = nn.MaxPool2d(2,stride=2,padding=0)
        self.d_conv1_layer2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.d_conv2_layer2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        
        self.s_conv_layer2 = nn.Conv2d(32, 8, 3, stride=1, padding=1)
        
        self.d_maxpool_layer3 = nn.MaxPool2d(2,stride=2,padding=0)
        self.d_conv1_layer3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.d_conv2_layer3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        self.s_conv_layer3 = nn.Conv2d(64, 16, 3, stride=1, padding=1)
        
        self.d_maxpool_layer4 = nn.MaxPool2d(2,stride=2,padding=0)
        self.d_conv1_layer4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.d_conv2_layer4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        self.s_conv_layer4 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        
        self.d_maxpool_layer5 = nn.MaxPool2d(2,stride=2,padding=0)
        self.d_conv1_layer5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.d_conv2_layer5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        
        self.u_trans_deconv_layer4 = nn.ConvTranspose2d(256, 224, 2, stride=2, padding=0)
        self.u_conv1_layer4 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.u_conv2_layer4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        self.u_trans_deconv_layer3 = nn.ConvTranspose2d(128, 112, 2, stride=2, padding=0)
        self.u_conv1_layer3 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.u_conv2_layer3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        self.u_trans_deconv_layer2 = nn.ConvTranspose2d(64, 56, 2, stride=2, padding=0)
        self.u_conv1_layer2 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.u_conv2_layer2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        
        self.u_trans_deconv_layer1 = nn.ConvTranspose2d(32, 28, 2, stride=2, padding=0)
        self.u_conv1_layer1 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.u_conv2_layer1 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.out_conv = nn.Conv2d(16, 4, 1, stride=1, padding=0)
        

        
    def forward(self, x):        
        down = F.leaky_relu(self.d_conv1_layer1(x))
        down = F.leaky_relu(self.d_conv2_layer1(down))
        
        skip_1 = self.s_conv_layer1(down)
        
        down = self.d_maxpool_layer2(down)
        down = F.leaky_relu(self.d_conv1_layer2(down))
        down = F.leaky_relu(self.d_conv2_layer2(down))
        
        skip_2 = self.s_conv_layer2(down)

        down = self.d_maxpool_layer3(down)
        down = F.leaky_relu(self.d_conv1_layer3(down))
        down = F.leaky_relu(self.d_conv2_layer3(down))
        
        skip_3 = self.s_conv_layer3(down)
        
        down = self.d_maxpool_layer4(down)
        down = F.leaky_relu(self.d_conv1_layer4(down))
        down = F.leaky_relu(self.d_conv2_layer4(down))
        
        skip_4 = self.s_conv_layer4(down)

        down = self.d_maxpool_layer5(down)
        down = F.leaky_relu(self.d_conv1_layer5(down))
        down = F.leaky_relu(self.d_conv2_layer5(down))
        
        # print(down.shape)
        
        up = self.u_trans_deconv_layer4(down)
        
        up = torch.cat([skip_4,up],1)
        
        up = F.leaky_relu(self.u_conv1_layer4(up))
        up = F.leaky_relu(self.u_conv2_layer4(up))

        up = self.u_trans_deconv_layer3(up)
        
        up = torch.cat([skip_3,up],1)
        
        up = F.leaky_relu(self.u_conv1_layer3(up))
        up = F.leaky_relu(self.u_conv2_layer3(up))

        up = self.u_trans_deconv_layer2(up)
        
        up = torch.cat([skip_2,up],1)
        
        up = F.leaky_relu(self.u_conv1_layer2(up))
        up = F.leaky_relu(self.u_conv2_layer2(up))
        
        up = self.u_trans_deconv_layer1(up)
        
        up = torch.cat([skip_1,up],1)
        
        up = F.leaky_relu(self.u_conv1_layer1(up))
        up = F.leaky_relu(self.u_conv2_layer1(up))

        up = self.out_conv(up)
        
        return up