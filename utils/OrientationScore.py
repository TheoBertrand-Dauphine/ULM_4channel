#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 11:12:32 2021

@author: tbertrand
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# import napari

def gaussian_OS(Im, sigma = 0.001, eps = 0.1, N_o = 64):
    
    theta = torch.arange(0, np.pi, np.pi/N_o).unsqueeze(0).unsqueeze(0)

    Nx = Im.shape[0]
    Ny = Im.shape[1]
    
    kx = torch.arange(-(Nx/Ny),(Nx/Ny),step = (2*(Nx/Ny))/Nx)
    ky = torch.arange(-1,1,step = 2/Ny)
    
    [Yg,Xg] = torch.meshgrid(kx,ky)
    
    P = torch.exp((-(torch.cos(theta+np.pi/2)*Xg.unsqueeze(2) + torch.cos(theta)*Yg.unsqueeze(2))**2)/(sigma**2))*torch.exp(-(Xg.unsqueeze(2)**2+Yg.unsqueeze(2)**2)/(eps**2))
    P = P/P.sum(dim=(0,1))
    
    # napari.view_image(P.numpy())

        
    A = convolve(P.numpy(), Im.unsqueeze(2).numpy(), mode='same')

    return(A)


def compact_OS(Im, sigma = 0.001, eps = 0.1, N_o = 64):
    
    theta = torch.arange(0, np.pi, np.pi/N_o).unsqueeze(0).unsqueeze(0)
    
    Nx = Im.shape[0]
    Ny = Im.shape[1]
    
    kx = torch.arange(-1,1,step = 2/Nx)
    ky = torch.arange(-1,1,step = 2/Ny)
    
    [Yg,Xg] = torch.meshgrid(kx,ky)
    
    N = ((torch.cos(theta+np.pi/2)*Xg.unsqueeze(2) + torch.cos(theta)*Yg.unsqueeze(2))**2)/(sigma**2) + (Xg.unsqueeze(2)**2+Yg.unsqueeze(2)**2)/(eps**2)
    
    # P = torch.exp((-(torch.cos(theta+np.pi/2)*Xg.unsqueeze(2) + torch.cos(theta)*Yg.unsqueeze(2))**2)/(sigma**2))*torch.exp(-(Xg.unsqueeze(2)**2+Yg.unsqueeze(2)**2)/(eps**2))
    # P = P/P.sum(dim=(0,1))
    
    # print(P.shape)
        
    P = torch.exp((-1/(1-N)))*(N<1); P[torch.isnan(P)]=0
    
    P = P/P.max()

    
    A = convolve(P.numpy(), Im.unsqueeze(2).numpy(), mode='same')

    return(A)