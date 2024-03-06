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

import napari
import scipy
import kornia

import torchvision

def gaussian_OS(Im, sigma = 0.001, eps = 0.1, N_o = 64):
    """
    Computes the Gaussian Orientation Score for an input image.

    Parameters:
    - Im: torch.Tensor, input image
    - sigma: float, standard deviation of the Gaussian kernel (default: 0.001)
    - eps: float, regularization parameter (default: 0.1)
    - N_o: int, number of orientations (default: 64)

    Returns:
    - out: torch.Tensor, Gaussian Orientation Score of the input image
    """
    
    theta = torch.arange(-np.pi, np.pi, np.pi/N_o).unsqueeze(0).unsqueeze(0)

    Nx = Im.shape[0]
    Ny = Im.shape[1]
    
    kx = torch.arange(-(Nx/Ny),(Nx/Ny),step = (2*(Nx/Ny))/Nx)
    ky = torch.arange(-1,1,step = 2/Ny)
    
    [Yg,Xg] = torch.meshgrid(kx,ky)
    
    P = torch.exp((-(torch.sin(theta)*Xg.unsqueeze(2) + torch.cos(theta)*Yg.unsqueeze(2))**2)/(sigma**2))*torch.exp(-(Xg.unsqueeze(2)**2+Yg.unsqueeze(2)**2)/(eps**2))*((-torch.cos(theta)*Xg.unsqueeze(2) + torch.sin(theta)*Yg.unsqueeze(2))>0)
    P = P/P.sum(dim=(0,1))
    
    A = convolve(P.numpy(), Im.unsqueeze(2).numpy(), mode='same')

    out = 0.5*( A[...,:N_o] + A[...,N_o:])
    return(out)


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
    
    P = P/P.sum(dim=(0,1),keepdim=True)

    print(P.shape)

    
    A = convolve(P.numpy(), Im.unsqueeze(2).numpy(), mode='same')

    return(A)

def vesselness_from_OS(OS, eps=1e-2):
    """
    Computes the vesselness score from the orientation score.

    Args:
        OS (torch.Tensor): The orientation score tensor.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-2.

    Returns:
        torch.Tensor: The vesselness score tensor.
    """

    x = torch.linspace(0,1.,OS.shape[0])
    y = torch.linspace(0,1.,OS.shape[1])
    theta = torch.linspace(0,torch.pi,OS.shape[2])

    assert OS.shape[0]==OS.shape[1]
    assert OS.ndim==3

    OS_unsqueezed = OS.unsqueeze(0).unsqueeze(0)

    [X,Y,T] = torch.meshgrid(x,y,theta)
    
    sigma1 =.5

    Gradient3d = kornia.filters.SpatialGradient3d(mode='diff', order=1)

    print(OS_unsqueezed.shape)
    G = Gradient3d(OS_unsqueezed)
    print(G.shape)
    H = Gradient3d(G[0]).squeeze()

    print(H.shape)

    S = ((torch.cos(T)**2)*H[0,0] + 2*torch.cos(T)*torch.sin(T)*H[0,1] + (torch.sin(T)**2)*H[1,1])**2 + ((torch.sin(T)**2)*H[0,0] - 2*torch.cos(T)*torch.sin(T)*H[0,1] + (torch.cos(T)**2)*H[1,1])**2 
    R = ((torch.cos(T)**2)*H[0,0] + 2*torch.cos(T)*torch.sin(T)*H[0,1] + (torch.sin(T)**2)*H[1,1])/((torch.sin(T)**2)*H[0,0] - 2*torch.cos(T)*torch.sin(T)*H[0,1] + (torch.cos(T)**2)*H[1,1])
    Q = ((torch.sin(T)**2)*H[0,0] - 2*torch.cos(T)*torch.sin(T)*H[0,1] + (torch.cos(T)**2)*H[1,1])

    sigma2 = .2*S.max()

    V = torch.zeros_like(S)
    V[Q>0] = (torch.exp(-R**2/(2*sigma1**2))*(1-torch.exp(-S/(2*sigma2))))[Q>0]

    return V

# B-Splines functions

def B_3(input):
    x = torch.maximum(torch.minimum(input.clone(), 5*torch.ones_like(input)), -5*torch.ones_like(input))
    # x = input.clone()
    return (4*(-1 + x)**3*torch.sign(1 - x) - (-2 + x)**3*torch.sign(2 - x) + 6*x**3*torch.sign(x) - 4*(1 + x)**3*torch.sign(1 + x) + (2 + x)**3*torch.sign(2 + x))/12

def B_8(input):
    # x = torch.maximum(torch.minimum(input.clone(), 5*torch.ones_like(input)), -5*torch.ones_like(input))
    x = input.clone()
    out = (126*(-1/2 + x)**8*torch.sign(1/2 - x) - 84*(-3/2 + x)**8*torch.sign(3/2 - x) + 36*(-5/2 + x)**8*torch.sign(5/2 - x) - 9*(-7/2 + x)**8*torch.sign(7/2 - x) + (-9/2 + x)**8*torch.sign(9/2 - x) + (63*(1 + 2*x)**8*torch.sign(1/2 + x))/128 - 84*(3/2 + x)**8*torch.sign(3/2 + x) + (9*(5 + 2*x)**8*torch.sign(5/2 + x))/64 - 9*(7/2 + x)**8*torch.sign(7/2 + x) + ((9 + 2*x)**8*torch.sign(9/2 + x))/256)/80640
    return torch.maximum(out, torch.zeros_like(out))

def B_30(x):
    return (-300540195*(-1/2 + x)**30*torch.sign(1/2 - x) + 265182525*(-3/2 + x)**30*torch.sign(3/2 - x) - 206253075*(-5/2 + x)**30*torch.sign(5/2 - x) + 141120525*(-7/2 + x)**30*torch.sign(7/2 - x) - 84672315*(-9/2 + x)**30*torch.sign(9/2 - x) + 44352165*(-11/2 + x)**30*torch.sign(11/2 - x) - 20160075*(-13/2 + x)**30*torch.sign(13/2 - x) + 7888725*(-15/2 + x)**30*torch.sign(15/2 - x) - 2629575*(-17/2 + x)**30*torch.sign(17/2 - x) + 736281*(-19/2 + x)**30*torch.sign(19/2 - x) - 169911*(-21/2 + x)**30*torch.sign(21/2 - x) + 31465*(-23/2 + x)**30*torch.sign(23/2 - x) - 4495*(-25/2 + x)**30*torch.sign(25/2 - x) + 465*(-27/2 + x)**30*torch.sign(27/2 - x) - 31*(-29/2 + x)**30*torch.sign(29/2 - x) + (-31/2 + x)**30*torch.sign(31/2 - x) - 300540195*(1/2 + x)**30*torch.sign(1/2 + x) + 265182525*(3/2 + x)**30*torch.sign(3/2 + x) - 206253075*(5/2 + x)**30*torch.sign(5/2 + x) + 141120525*(7/2 + x)**30*torch.sign(7/2 + x) - 84672315*(9/2 + x)**30*torch.sign(9/2 + x) + 44352165*(11/2 + x)**30*torch.sign(11/2 + x) - 20160075*(13/2 + x)**30*torch.sign(13/2 + x) + 7888725*(15/2 + x)**30*torch.sign(15/2 + x) - 2629575*(17/2 + x)**30*torch.sign(17/2 + x) + 736281*(19/2 + x)**30*torch.sign(19/2 + x) - 169911*(21/2 + x)**30*torch.sign(21/2 + x) + 31465*(23/2 + x)**30*torch.sign(23/2 + x) - 4495*(25/2 + x)**30*torch.sign(25/2 + x) + 465*(27/2 + x)**30*torch.sign(27/2 + x) - 31*(29/2 + x)**30*torch.sign(29/2 + x) + (31/2 + x)**30*torch.sign(31/2 + x))/530505719624382117272616960000000

def B_16(x):
    return (24310*(-1/2 + x)**16*torch.sign(1/2 - x) - 19448*(-3/2 + x)**16*torch.sign(3/2 - x) + 12376*(-5/2 + x)**16*torch.sign(5/2 - x) - 6188*(-7/2 + x)**16*torch.sign(7/2 - x) + 2380*(-9/2 + x)**16*torch.sign(9/2 - x) - 680*(-11/2 + x)**16*torch.sign(11/2 - x) + 136*(-13/2 + x)**16*torch.sign(13/2 - x) - 17*(-15/2 + x)**16*torch.sign(15/2 - x) + (-17/2 + x)**16*torch.sign(17/2 - x) + 24310*(1/2 + x)**16*torch.sign(1/2 + x) - 19448*(3/2 + x)**16*torch.sign(3/2 + x) + (1547*(5 + 2*x)**16*torch.sign(5/2 + x))/8192 - 6188*(7/2 + x)**16*torch.sign(7/2 + x) + 2380*(9/2 + x)**16*torch.sign(9/2 + x) - 680*(11/2 + x)**16*torch.sign(11/2 + x) + (17*(13 + 2*x)**16*torch.sign(13/2 + x))/8192 - 17*(15/2 + x)**16*torch.sign(15/2 + x) + (17/2 + x)**16*torch.sign(17/2 + x))/41845579776000
 
def psit_MS(w, N_a=12, N_o=32, Nf=128, scalemin=25, scalemax=250, window_size = 25):
    #format w [2,N,N]
    assert w.ndim==3

    [X,Y] = torch.meshgrid(torch.linspace(-1,1,w.shape[1]),torch.linspace(-1,1,w.shape[2]))

    s_theta = .1*torch.pi/N_o

    scale_min = scalemin*2/w.shape[1]
    scale_max = scalemax*2/w.shape[1]


    a = torch.exp(torch.linspace(torch.log(torch.tensor(scale_min)),torch.log(torch.tensor(scale_max)),N_a))
    s_rho = (torch.log(torch.tensor(scale_max))- torch.log(torch.tensor(scale_min)))/N_a

    print(a)

    rho = (w[0]**2 + w[1]**2).sqrt()
    phi = torch.angle(w[0]+1j*w[1])

    rho_a = a[:,None,None,None]*rho[None,None,:,:]

    theta = torch.linspace(0, 2*torch.pi, N_o)[None,:,None,None]

    fft_res = torch.zeros([N_a,N_o,w.shape[1], w.shape[2]])
    fft_res[:,:,rho>0] = (B_3(torch.log(rho_a)/s_rho)*B_3((torch.remainder(phi- theta - .5*torch.pi,2*torch.pi)-torch.pi)/s_theta))[:,:,rho>0]
    # fft_res[:,:,rho>0] = (B_3(torch.log(rho_a)/s_rho)*B_3(((torch.remainder(phi-theta-torch.pi/2,torch.pi))-.5*torch.pi)/s_theta))[:,:,rho>0]
    fft_res[:,:,rho==0] = 1/N_o
    
    s_x = (window_size/w.shape[1])/a.min()

    gauss_factor = torch.exp(((-X**2-Y**2)/(s_x**2))[None,None,:,:]/(a[:,None,None,None]**2))
    gauss_factor = gauss_factor/gauss_factor.abs().sum(dim=[-1,-2], keepdims=True)

    psit = torch.fft.ifft2(fft_res)*gauss_factor

    # napari.view_image(psit.mean(dim=0).real.numpy())

    # plt.imshow(psit[4,0].real)
    # plt.colorbar()

    # M = (torch.fft.fft2(psit).abs()/a[:,None,None,None]).sum(dim=[0,1],keepdims=True)/N_o/N_a
    M = torch.fft.fft2(psit).abs().mean(dim=[0,1],keepdims=True)

    # plt.imshow((torch.fft.fft2(psit)/M)[4,0].real)

    # Crop = torchvision.transforms.CenterCrop(Nf)
    return torch.fft.ifft2(torch.fft.fft2(psit)/M)


def Cake_OS(Im, No=128, Na=50, scale_min=25, scale_max=250, window_size = 25):
    N = Im.shape[0]

    f = torch.fft.fftfreq(N,d=2/N)/2
    W = torch.stack(torch.meshgrid(f,f, indexing='ij'))

    A = psit_MS(W, N_a=Na, N_o=No,)
    I = torch.tensor(Im).float() +0.j


    I_expand = I.expand([1,1,N,N])

    dirac = torch.zeros_like(I_expand)
    dirac[0,0,N//2,N//2] = 1.

    filtered = torch.fft.ifft2(torch.fft.fft2(dirac)*torch.fft.fft2(I_expand)*torch.fft.fft2(A.conj())) #Weird fix but OK
    
    I_OS = ((filtered).mean(dim=0))
    I_OS = (I_OS.real.abs()/I_OS.abs().max())**2

    # return .5*(torch.flip(I_OS[:No//2], dims=[0])+I_OS[No//2:])
    return .5*(I_OS[:No//2] + I_OS[No//2:])
    # return I_OS

if __name__ == '__main__':
    from skimage import io
    from skimage.filters import frangi

    N = 256

    Crop = torchvision.transforms.CenterCrop(N)

    # u = io.imread('../data_synthetic/test_images/images_ULM/big_test_image.png')
    # I = Crop(torch.tensor(u/255.).float()) + 0.j
    # u = io.imread('../data_IOSTAR/test_images/images_IOSTAR/test_IOSTAR_6.png')
    # I = torch.tensor(frangi(Crop(torch.tensor(u[:256,:,1]/255.).float()).numpy())) +0.j
    [X,Y] = torch.meshgrid(torch.linspace(-1,1,N),torch.linspace(-1,1,N))
    I = (X.abs()<.01) + (Y.abs()<.01) + ((X+Y).abs()<.01) + ((X-Y).abs()<.01) +0.j
    

    I_OS = Cake_OS(I, Na=10, No = 128, scale_min=10, scale_max=50, window_size = 25)

    plt.figure()
    plt.imshow(I_OS.real.sum(dim=0))
    plt.colorbar()

    lamb = 1000
    C = 1/(1 + lamb*I_OS**2)

    V = vesselness_from_OS((I_OS.abs()/I_OS.abs().max()).permute([1,2,0]))

    # napari.view_image((I_OS.abs()/I_OS.abs().max()).permute([0,2,1]).numpy()**.25)
    # napari.run()


    # Vesselness = vesselness_from_OS(filtered.reshape([Na,No,N,N]).real.permute([0,1,3,2]))

    # Vesselness_thresholded = (Vesselness - kornia.filters.gaussian_blur2d(Vesselness, (3,3), (1,1))).numpy()>0.05 + 0.
    
