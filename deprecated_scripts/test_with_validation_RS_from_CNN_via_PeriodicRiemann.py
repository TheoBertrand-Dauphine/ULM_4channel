#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:34:30 2021

@author: tbertrand
"""

# import sys
# sys.path.append("../ULM_data")
import torch

# import torch.nn as nn
import torchvision
from PIL import Image
# import re
import numpy as np
import matplotlib.pyplot as plt


#%%
import csv

# from OrientationScore import gaussian_OS, compact_OS
            
#%%

I = Image.open('input_image.png')


pil_to_tensor = torchvision.transforms.ToTensor()

MatOut = pil_to_tensor(I)[0,:,:]

point_list = []
with open('output_point_list.csv') as f:
    rd = csv.reader(f)
    for row in rd:
        if len(row)!=0:
            point_list.append([int(row[1]),int(row[2]),int(row[0])])
            
MatOut_OL_window = torch.load('lifted_input_image.pt')

#%%
[Nx,Ny,Nt] = MatOut_OL_window.shape


# MatOut_OL = torch.tensor(gaussian_OS(MatOut,  sigma = 0.01, eps = 0.01, N_o = Nt))

MatOut_window = MatOut
# MatOut_OL_window = MatOut_OL[x:x+Nx,y:y+Ny,:]
# MatOut_OL_window = torch.tensor(gaussian_OS(MatOut_window, sigma = 0.001, eps = 0.1, N_o = Nt))

MatOut_OL_window = (MatOut_OL_window/MatOut_OL_window.max()).sqrt()
MatOut_window = MatOut_window/MatOut_window.max()

plt.figure(1)
plt.clf()
plt.imshow((MatOut_OL_window.sqrt()>0.6).max(dim=2).values, cmap="gray")



label_list = []

for i in range(len(point_list)):

        label_list.append(point_list[i][2])


points_tensor = torch.tensor(point_list)

# sub_list = [19,15,22,6,25,16]
# points_tensor = points_tensor[sub_list,:]

plt.figure(1)
plt.clf()
plt.imshow(MatOut_window, cmap='gray')
plt.scatter(points_tensor[:,1], points_tensor[:,0], s=500, c=2*points_tensor[:,2]+1,marker='.')
# plt.axis('off')
plt.show()
# plt.close()


theta_indices = MatOut_OL_window[[points_tensor[:,0].long(),points_tensor[:,1].long()]].argmax(dim=1) # the theta index of a point is the index that maximizes density of microbubble in image

# Adding double points for crossings and column of 1 for bifurcations

tol_prox_theta = Nt/3

bif_width = 3

for i in range(points_tensor.shape[0]):
    # print( MatOut_OL_window[[points_tensor[i,0].long(),points_tensor[i,1].long()]].topk(Nt).values[0])
    if label_list[i]==2:
        theta_ind_prox = MatOut_OL_window[[points_tensor[i,0].long(),points_tensor[i,1].long()]].topk(Nt).indices
        # indic_prox_theta = torch.abs(theta_ind_prox-theta_indices[i])>tol_prox_theta
        indic_prox_theta = (Nt - torch.abs((2*(theta_ind_prox-theta_indices[i]).abs()) - Nt))>tol_prox_theta #"periodic" distance to take into account periodicity in angle
        
        
        theta_ind = theta_ind_prox[indic_prox_theta][0]
        theta_indices = torch.cat([theta_indices, theta_ind.unsqueeze(0)], dim=0)
        
        points_tensor = torch.cat([points_tensor, points_tensor[i,:].unsqueeze(0)], dim=0)            
        
    if label_list[i] == 1:
        MatOut_OL_window[points_tensor[i,0].long()-bif_width: points_tensor[i,0].long()+bif_width,points_tensor[i,1].long()-bif_width:points_tensor[i,1].long()+bif_width,:]=1.0

MatOut_OL_window[points_tensor[:,0],points_tensor[:,1], theta_indices] = 1.0 #desperate

N_points = points_tensor.shape[0]
print('nombre de points : {}'.format(N_points))

theta_indices = theta_indices.long()

# fig = plt.figure(2)
# ax = fig.add_subplot(projection='3d')
# ax.scatter(points_tensor[:,1], points_tensor[:,0], theta_indices,s = 500, marker='.', c='r')
# ax.set_xlabel('$X$')
# ax.set_ylabel('$Y$')
# plt.close()

#%% Iterating

from agd import Eikonal
from agd.Metrics import Riemann
from agd.LinearParallel import outer_self as Outer # outer product v v^T of a vector with itself
#from agd.Plotting import savefig, quiver; #savefig.dirName = 'Figures/Riemannian'
#from agd import LinearParallel as lp
from agd import AutomaticDifferentiation as ad

import time


norm_infinity = ad.Optimization.norm_infinity


eps = 1e-4
W = 1/((MatOut_OL_window.abs()>0.25).double()*MatOut_OL_window.abs() + eps).transpose(0,1)
# W = 1/( MatOut_OL_window.abs()*(MatOut_OL_window.abs()>0.7).double() + eps).transpose(0,1)
# W[W<1/0.6] = 0.5




hfmIn = Eikonal.dictIn({
    'model':'Riemann3_Periodic', # The third dimension is periodic (and only this one), in this model.
})

# hfmIn = Eikonal.dictIn({
#     'model': 'ReedsShepp2', # Two-dimensional Riemannian eikonal equation
#     'xi': 0.5/np.pi,
#     'seedValue':0, # Can be omitted, since this is the default.
#     'projective':1
# })

hfmIn.SetRect(sides=[[0,1],[0,1],[0,np.pi]],dims=[Nx,Ny,Nt])

hfmIn['order'] = 1

X,Y,Theta = hfmIn.Grid()
zero = np.zeros_like(X)

alpha = 0.1
xi = 0.5/np.pi

ReedsSheppMetric = Riemann( # Riemannian metric defined by a positive definite tensor field
    Outer([np.cos(Theta), np.sin(Theta),zero])
    + alpha**(-2)*Outer([-np.sin(Theta),np.cos(Theta),zero])
    + xi**2*Outer([zero,zero,1+zero])
).with_cost(W)

hfmIn['metric'] = ReedsSheppMetric

hfmIn['exportValues'] = 1
hfmIn['verbosity'] = 0
hfmIn['geodesicStep'] = 0.05

n_points = points_tensor.shape[0]

L = []

D = np.zeros((n_points,n_points))

for j in range(n_points-1):    
    print(round((j+1)/n_points,2))
    
    hfmIn['seed'] = np.hstack([points_tensor[j,1], points_tensor[j,0], np.pi*theta_indices[j]])/np.array(W.shape)
    hfmIn['tips'] = np.array(torch.hstack([points_tensor[j+1:,1].unsqueeze(1), points_tensor[j+1:,0].unsqueeze(1), np.pi*theta_indices[j+1:].unsqueeze(1)]))/np.array(W.shape)
    hfmIn['stopWhenAllAccepted'] = hfmIn['tips']
    # hfmIn['stopAtEuclideanLength'] = 100
    
    a = time.time()
    hfmOut = hfmIn.Run()
    
    print(time.time()-a)
    
    D[j,j+1:] = hfmOut['values'][points_tensor[j+1:,1],points_tensor[j+1:,0],theta_indices[j+1:]]
    
    L.append(hfmOut['geodesics'])

D = D+D.transpose()

# max_anisotropy = hfmIn['metric'].anisotropy().max()




#%%
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


D_norm = np.zeros_like(D)

D_norm[D<np.inf] = D[D<np.inf]/(D[D<np.inf]).max()
D_norm[D==np.inf] = 10.0

D[D==np.inf] = 10.0*(D[D<np.inf]).max()

ag = AgglomerativeClustering(linkage='single', distance_threshold = 80.0, n_clusters=None, compute_distances=True, affinity='precomputed')
# ag = AgglomerativeClustering(linkage="single", n_clusters=6, affinity='precomputed')
labels = ag.fit_predict(D)

List_of_curves = []
list_of_stacks = []

prim_dict = np.array(list(range(N_points)))

fig = plt.figure(3)
plt.clf()
ax = fig.add_subplot(projection='3d')
# plt.imshow(torch.tensor(1/hfmIn['cost']).max(dim=2).values.transpose(0,1), cmap="gray")

Tcsr_list = []

suspect_index = 25

for j in range(labels.max()+1):
    ind = (labels==j)
    
    sec_dict = prim_dict[ind]
    
    
    X = csr_matrix(D_norm[ind,:][:,ind])
    
    Tcsr = minimum_spanning_tree(X)
    Tcsr_list.append(Tcsr)
    # Tcsr.toarray().astype(int)
    
    
    Edges_i, Edges_o = Tcsr.toarray().nonzero()
    
    # print(Edges_i.size)
    
    if Edges_i.size>0:
        
        local_list = []
        for r in range(Edges_i.size):
            first_index = sec_dict[Edges_i[r]]
            second_index = sec_dict[Edges_o[r]]
            
            if first_index<second_index:
                List_of_curves.append(L[first_index][second_index - first_index-1])
                local_list.append(L[first_index][second_index - first_index-1])
            else:
                List_of_curves.append(L[second_index][first_index - second_index-1])
                local_list.append(L[second_index][first_index - second_index-1])
                
            if first_index==suspect_index or second_index==suspect_index:
                if first_index<second_index:
                    ax.scatter(Nx*L[first_index][second_index - first_index-1][0,:], Ny*L[first_index][second_index - first_index-1][1,:], np.remainder(Nt*L[first_index][second_index - first_index-1][2,:]/np.pi,Nt))
                else:
                    ax.scatter(Nx*L[second_index][first_index - second_index-1][0,:], Ny*L[second_index][first_index - second_index-1][1,:], np.remainder(Nt*L[second_index][first_index - second_index-1][2,:]/np.pi,Nt))
                
        stacked_list = np.hstack(local_list)
        list_of_stacks.append(stacked_list)
    
        # plt.scatter(Nx*stacked_list[0,:], Ny*stacked_list[1,:], marker='.', s=5, alpha=0.9)
ax.scatter(points_tensor[:,1], points_tensor[:,0], theta_indices, marker='.', s=100, c='r')
# plt.close()

#plt.xlim([0,512])
#plt.ylim([0,512])

plt.figure(4)
plt.clf()
plt.imshow(MatOut, cmap='gray')
# plt.imshow((1/W).max(dim=2).values.transpose(0,1),origin='upper', cmap = "gray")
# plt.imshow(torch.tensor(1/hfmIn['cost']).max(dim=2).values.transpose(0,1), cmap="gray")
for i in range(len(list_of_stacks)):
    stacked_list = list_of_stacks[i]
    plt.scatter(Nx*stacked_list[0,:], Ny*stacked_list[1,:], marker='.', s=50, alpha=0.5)
plt.scatter(points_tensor[:,1], points_tensor[:,0], marker='.', s=500, c=labels)
# plt.title('result with factor in the direction theta {}'.format(theta_cost))
# for i in range(N_points):
#     plt.annotate('{}'.format(i),(points_tensor[i,1]+5, max(points_tensor[i,0]-5,0)), c='g', fontsize=30)

plt.axis('off')
plt.tight_layout()
    

#%%
fig = plt.figure(5)
ax = fig.add_subplot(projection='3d')
for i in range(len(list_of_stacks)):
    stacked_list = list_of_stacks[i]
    ax.scatter(Nx*stacked_list[0,:], Ny*stacked_list[1,:], np.remainder(Nt*stacked_list[2,:]/np.pi, Nt), marker='.', s=50, alpha=0.2)
ax.scatter(points_tensor[:,1], points_tensor[:,0], theta_indices)
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$\theta$')

# plt.show()
#%%
import napari

points_array_3d = np.hstack([points_tensor[:,:-1], theta_indices.unsqueeze(1)])

# viewer = napari.view_image((1/W).numpy())
# viewer.add_points(points_array_3d[:,[1,0,2]], face_color = 'r', size = 3)
# for i in range(len(list_of_stacks)):
#     stacked_list = list_of_stacks[i]
#     viewer.add_points(np.array([Nx*stacked_list[0,::8], Ny*stacked_list[1,::8], np.remainder(Nt*stacked_list[2,::8]/np.pi, Nt)]).transpose(), face_color = ['blue','orange'][i], size=1)

#%%
import networkx as nx

T = Tcsr_list[0]
G = nx.from_scipy_sparse_matrix(T)
lab = [(i,'{}'.format(prim_dict[labels==0][i])) for i in range(len(prim_dict[labels==0]))]
plt.figure(7)
plt.clf()
nx.draw_planar(G,labels=dict(lab),with_labels=True,node_size=1000)
plt.show()