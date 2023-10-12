import numpy as np

# from PIL import Image

import torch
import torchvision
from torchvision import transforms


# import agd
from agd import Eikonal
from agd.Metrics import Riemann
from agd.LinearParallel import outer_self as Outer # outer product v v^T of a vector with itself
#from agd.Plotting import savefig, quiver; #savefig.dirName = 'Figures/Riemannian'
#from agd import LinearParallel as lp
# from agd import AutomaticDifferentiation as ad

import time

from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

import matplotlib.pyplot as plt

from utils.OrientationScore import gaussian_OS
from utils.dataset import ULMDataset, IOSTARDataset

from nn.ulm_unet import ULM_UNet

from tqdm import tqdm

from skimage.filters import frangi

try:
    from utils.transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image, ColorJitter, GlobalContrastNormalization, RandomAffine, Padding, CenterCrop, ToArray
except:
    from transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image, ColorJitter, GlobalContrastNormalization, RandomAffine, Padding, CenterCrop, ToArray
    
import networkx as nx

import sys
sys.path.append("./../ULM_data")

from make_ulm_images import making_ULM_halfleft_rat_brain2D_and_orientation, making_ULM_bolus_full_rat_brain2D_and_orientation



def Compute_Distance_Matrix(W, points_tensor, theta_indices, alpha = 0.1, xi = 0.1/np.pi):
    hfmIn = Eikonal.dictIn({
        'model':'Riemann3_Periodic', # The third dimension is periodic (and only this one), in this model.
        })

    [Nx,Ny,Nt] = W.shape; print(Nx,Ny,Nt)

    hfmIn.SetRect(sides=[[0,Nx/Ny],[0,1],[0,np.pi]],dims=[Nx,Ny,Nt])

    hfmIn['order'] = 2

    X,Y,Theta = hfmIn.Grid()
    zero = np.zeros_like(X)

    ReedsSheppMetric = Riemann( # Riemannian metric defined by a positive definite tensor field
        Outer([np.cos(Theta), np.sin(Theta),zero])
        + alpha**(-2)*Outer([-np.sin(Theta),np.cos(Theta),zero])
        + xi**2*Outer([zero,zero,1+zero])
        ).with_cost(W)

    hfmIn['metric'] = ReedsSheppMetric
    
    # hfmIn = Eikonal.dictIn({
    # 'model': 'ReedsShepp2', # Two-dimensional Riemannian eikonal equation
    # 'xi': xi,
    # 'seedValue':0, # Can be omitted, since this is the default.
    # 'projective':1
    # })
    
    # W = np.ones([Nx,Nx,Nt])
    
    # [Nx,Ny,Nt] = W.shape; print(Nx,Ny,Nt)
    
    # hfmIn.SetRect(sides=[[0,Nx/Ny],[0,1],[0,np.pi]],dims=[Nx,Ny,Nt])
    
    # hfmIn['cost'] = W
    
    # hfmIn.SetRect(sides=[[0,1],[0,1]], gridScale = 1/Nx)
    # hfmIn.nTheta = 2*Nt

    hfmIn['exportValues'] = 1
    hfmIn['verbosity'] = 0
    hfmIn['geodesicStep'] = 0.05

    n_points = points_tensor.shape[0]

    L = []

    D = np.zeros((n_points,n_points))
    
    points_array_3d = np.hstack([points_tensor[:,:-1], theta_indices.unsqueeze(1)])

    for j in tqdm(range(n_points-1)):
        # print(round((j+1)/n_points,2))
        
        hfmIn['seed'] = np.hstack([points_tensor[j,1]*(Nx/Ny)/Nx, points_tensor[j,0]/Ny, np.pi*theta_indices[j]/Nt]);

        
        hfmIn['tips'] = np.array(torch.hstack([points_tensor[j+1:,1].unsqueeze(1)*(Nx/Ny)/Nx, points_tensor[j+1:,0].unsqueeze(1)/Ny, np.pi*theta_indices[j+1:].unsqueeze(1)/Nt])); 
        # hfmIn['tips'] = hfmIn.PointFromIndex(points_array_3d[:,[1,0,2]])
        hfmIn['stopWhenAllAccepted'] = hfmIn['tips']
        # hfmIn['stopAtEuclideanLength'] = 100
        
        a = time.time()
        hfmOut = hfmIn.Run()
        
        # print(time.time()-a)
        
        # print(points_tensor[j+1:])
        D[j,j+1:] = hfmOut['values'][points_tensor[j+1:,1],points_tensor[j+1:,0],theta_indices[j+1:]]
        
        L.append(hfmOut['geodesics'])

    D = D+D.transpose()
    return([D, L, hfmIn])


def Cluster_from_Distance(D, L, distance_threshold = 10.):
    D_norm = np.zeros_like(D)
    
    D_norm[D<np.inf] = D[D<np.inf]/(D[D<np.inf]).max()
    D_norm[D==np.inf] = 10.0

    D[D==np.inf] = 10.0*(D[D<np.inf]).max()
    
    ag = AgglomerativeClustering(linkage='single', distance_threshold = distance_threshold, n_clusters=None, compute_distances=True, affinity='precomputed')
    # ag = AgglomerativeClustering(linkage="single", n_clusters=6, affinity='precomputed')
    labels = ag.fit_predict(D)

    List_of_curves = []
    list_of_stacks = []

    prim_dict = np.array(list(range(D.shape[0])))

    # fig = plt.figure(3)
    # plt.clf()
    # ax = fig.add_subplot(projection='3d')
    
    print('Nombre de clusters etablis :' +  str(labels.max()+1))

    Tcsr_list = []

    for j in range(labels.max()+1):
        
        # print(j)
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
                    
            stacked_list = np.hstack(local_list)
            list_of_stacks.append(stacked_list)
    return([List_of_curves, list_of_stacks, Tcsr_list, prim_dict, labels])



def Modify_Metric_and_Points(I, points_tensor, theta_indices):

    [Nx, Ny, Nt] = I.shape

    tol_prox_theta = Nt/5

    for i in range(points_tensor.shape[0]):

        if points_tensor[i,2] == 2:
            theta_ind_prox = I[[points_tensor[i,1].long(),points_tensor[i,0].long()]].topk(Nt).indices
            indic_prox_theta = torch.abs(theta_ind_prox-theta_indices[i])>tol_prox_theta

            theta_ind = theta_ind_prox[indic_prox_theta][0]
            theta_indices = torch.cat([theta_indices, theta_ind.unsqueeze(0)], dim=0)

            points_tensor = torch.cat([points_tensor, points_tensor[i,:].unsqueeze(0)], dim=0)

        if points_tensor[i,2] == 1:
            I[points_tensor[i,1].long(),points_tensor[i,0].long(),:] = 1.0

    print('nombre de points : {}'.format(points_tensor.shape[0]))

    theta_indices = theta_indices.long()

    return([I, points_tensor, theta_indices])

def Show_Tree(Tcsr_element, labels, prim_dict):

    T = Tcsr_element
    G = nx.from_scipy_sparse_array(T)
    lab = [(i,'{}'.format(prim_dict[labels==0][i])) for i in range(len(prim_dict[labels==0]))]
    plt.figure(7)
    plt.clf()
    nx.draw_planar(G,labels=dict(lab),with_labels=True,node_size=300)
    plt.show()
    return(None)

def Detection_Model(model, val_batch, threshold=0.05):
    if val_batch['image'].ndim==2:
        y = model(val_batch['image'].unsqueeze(0).unsqueeze(0))
    else:
        y = model(val_batch['image'].unsqueeze(0))

    local_max_filt = torch.nn.MaxPool2d(17, stride=1, padding=8)

    max_output = local_max_filt(y)
    detected_points = ((max_output==y)*(y>threshold)).permute([0,2,3,1]).nonzero()[:,1:]
    
    return(detected_points)

def Show_Curves(I, im_tensor, points_tensor, list_of_stacks=[], show_metric=False):
    plt.figure(4)
    plt.clf()
    # plt.imshow(I.max(dim=2).values, cmap="gray")
    if im_tensor.ndim==3:
        if show_metric:
            plt.imshow(I.max(dim=2).values.T, cmap="gray")
        else:
            plt.imshow(im_tensor.permute([1,2,0]), cmap="gray")
    else:
        if show_metric:
            plt.imshow(I.max(dim=2).values.T, cmap="gray")
        else:
            plt.imshow(im_tensor, cmap="gray")
        
    [Nx,Ny, Nt] = I.shape
    
    # colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    
    N_points = points_tensor.shape[0]
    for i in range(len(list_of_stacks)):
        stacked_list = list_of_stacks[i]
        
        plt.scatter(Ny*stacked_list[0]/(Ny/Nx), Nx*stacked_list[1], marker='.', s=50, alpha=.3)
    plt.scatter(points_tensor[:,1], points_tensor[:,0], marker='.', s=100, c='r')
    # plt.title('result with factor in the direction theta {}'.format(theta_cost))
    # for i in range(N_points):
    #         plt.annotate('{}'.format(i),(points_tensor[i,1]+5, max(points_tensor[i,0]+25,0)), c='g')

    plt.axis('off')
    plt.tight_layout()

    plt.show()
    return(None)

#%%
if __name__ == '__main__':
    
    data = 'IOSTAR'
    np.random.seed(82)

    if data=='synthetic':
        validation_dataset = ULMDataset(root_dir =  './data_synthetic/test_images', transform=transforms.Compose([RandomCrop(1024), GlobalContrastNormalization(), HeatMap(s=9, alpha=3, out_channels = 4), ToTensor()])) 
        batch = validation_dataset[0]
        im_tensor = batch['image']
        
        model = ULM_UNet(in_channels=1, init_features=48, threshold = 0.05, out_channels = 3)
        model.load_state_dict(torch.load('./weights/ulm_net_synthetic_epochs_2000_batch_1_out_channels_3_30_5.pt'))
        Nt = 64
        
        points = Detection_Model(model, batch, threshold=0.5)
        
        points_tensor = torch.tensor(points).long()
        # points_tensor = batch['landmarks'][ (batch['landmarks']**2).sum(dim=-1)>0,:]
        
        batch['landmarks'] = points_tensor[points_tensor[:,2]!=3,:].numpy()
        batch['image'] = batch['image'].numpy()
        
        print(batch['landmarks'].shape[0])
        print((batch['landmarks'][:,2]==2).sum())
        
        batch_transform = transforms.Compose([ Rescale(512), HeatMap(s=9, alpha=3, out_channels = 3), ToTensor()])
        
        batch_rescaled = batch_transform(batch)
        
        im_tensor, points_tensor = batch_rescaled['image'], batch_rescaled['landmarks'][:,:].long()
        
        pil_to_tensor = torchvision.transforms.ToTensor()
    
        # lifted_im_array = (gaussian_OS(im_tensor.T, sigma = 0.001, eps = 0.05, N_o = Nt))   
        lifted_im_array = (gaussian_OS(im_tensor.T, sigma = 0.005, eps = 0.1, N_o = Nt))
        
        plt.figure(0)
        plt.imshow(im_tensor.double(), cmap='gray', vmin=0, vmax=1)
        plt.scatter(points_tensor[:,1], points_tensor[:,0], c=points_tensor[:,2])
        plt.show()
    
        
    elif data=='IOSTAR':
        validation_dataset = IOSTARDataset(root_dir =  './data_IOSTAR/test_images', transform=transforms.Compose([RandomCrop(512), GlobalContrastNormalization(), HeatMap(s=9, alpha=3, out_channels = 4), ToTensor(), Padding(0)])) 
        batch = validation_dataset[0]
        
        im_tensor = batch['image']
        
        model = ULM_UNet(in_channels=3, init_features=64, threshold = 0.1, out_channels = 4)
        model.load_state_dict(torch.load('./weights/ulm_net_IOSTAR_epochs_1000_size_256_batch_4_out_channels_4_alpha_3.555774513043065_18_9_NoEndpoints_0.pt'))
        Nt = 64
        
        points = Detection_Model(model, batch, threshold=0.3)
        
        points_tensor = torch.tensor(points).long()
        # points_tensor = batch['landmarks'][ (batch['landmarks']**2).sum(dim=-1)>0,:]
        
        batch['landmarks'] = points_tensor[points_tensor[:,2]!=3,:].numpy()
        batch['image'] = batch['image'].numpy()
        
        print(batch['landmarks'].shape[0])
        print((batch['landmarks'][:,2]==2).sum())
        
        batch_transform = transforms.Compose([Rescale(256), HeatMap(s=13, alpha=3.55, out_channels = 4), ToTensor(), Padding(0)])
        
        batch_rescaled = batch_transform(batch)
        
        im_tensor, points_tensor = batch_rescaled['image'], batch_rescaled['landmarks'][:,:].long()
        
        # grey_levels_im_tensor = (1.-(im_tensor**2).mean(dim=0).sqrt())*(im_tensor.mean(dim=0)>1e-1)
        A = np.array([frangi(im_tensor[0].numpy(), beta=0.1), frangi(im_tensor[1].numpy(), beta=0.1), frangi(im_tensor[2].numpy(), beta=0.1)])*(im_tensor.mean(dim=0)>0.15).numpy()
        A = torch.tensor(A/A.max(axis=(1,2), keepdims=True)).mean(dim=0).sqrt()
        pil_to_tensor = torchvision.transforms.ToTensor()

        lifted_im_array = gaussian_OS(A.T, sigma = 0.01, eps = 0.08, N_o = Nt)
        plt.figure(0)
        plt.imshow(im_tensor.double().permute([1,2,0]), cmap='gray', vmin=0, vmax=1)
        plt.scatter(points_tensor[:,1], points_tensor[:,0], c=points_tensor[:,2])
        plt.show()
        
    elif data=='ULM':
        validation_dataset = ULMDataset(root_dir =  './data/test_images', transform=transforms.Compose([ GlobalContrastNormalization(), HeatMap(s=9, alpha=3, out_channels = 4), ToTensor(), Padding(0)])) 
        batch = validation_dataset[1]
        
        # im_tensor = batch['image']
        
        model = ULM_UNet(in_channels=1, init_features=48, threshold = 0.1, out_channels = 3)
        model.load_state_dict(torch.load('./weights/ulm_net_ULM_epochs_1500_batch_1_out_channels_3_7_6.pt'))
        Nt = 64
        
        # p = making_ULM_halfleft_rat_brain2D_and_orientation(N_t=Nt, scaling_effect=0.4)
        p = making_ULM_bolus_full_rat_brain2D_and_orientation(N_t = Nt, scaling_effect = 0.4)

        #%%
        
        np.random.seed(3)
        crop_transform = transforms.Compose([ToArray(), RandomCrop(800), HeatMap(s=13, alpha=3, out_channels = 4), ToTensor(), Padding(128)])
        
        batch_crop_input = batch.copy()
        batch_crop_input['image'] = torch.cat([batch['image'].unsqueeze(0), p[:,1605:,:].permute([2,0,1])], dim=0)
        
        batch_cropped = crop_transform(batch_crop_input)

        image_cropped_stack = batch_cropped['image']
        im_lifted_cropped = batch_cropped['image'][1:]
        batch_cropped['image'] = batch_cropped['image'][0] # return to single channel to apply model
        
        points = Detection_Model(model, batch_cropped, threshold=0.05)
        
        points_tensor = torch.tensor(points).long()
        # points_tensor = batch_cropped['landmarks'][ (batch_cropped['landmarks']**2).sum(dim=-1)>0,:]
        
        batch_cropped['landmarks'] = points_tensor[points_tensor[:,2]!=3,:]
        
        
        print(batch_cropped['landmarks'].shape[0])
        print((batch_cropped['landmarks'][:,2]==2).sum())
        
        
        batch_cropped['image'] = image_cropped_stack # return to stacking with lifted image for rescaling
        batch_transform = transforms.Compose([Padding(-128), ToArray(), Rescale(512), HeatMap(s=13, alpha=3, out_channels = 4), ToTensor()])
        
        batch_rescaled = batch_transform(batch_cropped)
        
        im_tensor, lifted_im_array, points_tensor = batch_rescaled['image'][0], batch_rescaled['image'][1:].permute([1,2,0]).numpy(), batch_rescaled['landmarks'].long()
        
        pil_to_tensor = torchvision.transforms.ToTensor()
        
        plt.figure(0)
        plt.imshow((lifted_im_array.mean(axis=2))**0.25, cmap='gray', vmin=0, vmax=1)
        plt.scatter(points_tensor[:,1], points_tensor[:,0]) #, c=points_tensor[:,2])
        plt.show()
    
        
        # import napari
        # # napari.view_image(W.numpy())
        # viewer = napari.view_image(lifted_im_array)
        # # viewer.add_points(points_ar1ray_3d[:,[1,0,2]], face_color = 'r', size = 3)
        # napari.run()
    #%%
    
    if data=='ULM':
        theta_indices = torch.tensor(lifted_im_array)[[points_tensor[:,0].long(),points_tensor[:,1].long()]].argmax(dim=1)
    else:
        theta_indices = torch.tensor(lifted_im_array)[[points_tensor[:,1].long(),points_tensor[:,0].long()]].argmax(dim=1)
    
    shift = 1e-3
    [A, points_tensor_mod, theta_indices_mod] = Modify_Metric_and_Points(torch.tensor(lifted_im_array).permute([1,0,2]), points_tensor, theta_indices)
    # [A, points_tensor_mod, theta_indices_mod] = Modify_Metric_and_Points(torch.tensor(lifted_im_array), points_tensor, theta_indices)

    
    # W = np.sqrt(A*(A>0))*(np.sqrt(A*(A>0))>0.5)+0. # for synthetic
    # W = np.sqrt(A*(A>0)*(np.sqrt(A*(A>0))>0.15)+0.)
    W = A.sqrt()*(A.sqrt()>0.15) + 0. # for  ULM
    
    points_array_3d = np.hstack([points_tensor_mod[:,:-1], theta_indices_mod.unsqueeze(1)])
    
    #%% Distance computation
    
    [D, L, hfmIn] = Compute_Distance_Matrix(1/(W+shift), points_tensor_mod, theta_indices_mod, alpha=0.05, xi=0.08/(np.pi))     
    
    #%% Visualization

    curves, list_of_stacks, Tcsr_list, prim_dict, labels = Cluster_from_Distance(D, L, distance_threshold = 5)
    Show_Curves(W, im_tensor, points_tensor_mod, list_of_stacks, show_metric=False)
    Show_Tree(Tcsr_list[0], labels, prim_dict)
    
    #%%
    import napari

    points_array_3d = np.hstack([points_tensor_mod[:,:-1], theta_indices_mod.unsqueeze(1)])

    # [Nx,Ny] = im_tensor.shape[1:]
    viewer = napari.view_image((W).numpy())
    viewer.add_points(points_array_3d[:,[1,0,2]], face_color = 'r', size = 3)
    for i in range(len(list_of_stacks)):
        stacked_list = list_of_stacks[i]
        viewer.add_points(hfmIn.IndexFromPoint(stacked_list[:,::5].T)[0], face_color='b', size=1)
