#%%
import numpy as np
import wandb

import torch
import torchvision
from torchvision import transforms

from agd import Eikonal
from agd.Metrics import Riemann
from agd.LinearParallel import outer_self as Outer # outer product v v^T of a vector with itself

import time

from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

import matplotlib.pyplot as plt

from utils.OrientationScore import gaussian_OS
from utils.dataset import ULMDataset, IOSTARDataset

from nn.ulm_unet import ULM_UNet

from tqdm import tqdm

from datetime import datetime

from skimage.filters import frangi

from skimage import transform

try:
    from utils.transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image, ColorJitter, GlobalContrastNormalization, RandomAffine, Padding, CenterCrop, ToArray
except:
    from transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image, ColorJitter, GlobalContrastNormalization, RandomAffine, Padding, CenterCrop, ToArray
    
import networkx as nx

import sys
# sys.path.append("./../ULM_data")

# from make_ulm_images import making_ULM_halfleft_rat_brain2D_and_orientation, making_ULM_bolus_full_rat_brain2D_and_orientation

from PIL import Image

def Compute_Distance_Matrix(W, points_tensor, theta_indices, alpha = 0.1, xi = 0.1/np.pi):
    """
    Computes the distance matrix between a set of points using the Fast Marching Method (FMM) algorithm.

    Args:
    - W: numpy array of shape (Nx, Ny, Nt) representing the Riemannian metric tensor field.
    - points_tensor: torch tensor of shape (n_points, 3) representing the set of points in 3D space.
    - theta_indices: torch tensor of shape (n_points,) representing the indices of the theta coordinate for each point.
    - alpha: float representing the weight of the first term in the Riemannian metric tensor field.
    - xi: float representing the weight of the third term in the Riemannian metric tensor field.

    Returns:
    - A list containing:
        - D: numpy array of shape (n_points, n_points) representing the distance matrix between the set of points.
        - L: list of length n_points-1 containing the geodesics between consecutive points.
        - hfmIn: dictionary containing the input parameters for the FMM algorithm.
    """
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
        hfmIn['seed'] = np.hstack([points_tensor[j,1]*(Nx/Ny)/Nx, points_tensor[j,0]/Ny, np.pi*theta_indices[j]/Nt]);

        
        hfmIn['tips'] = np.array(torch.hstack([points_tensor[j+1:,1].unsqueeze(1)*(Nx/Ny)/Nx, points_tensor[j+1:,0].unsqueeze(1)/Ny, np.pi*theta_indices[j+1:].unsqueeze(1)/Nt])); 
        hfmIn['stopWhenAllAccepted'] = hfmIn['tips']
        
        a = time.time()
        hfmOut = hfmIn.Run()
        
        D[j,j+1:] = hfmOut['values'][points_tensor[j+1:,1],points_tensor[j+1:,0],theta_indices[j+1:]]
        
        L.append(hfmOut['geodesics'])

    D = D+D.transpose()
    return([D, L, hfmIn])


def Cluster_from_Distance(D, L, distance_threshold=10.):
    """
    Clusters a set of curves based on their pairwise distances.

    Parameters
    ----------
    D : numpy.ndarray
        A square matrix of pairwise distances between curves.
    L : list
        A list of curves.
    distance_threshold : float, optional
        The maximum distance between two curves for them to be in the same cluster.
        Defaults to 10.

    Returns
    -------
    list
        A list of curves, where each curve is a concatenation of the curves in a cluster.
    list
        A list of stacked curves, where each stacked curve is a concatenation of the curves in a cluster.
    list
        A list of minimum spanning trees, one for each cluster.
    numpy.ndarray
        A dictionary that maps the indices of the original curves to the indices of the clusters.
    numpy.ndarray
        A list of cluster labels, one for each curve.
    """
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
    
    print('Nombre de clusters etablis :' +  str(labels.max()+1))

    Tcsr_list = []

    for j in range(labels.max()+1):
        
        ind = (labels==j)
        
        sec_dict = prim_dict[ind]
        
        
        X = csr_matrix(D_norm[ind,:][:,ind])
        
        Tcsr = minimum_spanning_tree(X)
        Tcsr_list.append(Tcsr)
        
        
        Edges_i, Edges_o = Tcsr.toarray().nonzero()
                
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



def Modify_Metric_and_Points(I, points_tensor, theta_indices, decalage = 3):
    """
    Modify the metric and points tensor based on the input image I, points tensor and theta indices.

    Args:
    - I (torch.Tensor): Input image tensor of shape (Nx, Ny, Nt).
    - points_tensor (torch.Tensor): Tensor of shape (n_points, 3) containing the x and y coordinates of each point and its type (1 or 2).
    - theta_indices (torch.Tensor): Tensor of shape (n_points,) containing the theta index for each point.
    - decalage (int): The number of pixels to shift the points in their neighborhood.

    Returns:
    - A list containing:
        - The modified image tensor of shape (Nx, Ny, Nt).
        - The modified points tensor of shape (n_points, 3).
        - The modified theta indices tensor of shape (n_points,).
    """
    [Nx, Ny, Nt] = I.shape

    # Décalage des points dans leur voisinage
    # for i in range(points_tensor.shape[0]):
    #     I_neigh = I[points_tensor[i,1].long()-decalage:points_tensor[i,1].long()+ decalage + 1,points_tensor[i,0].long()-decalage :points_tensor[i,0].long()+decalage + 1,:]
    #     idx = I_neigh.argmax()

    #     u = (I_neigh==I_neigh.reshape(-1)[idx]).nonzero()
    #     points_tensor[i,0] = u[0,1] + points_tensor[i,0]-3
    #     points_tensor[i,1] = u[0,0] + points_tensor[i,1]-3
    #     theta_indices[i] = u[0,2]


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
    """
    Visualize a tree represented as a sparse matrix in Compressed Sparse Row format.

    Parameters:
    -----------
    Tcsr_element : scipy.sparse.csr_matrix
        The sparse matrix representing the tree.
    labels : numpy.ndarray
        An array of integers representing the labels of the nodes in the tree.
    prim_dict : numpy.ndarray
        An array of strings representing the names of the primitives used to build the tree.

    Returns:
    --------
    None
    """
    T = Tcsr_element
    G = nx.from_scipy_sparse_array(T)
    lab = [(i,'{}'.format(prim_dict[labels==0][i])) for i in range(len(prim_dict[labels==0]))]
    plt.figure(7)
    plt.clf()
    nx.draw_planar(G,labels=dict(lab),with_labels=True,node_size=300)
    plt.show(block=False)
    return(None)

def Detection_Model(model, val_batch, threshold=0.05):
    """
    Applies a detection model to a validation batch and returns the coordinates of detected points.

    Args:
        model (torch.nn.Module): The detection model to use.
        val_batch (dict): A dictionary containing the validation batch data.
        threshold (float, optional): The detection threshold. Defaults to 0.05.

    Returns:
        torch.Tensor: A tensor containing the coordinates of detected points.
    """
    if val_batch['image'].ndim==2:
        y = model(val_batch['image'].unsqueeze(0).unsqueeze(0))
    else:
        y = model(val_batch['image'].unsqueeze(0))

    local_max_filt = torch.nn.MaxPool2d(13, stride=1, padding=6)

    max_output = local_max_filt(y)
    detected_points = ((max_output==y)*(y>threshold)).permute([0,2,3,1]).nonzero()[:,1:]
    
    return(detected_points)

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def Show_Curves(I, im_tensor, points_tensor, list_of_stacks=[], data='synthetic', show_metric=False, th_output = True, save_string='output.png'):
    """
    Display curves on an image.

    Args:
    - I (torch.Tensor): 3D tensor representing the image.
    - im_tensor (torch.Tensor): 3D tensor representing the image.
    - points_tensor (torch.Tensor): 2D tensor representing the points to be plotted.
    - list_of_stacks (list): list of stacked points.
    - data (str): name of the data.
    - show_metric (bool): whether to show metric or not.
    - th_output (bool): whether to output threshold or not.

    Returns:
    - None
    """
    plt.figure(4)
    plt.clf()
    if im_tensor.ndim==3:
        if show_metric:
            plt.imshow(I.max(dim=2).values.T, cmap="gray")
        else:
            plt.imshow(im_tensor.permute([1,2,0]), cmap="gray")
    else:
        if show_metric:
            plt.imshow(I.max(dim=2).values, cmap="gray")
        else:
            plt.imshow(im_tensor, cmap="gray")
        
    [Nx,Ny, Nt] = I.shape
        
    N_points = points_tensor.shape[0]
    for i in range(len(list_of_stacks)):
        stacked_list = list_of_stacks[i]
        
        plt.scatter(Ny*stacked_list[0]/(Ny/Nx), Nx*stacked_list[1], marker='.', s=50, alpha=.5)
    plt.scatter(points_tensor[:,1], points_tensor[:,0], marker='.', s=100, c='r')
    # plt.title('result with factor in the direction theta {}'.format(theta_cost))
    # for i in range(N_points):
    #         plt.annotate('{}'.format(i),(points_tensor[i,1]+5, max(points_tensor[i,0]+25,0)), c='g')

    plt.axis('off')
    plt.tight_layout()
    # plt.savefig('./figures/' + data + '/curves_output' + str(0) + str(th_output) + datetime.now().strftime("%H:%M:%S") + '.png')
    plt.savefig(save_string)
    plt.show(block=False)

    return(None)

def mask_from_list_of_stacks(list_of_stacks, image, dist=.01):
    """
    Create a mask from a list of stacks and an image.

    Parameters:
    -----------
    list_of_stacks : list
        A list of stacks.
    image : numpy.ndarray
        An image.
    dist : float, optional
        The distance threshold. Default is 0.01.

    Returns:
    --------
    numpy.ndarray
        The mask.
    """
    
    mask = np.zeros_like(image)

    [X,Y] = np.meshgrid(np.linspace(0,1,image.shape[0]), np.linspace(0,1,image.shape[1]))

    for k in range(len(list_of_stacks)):
        mask = np.maximum(mask, (((X[None,:,:]-list_of_stacks[k][0,::3,None,None])**2 + (Y[None,:,:]-list_of_stacks[k][1,::3,None,None])**2)<dist**2).max(axis=0))

    return mask

def main(args):
    wandb.login()
    wandb.init(project="ULM_4CHANNEL", config=args)
    data = 'IOSTAR'
    np.random.seed(82)

    validation_dataset = IOSTARDataset(root_dir =  './data_IOSTAR/test_images', transform=transforms.Compose([RandomCrop(32*14), HeatMap(s=9, alpha=3, out_channels = 4), ToTensor(), Padding(0)])) 
    batch = validation_dataset[args.nb]

    im_tensor = batch['image']/batch['image'].max()
    batch['image'] = im_tensor

    original_image = im_tensor

    model = ULM_UNet(in_channels=3, init_features=64, threshold = 0.1, out_channels = 4)
    model.load_state_dict(torch.load('./weights/ulm_net_IOSTAR_epochs_1000_size_256_batch_4_out_channels_4_alpha_3.555774513043065_18_9_NoEndpoints_0.pt'))
    Nt = 64

    points = Detection_Model(model, batch, threshold=0.2)

    points_tensor = torch.tensor(points).long()
    # points_tensor = batch['landmarks'][ (batch['landmarks']**2).sum(dim=-1)>0,:]

    batch['landmarks'] = points_tensor[points_tensor[:,2]!=3,:].numpy()

    im_tensor = batch['image']

    im_frangi = frangi(im_tensor.min(axis=0).values.numpy(),sigmas = np.exp(np.linspace(np.log(.001),np.log(1.5),1000)), beta=100, alpha=.5, gamma = 15)**.25
    im_frangi = im_frangi/im_frangi.max() 

    batch['image'] = np.concatenate([torch.tensor(im_frangi).unsqueeze(0).numpy(),original_image])

    print(batch['image'].shape)

    batch_transform = transforms.Compose([ToTensor(), Padding(0), ToArray(), HeatMap(s=13, alpha=3.55, out_channels = 4), Rescale(32*14), ToTensor(), Padding(0)])

    batch_rescaled = batch_transform(batch)

    im_tensor, points_tensor = batch_rescaled['image'][0], batch_rescaled['landmarks'][ (batch_rescaled['landmarks']**2).sum(dim=-1)>0,:].long()
    print(points_tensor)

    # pil_to_tensor = torchvision.transforms.ToTensor()

    lifted_im_array = gaussian_OS(im_tensor.squeeze(), sigma = 0.01, eps = 0.1, N_o = Nt)

    theta_indices = torch.tensor(lifted_im_array)[[points_tensor[:,1].long(),points_tensor[:,0].long()]].argmax(dim=1)

    [A, points_tensor_mod, theta_indices_mod] = Modify_Metric_and_Points(torch.tensor(lifted_im_array).permute([1,0,2]), points_tensor, theta_indices, decalage = args.decalage)
    # [A, points_tensor_mod, theta_indices_mod] = Modify_Metric_and_Points(torch.tensor(lifted_im_array), points_tensor, theta_indices)

    W = (A*(A>0)*(np.sqrt(A*(A>0))>0.3)+0.)**2

    # W = A

    points_array_3d = np.hstack([points_tensor_mod[:,:-1], theta_indices_mod.unsqueeze(1)])

    mask_nonzero_points = (W[points_tensor_mod[:,1].long(),points_tensor_mod[:,0].long(),theta_indices_mod]>0)

    points_tensor_mod = points_tensor_mod[mask_nonzero_points,:]
    theta_indices_mod = theta_indices_mod[mask_nonzero_points]

    #%% Distance computation

    [D, L, hfmIn] = Compute_Distance_Matrix(1/(1+1000*W**args.power), points_tensor_mod, theta_indices_mod, alpha=args.eps, xi=args.xi/(np.pi))     

    #%% Visualization

    curves, list_of_stacks, Tcsr_list, prim_dict, labels = Cluster_from_Distance(D, L, distance_threshold = 0.08)
    Show_Curves(W, batch_rescaled['image'][1:], points_tensor_mod, list_of_stacks, show_metric=True, save_string = './figures/IOSTAR_output/output_IOSTAR_512_' + str(args.nb) + '.png')
    # Show_Tree(Tcsr_list[0], labels, prim_dict)

    # plt.close('all')

    # mask_th = transform.resize(np.array(Image.open('./data_IOSTAR/test_images/GT_test/IOSTAR_GT_31.tif')),(512,512))

    # mask_th = mask_th/mask_th.max()

    # mask_from_tree = mask_from_list_of_stacks(list_of_stacks, mask_th, dist=args.dist)

    # wandb.log({'Ground Truth':wandb.Image(mask_th), 'Output Mask':wandb.Image(mask_from_tree)})

    # plt.figure()
    # plt.imshow(mask_from_tree)

    # plt.figure()
    # plt.imshow(mask_th)
    # plt.show()


    # Seg_score = (mask_from_tree*mask_th).sum()/np.maximum(mask_from_tree,mask_th).sum()

    # wandb.log({'Segmentation score': Seg_score})
    # print(Seg_score)

    # import napari

    # points_array_3d = np.hstack([points_tensor_mod[:,:-1], theta_indices_mod.unsqueeze(1)])
    # viewer = napari.view_image((W).numpy())
    # viewer.add_points(points_array_3d[:,[1,0,2]], face_color = 'r', size = 3)
    # for i in range(len(list_of_stacks)):
    #     stacked_list = list_of_stacks[i]
    #     viewer.add_points(hfmIn.IndexFromPoint(stacked_list[:,::5].T)[0], face_color='b', size=1)

# %%
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Compute the HFM distance matrix between a set of points.')
    parser.add_argument('--nb', type=int, default=20, help='number of the image in the dataset.')
    parser.add_argument('--threshold_landmarks', type=float, default=0.4, help='thershold for detection of landmarks')
    parser.add_argument('--threshold_metric', type=float, default=0.2, help='thershold for metric cost definition')
    parser.add_argument('--threshold_tree', type=float, default=0.05, help='thershold for metric cost definition')
    parser.add_argument('--xi', type=float, default=1., help='thershold for metric cost definition')
    parser.add_argument('--eps', type=float, default=.5, help='thershold for metric cost definition')
    parser.add_argument('--power', type=float, default=2., help='thershold for metric cost definition')
    parser.add_argument('--bool_erase_far', type=int, default=1, help='thershold for metric cost definition')
    parser.add_argument('--dist', type=float, default=.01, help='thershold for metric cost definition')
    parser.add_argument('--decalage', type=int, default=0, help='thershold for metric cost definition')
    parser.add_argument('--beta', type=float, default=.1, help='thershold for metric cost definition')



    args = parser.parse_args()

    main(args)