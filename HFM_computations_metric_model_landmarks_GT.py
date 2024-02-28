#%%
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

from datetime import datetime

from skimage.filters import frangi

try:
    from utils.transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image, ColorJitter, GlobalContrastNormalization, RandomAffine, Padding, CenterCrop, ToArray
except:
    from transforms import Rescale, RandomCrop, ToTensor, HeatMap, Rescale_image, ColorJitter, GlobalContrastNormalization, RandomAffine, Padding, CenterCrop, ToArray
    
import networkx as nx

import sys
sys.path.append("./../ULM_data")

from multiprocessing import Pool, freeze_support, cpu_count

from PIL import Image
from scipy.io import loadmat

from matplotlib.pyplot import cm

naming = 'IOSTAR_model_landmarks_GT_cost_' + datetime.now().strftime('%m_%d_%H')
os.makedirs('./figures/' + naming, exist_ok=True)
# from make_ulm_images import making_ULM_halfleft_rat_brain2D_and_orientation, making_ULM_bolus_full_rat_brain2D_and_orientation


def FM_iteration(hfmInput, points_tensor, theta_indices, j, Nx, Ny, Nt):

    hfmIn = Eikonal.dictIn(hfmInput)
    hfmIn['seed'] = np.hstack([points_tensor[j,1]*(Nx/Ny)/Nx, points_tensor[j,0]/Ny, np.pi*theta_indices[j]/Nt]);

        
    hfmIn['tips'] = np.array(torch.hstack([points_tensor[j+1:,1].unsqueeze(1)*(Nx/Ny)/Nx, points_tensor[j+1:,0].unsqueeze(1)/Ny, np.pi*theta_indices[j+1:].unsqueeze(1)/Nt])); 
    hfmIn['stopWhenAllAccepted'] = hfmIn['tips']
    
    hfmOut = hfmIn.Run()

    vector_D = np.zeros((points_tensor.shape[0]))
    vector_D[j+1:] = hfmOut['values'][points_tensor[j+1:,1],points_tensor[j+1:,0],theta_indices[j+1:]]
    
    return(vector_D,hfmOut['geodesics'])

def wrapped_FM_iteration(args):
    return FM_iteration(*args)


from tqdm import tqdm
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

    W = np.concatenate([W[:,:,Nt//2:], W[:,:,:Nt//2]], axis=2)
    theta_indices = np.mod(theta_indices + Nt//2,Nt)

    ReedsSheppMetric = Riemann( # Riemannian metric defined by a positive definite tensor field
        Outer([np.cos(Theta+np.pi/2), np.sin(Theta+np.pi/2),zero])
        + alpha**(-2)*Outer([-np.sin(Theta+np.pi/2),np.cos(Theta+np.pi/2),zero])
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
    hfmIn['geodesicStep'] = 0.1

    n_points = points_tensor.shape[0]

    L = []

    D = np.zeros((n_points,n_points))
    
    # points_array_3d = np.hstack([points_tensor[:,:-1], theta_indices.unsqueeze(1)])

    pool = Pool(cpu_count()//2)

    results = pool.map(wrapped_FM_iteration, ([(hfmIn.store, points_tensor, theta_indices, j, Nx, Ny, Nt) for j in range(n_points-1)]))    
    
    for j in range(n_points-1):
        [vector_D, geodesics] = results[j]
        D[j+1:,j] = vector_D[j+1:]
        L.append(geodesics)

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
                    
            # stacked_list = np.hstack(local_list)
            # list_of_stacks.append(stacked_list)
            list_of_stacks.append(local_list)
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
    for i in range(points_tensor.shape[0]):
        I_neigh = I[max(points_tensor[i,1].long()-decalage,0):min(points_tensor[i,1].long()+ decalage + 1,I.shape[1]),max(points_tensor[i,0].long()-decalage,0) :min(points_tensor[i,0].long()+decalage + 1,I.shape[0]),:]
        idx = I_neigh.argmax()

        u = (I_neigh==I_neigh.reshape(-1)[idx]).nonzero()
        points_tensor[i,0] = u[0,1] + points_tensor[i,0]-decalage
        points_tensor[i,1] = u[0,0] + points_tensor[i,1]-decalage
        theta_indices[i] = u[0,2]


    tol_prox_theta = Nt//4

    for i in range(points_tensor.shape[0]):

        if points_tensor[i,2] == 2:
            theta_ind_prox = I[[points_tensor[i,1].long(),points_tensor[i,0].long()]].topk(Nt).indices
            # indic_prox_theta = torch.abs(torch.mod(theta_ind_prox-theta_indices[i],Nt//2))>tol_prox_theta #Differences in indices are between 0 and Nt/2
            # indic_prox_theta = torch.abs((Nt//2) - torch.abs((theta_ind_prox-theta_indices[i]) -(Nt//2)))>tol_prox_theta #Differences in indices are between 0 and Nt/2
            indic_prox_theta = torch.minimum(torch.abs((theta_ind_prox-theta_indices[i])),torch.abs((theta_ind_prox-theta_indices[i]) -Nt))>tol_prox_theta #Differences in indices are between 0 and Nt/2

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
    plt.show()
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
        y = model(torch.tensor(val_batch['image']).unsqueeze(0).unsqueeze(0))
    else:
        y = model(torch.tensor(val_batch['image'][-3:]).unsqueeze(0))

    local_max_filt = torch.nn.MaxPool2d(9, stride=1, padding=4)

    max_output = local_max_filt(y)
    detected_points = ((max_output==y)*(y>threshold)).permute([0,2,3,1]).nonzero()[:,1:]
    
    return(detected_points)

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def Show_Curves(I, im_tensor, points_tensor, list_of_stacks=[], data='synthetic', show_metric=False, img_number=0, dist = 0.):
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

    color = iter(cm.rainbow(np.linspace(0, 1, len(list_of_stacks))))

    N_points = points_tensor.shape[0]
    for j in range(len(list_of_stacks)):
        stacked_list = list_of_stacks[j]
        c = next(color)
        for i in range(len(stacked_list)):
            # plt.scatter(Ny*stacked_list[0]/(Ny/Nx), Nx*stacked_list[1], marker='.', s=50, alpha=.5)
            plt.plot(Ny*stacked_list[i][0]/(Ny/Nx), Nx*stacked_list[i][1],c=c, linestyle='--', alpha=1.)
    plt.scatter(points_tensor[:,1], points_tensor[:,0], marker='.', s=100, c='r')
    # plt.title('result with factor in the direction theta {}'.format(theta_cost))
    # for i in range(N_points):
    #         plt.annotate('{}'.format(i),(points_tensor[i,1]+5, max(points_tensor[i,0]+25,0)), c='g')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./figures/' + naming + '/img' + str(img_number) + '_dist_' + str(dist)+ '.png')
    # plt.show()

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

    [X,Y] = np.meshgrid(np.linspace(0,1,image.shape[0]), np.linspace(0,1,image.shape[1]), indexing='xy')

    for k in range(len(list_of_stacks)):
        for j in range(len(list_of_stacks[k])):
            mask = np.maximum(mask, (((X[None,:,:]-list_of_stacks[k][j][0,:,None,None])**2 + (Y[None,:,:]-list_of_stacks[k][j][1,:,None,None])**2)<dist**2).max(axis=0))

    return mask


def test_mask_with_multiple_widths(list_of_stacks, image, target_mask, widths):
    
    mask = np.zeros_like(image)

    [X,Y] = np.meshgrid(np.linspace(0,1,image.shape[0]), np.linspace(0,1,image.shape[1]), indexing='xy')

    for k in range(len(list_of_stacks)):
        for j in range(len(list_of_stacks[k])):
            mask = np.maximum(mask, (((X[None,:,:]-list_of_stacks[k][j][0,::4,None,None])**2 + (Y[None,:,:]-list_of_stacks[k][j][1,::4,None,None])**2)[None,...]<widths[:,None,None,None]**2).max(axis=1))
    
    array_F1 = 2*(mask*(target_mask[None,...])).sum(axis=(1,2))/(mask.sum(axis=(1,2))+target_mask.sum())
    
    best_mask = mask[array_F1.argmax()]
    return array_F1, best_mask

def test_mask_with_multiple_widths_individual_curves(list_of_stacks, image, target_mask, widths):
    
    mask_tot = np.zeros_like(image)

    [X,Y] = np.meshgrid(np.linspace(0,1,image.shape[0]), np.linspace(0,1,image.shape[1]), indexing='xy')

    for k in range(len(list_of_stacks)):
        for j in range(len(list_of_stacks[k])):
            mask = np.maximum(mask_tot[None,:],(((X[None,:,:]-list_of_stacks[k][j][0,::4,None,None])**2 + (Y[None,:,:]-list_of_stacks[k][j][1,::4,None,None])**2)[None,...]<widths[:,None,None,None]**2).max(axis=1))
            array_F1 = 2*(mask*(target_mask[None,...])).sum(axis=(1,2))/(mask.sum(axis=(1,2))+target_mask.sum())
            
            mask_tot = np.maximum(mask_tot, mask[array_F1.argmax()])
    F1_tot = 2*(mask_tot*(target_mask)).sum()/(mask_tot.sum()+target_mask.sum())
    return F1_tot, mask_tot


def score_computation_with_distance_and_widths(target_mask, widths, distance_threshold):
    curves, list_of_stacks, Tcsr_list, prim_dict, labels = Cluster_from_Distance(D, L, distance_threshold = distance_threshold)
    mask_th = target_mask
    mask_th = mask_th/mask_th.max()

    widths = np.logspace(-3,-2,40)
    F1_array, best_mask = test_mask_with_multiple_widths(list_of_stacks, mask_th, mask_th, widths)

    print("best seg score : {}".format(F1_array.max()), 'best width is : {}'.format(widths[F1_array.argmax()]))
    return F1_array.max(), best_mask

def score_computation_with_distance_and_widths_wrapped(args):
    return score_computation_with_distance_and_widths(*args)

import os
img_names = sorted(os.listdir('./data_IOSTAR/test_with_GT/images/'))
GT_names = sorted(os.listdir('./data_IOSTAR/test_with_GT/GT/'))
landmarks_name = sorted(os.listdir('./data_IOSTAR/test_with_GT/landmarks/'))

data = 'IOSTAR'
np.random.seed(0)


target_size = 512

# validation_dataset = IOSTARDataset(root_dir =  './data_IOSTAR/test_with_GT/images', transform=transforms.Compose([RandomCrop(target_size), HeatMap(s=9, alpha=3, out_channels = 4), ToTensor(), Padding(0)])) 
# batch = validation_dataset[3]

batch_F1 = []

for img_idx in range(1):
    input_image = transforms.CenterCrop(512)(torch.tensor(np.array(Image.open('./data_IOSTAR/test_with_GT/images/' + img_names[img_idx]))).permute([2,0,1]).unsqueeze(0)/255)
    GT = transforms.CenterCrop(512)(torch.tensor(np.array(Image.open('./data_IOSTAR/test_with_GT/GT/' + GT_names[img_idx]))))

    batch = {'image': input_image[0], 'landmarks': np.zeros((1,3)), 'heat_map':torch.zeros_like(input_image[0])}
    plt.imshow((batch['image']).squeeze().permute([1,2,0]))

    im_tensor = batch['image']
    batch['image'] = im_tensor

    original_image = im_tensor

    model = ULM_UNet(in_channels=3, init_features=64, threshold = 0.3, out_channels = 4)
    model.load_state_dict(torch.load('./weights/ulm_net_IOSTAR_epochs_1000_size_256_batch_4_out_channels_4_alpha_3.555774513043065_18_9_NoEndpoints_0.pt'))
    Nt = 8

    # points_tensor = batch['landmarks'][ (batch['landmarks']**2).sum(dim=-1)>0,:]

    # batch['landmarks'] = points_tensor[points_tensor[:,2]!=3,:].numpy()

    im_tensor = batch['image']

    im_frangi = frangi(im_tensor.min(axis=0).values.numpy(),sigmas = np.exp(np.linspace(np.log(.05),np.log(5),10)), beta=100, alpha=.5, gamma = .5)**0.5
    im_frangi = im_frangi/im_frangi.max() 

    batch['image'] = torch.tensor(np.concatenate([torch.tensor(im_frangi).unsqueeze(0).numpy(), GT.unsqueeze(0),original_image]))

    threshold_landmarks=.2
    points = Detection_Model(model, batch, threshold=threshold_landmarks)

    points_tensor = torch.tensor(points).long()[points[:,2]!=3,:]
    landmarks_frame = loadmat('./data_IOSTAR/test_with_GT/landmarks/'+landmarks_name[img_idx])

    points_tensor = torch.minimum(torch.tensor(np.vstack([np.hstack([landmarks_frame['EndpointPos']-1.,0*np.ones([landmarks_frame['EndpointPos'].shape[0],1])]),
                    np.hstack([landmarks_frame['BiffPos']-1.,np.ones([landmarks_frame['BiffPos'].shape[0],1])]),
                    np.hstack([landmarks_frame['CrossPos']-1.,2.*np.ones([landmarks_frame['CrossPos'].shape[0],1])])])) - torch.tensor([[26,24,0]]),torch.tensor(511))
    # batch['landmarks'] = points_tensor
    batch_transform = transforms.Compose([Padding(0), ToArray(), HeatMap(s=13, alpha=3.55, out_channels = 4), CenterCrop(target_size), ToTensor(), Rescale(512)])

    # points_tensor = torch.tensor(np.vstack([np.hstack([landmarks_frame['EndpointPos']-1.,0*np.ones([landmarks_frame['EndpointPos'].shape[0],1])]),
    #                 np.hstack([landmarks_frame['BiffPos']-1.,np.ones([landmarks_frame['BiffPos'].shape[0],1])]),
    #                 np.hstack([landmarks_frame['CrossPos']-1.,2.*np.ones([landmarks_frame['CrossPos'].shape[0],1])])])) - torch.tensor([[26,24,0]])

    batch['landmarks'] = points_tensor
    batch_rescaled = batch_transform(batch)

    im_tensor, points_tensor = batch_rescaled['image'][0], batch_rescaled['landmarks'][ (points_tensor**2).sum(dim=-1)>0,:].long()
    # print(points_tensor)

    # pil_to_tensor = torchvision.transforms.ToTensor()

    # lifted_im_array = gaussian_OS(im_tensor.squeeze(), sigma = 0.01, eps = 0.1, N_o = Nt)

    # lifted_im_array = np.abs(OS_cakeWavelet(im_tensor.squeeze().numpy(),Nt).real.transpose(1,2,0))
    lifted_im_array = gaussian_OS(torch.tensor(im_tensor).squeeze(), sigma = 0.001, eps = 0.05, N_o = Nt)
    lifted_im_array = lifted_im_array/lifted_im_array.max()

    theta_indices = torch.tensor(lifted_im_array)[[points_tensor[:,1].long(),points_tensor[:,0].long()]].argmax(dim=1)

    [A, points_tensor_mod, theta_indices_mod] = Modify_Metric_and_Points(torch.tensor(lifted_im_array).permute([1,0,2]), points_tensor, theta_indices, decalage=0)
    # [A, points_tensor_mod, theta_indices_mod] = Modify_Metric_and_Points(torch.tensor(lifted_im_array), points_tensor, theta_indices)

    plt.figure(0)
    plt.imshow(im_tensor, cmap='gray', vmin=0, vmax=1)
    plt.scatter(points_tensor[:,1], points_tensor[:,0], c=points_tensor[:,2], alpha=.5)
    plt.show()

    s=0.5
    W = (A*(A>0)*(np.sqrt(A*(A>0))>s)+0.)**2

    points_array_3d = np.hstack([points_tensor_mod[:,:-1], theta_indices_mod.unsqueeze(1)])

    # Distance computation
    # mask_nonzero_points = (W[points_tensor_mod[:,1].long(),points_tensor_mod[:,0].long(),theta_indices_mod]>0)

    # points_tensor_mod = points_tensor_mod[mask_nonzero_points,:]
    # theta_indices_mod = theta_indices_mod[mask_nonzero_points]
    alpha=.5
    xi=1.
    [D, L, hfmIn] = Compute_Distance_Matrix(1/(1+10000*W**2), points_tensor_mod, theta_indices_mod, alpha=alpha, xi=xi)     

    # Visualization
    ndist = 10
    dist_list = np.logspace(-2,-.5,ndist)

    widths = np.logspace(-3,-2,40)

    mask_th = batch_rescaled['image'][1].numpy()

    pool = Pool(cpu_count()//3)

    results = pool.map(score_computation_with_distance_and_widths_wrapped, ([(mask_th, widths, dist_list[j]) for j in range(ndist)]))    

    F1_dist = np.array([results[i][0] for i in range(ndist)])

    distance_threshold = dist_list[np.argmax(F1_dist)]
    best_mask = results[np.argmax(F1_dist)][1]

    curves, list_of_stacks, Tcsr_list, prim_dict, labels = Cluster_from_Distance(D, L, distance_threshold = distance_threshold)
    # Show_Curves(W, batch_rescaled['image'][1:], points_tensor_mod, list_of_stacks, show_metric=False)
    # Show_Tree(Tcsr_list[0], labels, prim_dict)

    # mask_th = batch_rescaled['image'][1].numpy()
    # mask_th = mask_th/mask_th.max()

    # widths = np.logspace(-3,-2,40)
    # F1_array, best_mask = test_mask_with_multiple_widths(list_of_stacks, mask_th, mask_th, widths)

    Show_Curves(W, batch_rescaled['image'][2:], points_tensor_mod, list_of_stacks, show_metric=False, img_number=img_idx, dist=distance_threshold)

    print("best seg score : {}".format(F1_dist.max()), 'best width is : {}'.format(distance_threshold))

    plt.figure(6)
    plt.clf()
    plt.imshow(np.stack([best_mask, mask_th, mask_th*best_mask],axis=2))
    plt.axis('off')
    plt.savefig('./figures/' + naming + '/mask' + str(img_idx) +'_dist' + str(distance_threshold) + '.png')

    print('best distance threshold : {}'.format(distance_threshold))

    batch_F1.append(max(F1_dist))

# %%
print('mean F1 score : {}'.format(np.mean(batch_F1)))

with open('./figures/' + naming + '/variable_file.txt', "w") as variable_file:
    variable_file.write("s : " + str(s) + "\n")
    variable_file.write("alpha : " + str(alpha) + "\n")
    variable_file.write("xi : " + str(xi) + "\n")
    variable_file.write("ndist : " + str(ndist) + "\n")
    variable_file.write("Nt : " + str(Nt) + "\n")
    variable_file.write("threshold_landmarks : " + str(threshold_landmarks) + "\n")
    variable_file.write("F1 mean : " + str(np.mean(batch_F1)) + "\n")