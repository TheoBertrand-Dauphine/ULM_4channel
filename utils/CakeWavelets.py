import numpy as np

from scipy.special import binom, factorial
# from scipy import sign, heaviside

from numpy import sign, heaviside

from scipy.signal import fftconvolve


def OS_cakeWavelet(Im, No=64):
    K = cakeWaveletStack(2*No, Im.shape[0])

    Im_OS = fftconvolve(K,Im[None,:,:],mode='same')
    return .5*(Im_OS[:(2*No)//2] + Im_OS[(2*No)//2:])

def cakeWaveletStack(orientations, dimensions, splineOrder=3, overlapFactor=1,inflectionPoint=.8,mnOrder=8,gaussSigma=8):
    """
    Returns a set of invertible orientation sensitive (cake) wavelets that are designed in the Fourier domain.

    :param orientations: the number of orientations
    :param dimensions: the spatial dimensions nxn
    :param splineOrder: the order of B-splines used in the angular direction
    :param overlapFactor: the amount of overlap between wavelets
    :param inflectionPoint: inflection point of the radial profile
    :param mnOrder: the taylor-polynomial order used in the radial window
    :param gaussSigma: 
    :returns: a 3d array containing the orientation sensitive wavelets
    """
    
    # convert low-frequencies sigma to fourier domain
    dcSigma = (1/gaussSigma)*(dimensions/(2*np.pi))
    
    # design the cake wavelets in the Fourier domain
    waveletsF = cakeWaveletStackFourier(
        orientations,dimensions,splineOrder,overlapFactor,inflectionPoint,mnOrder,dcSigma
    )
    
    # compute the Fourier inverse
    wavelets = np.zeros(waveletsF.shape,dtype=complex)
    for idx, waveletF in enumerate(waveletsF):
        wavelets[idx] = centeredInverseFourier(waveletF)
        
    return wavelets


def cakeWaveletStackFourier(orientations, size, splineOrder, overlapFactor, inflectionPoint, mnOrder, dcStdDev):
    "designs the cake wavelets in the Fourier domain with the given parameters"
    
    # angular resolution
    sPhi = (2*np.pi) / orientations
    sPhiOverlapped = sPhi / overlapFactor
        
    # create empty container for the Fourier wavelets
    wavelet = np.zeros([orientations,size,size])
    
    # create inverse low-frequencies filters
    dcWindow = np.ones([size,size]) - WindowGauss(size,dcStdDev)
    
    # create radial damping 
    mnWindow = MnWindow(size,mnOrder,inflectionPoint)
    
    # angular grid
    angleGrid = PolarCoordinateGridAngular(size)

    # create fourier wavelet for each orientation
    for idx, theta in enumerate(np.linspace(0,2*np.pi,orientations,False)):
        wavelet[idx] = dcWindow * mnWindow * BSplineMatrixFunc(splineOrder,remainderShifted(angleGrid-theta,2*np.pi,-np.pi)/sPhi)/overlapFactor
        
    return wavelet

def remainderShifted(v,d,s):
    return np.remainder(v-s,d)+s
    

def BSplineMatrixFunc(n,polarAngularGrid):
    "Returns a matrix containing radial b-spline profile"

    d = len(polarAngularGrid)
    polarAngularGrid = np.array(polarAngularGrid)
    angularInterval = np.array([[[0. for x in range(d)] for y in range(d) ] for i in range(n+1)])
    resultingSpline = np.array([[[0. for x in range(d)] for y in range(d) ] for i in range(n+1)])
    orders = np.linspace(-n/2,n/2,n+1,True);

    # machine epsilon to shift zeros for heavyside function
    machineEpsilon = np.finfo(float).eps;

    # compute spline and interval for each order
    for j in range(orders.size):
        
        i = orders[j]
        
        # spline function
        spline = np.array([[0. for x in range(d)] for y in range(d)])
        for k in range(n+2):
            spline += binom(n+1,k)*np.power((polarAngularGrid+(n+1)/2-k),n)*np.power(-1,k)*sign(i+(n+1)/2-k)        
        resultingSpline[j] = 1/(2 * factorial(n)) * spline
        
        # create angular interval 
        angularInterval[j] = heaviside((polarAngularGrid-(i-1/2+machineEpsilon)).tolist(),1)*heaviside((-(polarAngularGrid-(i+1/2))).tolist(),1)
        
    # sum all orders 
    return np.sum(angularInterval*resultingSpline ,axis=0)
    
    
def MnWindow(size, n, inflectionPoint):
    "Returns a matrix containing a radial window with a Gaussian profile"

    # initialize rho matrix
    mnWindow = np.zeros((size,size))

    # compute rho matrix
    grid = PolarCoordinateGridRadial(size)
    for i in range(0,size):
        for j in range(0,size):
            rhoMatrix = grid[i,j] * (1/np.sqrt(2*np.power(inflectionPoint*np.floor(size/2),2)/(1+2*n))) 
            for k in range(0,n+1):
                mnWindow[i,j] += np.power(rhoMatrix,2*k)/factorial(k) * np.exp(-np.power(rhoMatrix,2))      

    # return mnwindow
    return mnWindow


def WindowGauss(size, sigma):
    "Returns a matrix containing a Gaussian window"

    # center in the middle, assuming size is odd
    dim = size//2

    # create gaussian 
    grid = np.zeros((size,size))
    for i in range(0,size):
        for j in range(0,size):
            grid[i][j] = np.exp((-np.power(i-dim,2) - np.power(j-dim,2))/(2*np.power(sigma,2))) 

    # Return the Radial Grid 
    return grid


def PolarCoordinateGridRadial(size):
    "Returns matrix with a 0 in the center and growing values outwards"

    # center in the middle, assuming size is odd
    dim = size//2

    # create radial grid
    grid = np.zeros((size,size))
    for i in range(0,size):
        for j in range(0,size):
            grid[i,j] = np.sqrt(np.power(i - dim,2) + np.power(j - dim,2))

    # return the radial grid 
    return np.array(grid)


def PolarCoordinateGridAngular(size):
    "Returns a grid with an angle assigned to each pixel with the rotation point at the center"

    # center in the middle, assuming size is odd
    dim = size//2
    
    # create grid angular grid
    grid = np.zeros((size,size))
    for i in range(0,size):
        for j in range(0,size):
            grid[j,i] = np.arctan2(i-dim,j-dim)

    # return the angular grid
    return grid


def centeredInverseFourier(centeredFourierImage):
    "computed the centered inverse fourier transform of a 2d image"
    
    # pixels from center to corner
    center = centeredFourierImage.shape[-1]//2
    
    # translate the zero frequency to the left top corner
    fourierImage = np.roll(centeredFourierImage,(-center,-center),(0,1))
    
    # compute the inverse Fourier transform
    image = np.fft.ifft2(fourierImage)
    
    # translate the spatial image centerpoint
    return np.roll(image,(center,center),(0,1))

    
