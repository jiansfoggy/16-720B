import numpy as np
import cv2
import matplotlib.pyplot as plt

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    # cv2.imwrite('../results/gaussian_pyramid.png', im_pyramid)
    # plt.imsave('../results/pc_gaussian_pyramid.png', im_pyramid, cmap='gray')
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def displayKeypoints(im, locsDoG):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    implot = plt.imshow(im,cmap='gray')
    x_points = locsDoG[:,0]
    y_points = locsDoG[:,1]
    fig = plt.scatter(x_points, y_points, c='r', s=20)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig("../results/keyPoints.png", bbox_inches='tight')
    plt.show()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    ################
    # TO DO ...
    # compute DoG_pyramid here

    DoG_pyramid = []
    DoG_levels = levels[1:]
    for i in range(1, len(DoG_levels)+1):
        DoG_pyramid.append(gaussian_pyramid[:,:,i] - gaussian_pyramid[:,:,i-1])

    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = np.zeros((DoG_pyramid.shape[0], DoG_pyramid.shape[1], DoG_pyramid.shape[2]))
    ##################
    # TO DO ...
    # Compute principal curvature here
    for i in range(0, DoG_pyramid.shape[2]):
        sobelx = cv2.Sobel(DoG_pyramid[:,:,i],cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(DoG_pyramid[:,:,i],cv2.CV_64F,0,1,ksize=3)

        sobelxx = cv2.Sobel(sobelx,cv2.CV_64F,1,0,ksize=3)
        sobelyy = cv2.Sobel(sobely,cv2.CV_64F,0,1,ksize=3)
        sobelxy = cv2.Sobel(sobelx,cv2.CV_64F,0,1,ksize=3)
        sobelyx = cv2.Sobel(sobely,cv2.CV_64F,1,0,ksize=3)
        traceH = np.square(np.add(sobelxx,sobelyy))
        detH = np.subtract(np.multiply(sobelxx, sobelyy), np.multiply(sobelxy,sobelyx))
        principal_curvature[:,:,i] = np.divide(traceH, detH)
        principal_curvature[:,:,i] = np.nan_to_num(principal_curvature[:,:,i],0)
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    imh, imw, iml = DoG_pyramid.shape
    extremaTensor = np.zeros((11,imh,imw,iml))
    for layer in range(0, iml):
        temp_pyramid = np.pad(DoG_pyramid[:,:,layer],(1,1),mode='constant',constant_values=0)
        extremaTensor[0,:,:,layer] = np.roll(temp_pyramid,1,axis=1)[1:-1,1:-1] #right
        extremaTensor[1,:,:,layer] = np.roll(temp_pyramid,-1,axis=1)[1:-1,1:-1] #left
        extremaTensor[2,:,:,layer] = np.roll(temp_pyramid,1,axis=0)[1:-1,1:-1] #down
        extremaTensor[3,:,:,layer] = np.roll(temp_pyramid,-1,axis=0)[1:-1,1:-1] #up
        extremaTensor[4,:,:,layer] = np.roll(np.roll(temp_pyramid, 1, axis=1),1,axis=0)[1:-1,1:-1] #right,down
        extremaTensor[5,:,:,layer] = np.roll(np.roll(temp_pyramid, -1, axis=1),1,axis=0)[1:-1,1:-1] #left,down
        extremaTensor[6,:,:,layer] = np.roll(np.roll(temp_pyramid, -1, axis=1),-1,axis=0)[1:-1,1:-1] #left,up
        extremaTensor[7,:,:,layer] = np.roll(np.roll(temp_pyramid, 1, axis=1),-1,axis=0)[1:-1,1:-1] #right,up
        if layer == 0:
            extremaTensor[9,:,:,layer] = DoG_pyramid[:,:,layer+1] #layer above
        elif layer == iml-1:
            extremaTensor[8,:,:,layer] = DoG_pyramid[:,:,layer-1] #layer below
        else:
            extremaTensor[8,:,:,layer] = DoG_pyramid[:,:,layer-1] #layer below
            extremaTensor[9,:,:,layer] = DoG_pyramid[:,:,layer+1] #layer above
        extremaTensor[10,:,:,layer] = DoG_pyramid[:,:,layer]

    extremas = np.argmax(extremaTensor, axis=0)
    extremaPoints = np.argwhere(extremas==10)
    locsDoG = []

    for point in extremaPoints:
        if np.absolute(DoG_pyramid[point[0],point[1],point[2]]) > th_contrast and principal_curvature[point[0],point[1],point[2]] < th_r:
            point = [point[1], point[0], point[2]]
            locsDoG.append(point)

    locsDoG = np.stack(locsDoG, axis=-1)
    locsDoG = locsDoG.T
    return locsDoG

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    gauss_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyramid, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    principal_curvature = computePrincipalCurvature(DoG_pyramid)
    locsDoG = getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,th_contrast,th_r)
    return locsDoG, gauss_pyramid

if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    # im_pyr = createGaussianPyramid(im)
    # displayPyramid(im_pyr)
    # test DoG pyramid
    # DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    # displayPyramid(DoG_pyr)
    # test compute principal curvature
    # pc_curvature = computePrincipalCurvature(DoG_pyr)
    # displayPyramid(pc_curvature)
    # test get local extrema
    # th_contrast = 0.03
    # th_r = 12
    # locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)
    displayKeypoints(im, locsDoG)