import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches
import matplotlib.pyplot as plt

def warpH(im, H2to1):
    out_size = (2000,800)
    imWarped = cv2.warpPerspective(im, H2to1, out_size)
    imWarped = np.uint8(imWarped)
    homorgrahpy_file = '../results/q6_1.npy'
    np.save(homorgrahpy_file, H2to1)
    return imWarped

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...

    out_size = (1200,800)
    im1h, imw1 = im1.shape[:2]
    im2Warped = cv2.warpPerspective(im2, H2to1, out_size)
    im1Warped = cv2.warpPerspective(im1, np.identity(3), out_size)

    mask1 = distance_transform_edt(im1Warped)
    mask2 = distance_transform_edt(im2Warped)

    result1 = np.multiply(im1Warped,mask1)
    result2 = np.multiply(im2Warped,mask2)

    pano_im = np.divide(np.add(result1, result2), np.add(mask1, mask2))
    pano_im = np.nan_to_num(pano_im,0)
    pano_im = np.uint8(pano_im)
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...

    imh1, imw1, imd1 = im1.shape
    imh2, imw2, imd2 = im2.shape

    corners = np.array([[0,imw2,0,imw2],[0,0,imh1,imh1],[1,1,1,1]])
    warpedCorners = np.matmul(H2to1, corners)
    warpedCorners = warpedCorners/warpedCorners[2,:]
    warp_corner = np.ceil(warpedCorners)

    row1 = im1.shape[0]
    col1 = im2.shape[1]
    maxrow = max(row1,max(warp_corner[1,:]))
    minrow = min(1,min(warp_corner[1,:]))
    maxcol = max(col1,max(warp_corner[0,:]))
    mincol = min(1,min(warp_corner[0,:]))

    scale = (maxcol-mincol)/(maxrow-minrow)

    W_out = 1200
    height = W_out/scale
    out_size = (W_out, int(round(height)+1))

    s = W_out / (maxcol-mincol)
    scaleM = np.array([[s,0,0],[0,s,0],[0,0,1]])
    transM = np.array([[1,0,0],[0,1,-minrow],[0,0,1]])
    M = np.matmul(scaleM,transM)

    im2Warped = cv2.warpPerspective(im2, np.matmul(M,H2to1), out_size)
    im1Warped = cv2.warpPerspective(im1, M, out_size)

    mask1 = distance_transform_edt(im1Warped)
    mask2 = distance_transform_edt(im2Warped)

    result1 = np.multiply(im1Warped,mask1)
    result2 = np.multiply(im2Warped,mask2)

    pano_im = np.divide(np.add(result1, result2), np.add(mask1, mask2))
    pano_im = np.nan_to_num(pano_im,0)
    pano_im = np.uint8(pano_im)
    return pano_im

def generatePanaroma(img1,img2):
    im1 = cv2.imread(img1)
    im2 = cv2.imread(img2)
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=10000, tol=2)
    return imageStitching_noClip(im1,im2,H2to1)

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=10000, tol=2)
    warpImage = warpH(im2, H2to1)
    pano_im = imageStitching(im1,im2,H2to1)
    pano_im_noClip = imageStitching_noClip(im1, im2, H2to1)
    final_panaroma = generatePanaroma('../data/incline_L.png', '../data/incline_R.png')
    cv2.imwrite('../results/q6_1warp.jpg', warpImage)
    cv2.imwrite('../results/6_1.jpg', pano_im)
    cv2.imwrite('../results/q6_2_pan.jpg', pano_im_noClip)
    cv2.imwrite('../results/q6_3.jpg', final_panaroma)
    # cv2.imshow('panoramas', pano_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()