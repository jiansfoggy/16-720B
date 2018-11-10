import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    A = np.zeros((2*p1.shape[1],9))
    p1 = p1.T
    p2 = p2.T
    
    length = p1.shape[0]
    for i in range(0,length):
        u,v = p1[i,0], p1[i,1]
        x,y = p2[i,0], p2[i,1]
        A[i*2,:] = np.array([-x,-y,-1,0,0,0,x*u,y*u,u])
        A[i*2+1,:] = np.array([0,0,0,-x,-y,-1,v*x,v*y,v])

    [D,V] = np.linalg.eig(np.matmul(A.T,A))
    idx = np.argmin(D)
    H2to1 = np.reshape(V[:,idx], (3,3))
    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...
    H2to1 = np.zeros((3,3))
    maxInliers = -1
    bestInliers = np.zeros((1,1))

    num_matches = matches.shape[0]
    p1 = locs1[matches[:,0], 0:2].T
    p2 = locs2[matches[:,1], 0:2].T

    for i in range(0, num_iter):
        idx = np.random.choice(num_matches, 4)
        rand1 = p1[:,idx]
        rand2 = p2[:,idx]
        H = computeH(rand1, rand2)

        p2_est = np.append(p2.T, np.ones([len(p2.T),1]),1)
        p2_est = p2_est.T
        p1_est = np.matmul(H,p2_est)
        p1_est = p1_est/p1_est[2,:]

        actual_diff = np.square(p1[0,:] - p1_est[0,:]) + np.square(p1[1,:] - p1_est[1,:])
        inliers = actual_diff < tol**2
        numInliers = sum(inliers)
        
        if numInliers > maxInliers:
            maxInliers = numInliers
            bestInliers = inliers

    H2to1 = computeH(p1[:,bestInliers], p2[:,bestInliers])
    return H2to1

if __name__ == '__main__':
    # im1 = cv2.imread('../data/model_chickenbroth.jpg')
    # im2 = cv2.imread('../data/chickenbroth_01.jpg')
    # im2 = cv2.imread('../data/model_chickenbroth.jpg')
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # p1=locs1[matches[:,0], 0:2].T
    # p2=locs2[matches[:,1], 0:2].T
    # H = computeH(p1,p2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=10000, tol=2)
    print (H2to1)