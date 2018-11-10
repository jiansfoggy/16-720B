import cv2
import numpy as np
from planarH import computeH
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def compute_extrinsics(K,H):
    H_hat = np.matmul(np.linalg.inv(K),H)
    rotation_hat = H_hat[:,:2]
    [U,S,V] = np.linalg.svd(rotation_hat)
    S = np.array([[1,0],[0,1],[0,0]])
    new_rotation = np.matmul(U,np.matmul(S,V))
    rotation_hat3 = np.cross(new_rotation[:,0], new_rotation[:,1])
    rotation_hat3 = np.reshape(rotation_hat3, (3,1))
    R = np.append(new_rotation, rotation_hat3, 1)
    det_R = np.linalg.det(R)
    if det_R == -1:
        R[:,2] = -1*R[:,2]

    lambda_val = 0
    for i in range(0,3):
        for j in range(0,2):
            lambda_val += rotation_hat[i,j]/R[i,j]

    lambda_val = lambda_val/6
    t = H_hat[:,2]/lambda_val
    t = np.reshape(t, (3,1))
    return R,t

def project_extrinsics(K,W,R,t):
    extrinsic_matrix = np.append(R, t, 1)
    a = np.ones([len(W[2,:]),1]).T
    W = np.append(W, a, 0)
    X_projected = np.matmul(K,np.matmul(extrinsic_matrix, W))
    X_projected[2,np.where(X_projected[2,:]==0)[0]] = 1
    X_projected = X_projected/X_projected[2,:]
    X = X_projected[:2,:]
    return X

def display_points(K,R,t):    
    W_new = np.loadtxt('../data/sphere.txt')
    shift = [5.2,11.2,-6.85/2]
    # shift = [0,0,0]
    W_new[0,:] += shift[0]
    W_new[1,:] += shift[1]
    W_new[2,:] += shift[2]
    X = project_extrinsics(K,W_new,R,t)

    sphere_x, sphere_y = [], []
    for i in range(X.shape[1]):
        sphere_x.append(int(X[0,i]))
        sphere_y.append(int(X[1,i]))

    im = cv2.imread('../data/prince_book.jpeg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    implot = plt.imshow(im)
    plt.plot(sphere_x, sphere_y, '-', color='yellow')
    plt.axis('off')
    plt.savefig("../results/ar.png", bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    K = np.array([[3043.72,0,1196],[0,3043.72,1604],[0,0,1]])
    W = np.array([[0,18.2,18.2,0],[0,0,26,26],[0,0,0,0]])
    X = np.array([[483, 1704, 2175, 67], [810, 781, 2217, 2286]])
    H = computeH(X, W[:2,:])
    R,t = compute_extrinsics(K,H)
    X_projected = project_extrinsics(K,W,R,t)
    display_points(K,R,t)