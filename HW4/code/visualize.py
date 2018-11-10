'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import submission as sub
import findM2 as fm

if __name__ == "__main__":
    im1 = mpimg.imread('../data/im1.png')
    im2 = mpimg.imread('../data/im2.png')
    h,w,d = im1.shape
    M = max(h,w)
    
    data = np.load('../data/some_corresp.npz')
    pts1 = data['pts1']
    pts2 = data['pts2']
    F = sub.eightpoint(pts1, pts2, M)

    data = np.load('../data/intrinsics.npz')
    K1 = data['K1']
    K2 = data['K2']

    data = np.load('../data/templeCoords.npz')
    x1 = data['x1']
    y1 = data['y1']

    x2 = np.zeros((x1.shape[0],1))
    y2 = np.zeros((y1.shape[0],1))

    for i in range(0, x1.shape[0]):
        x2_tmp, y2_tmp = sub.epipolarCorrespondence(im1, im2, F, x1[i,:][0],y1[i,:][0])
        x2[i,:] = x2_tmp
        y2[i,:] = y2_tmp

    p1 = np.hstack((x1, y1))
    p2 = np.hstack((x2, y2))

    M1, C1, M2, C2, P = fm.bestM2(p1, p2, M, K1, K2)

    np.savez('../results/files/q4_2.npz', F=F, M1=M1, C1=C1, M2=M2, C2=C2)
    # visualization code
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.scatter(P[:,0], P[:,1], P[:,2], color='b', marker='.')
    plt.show()