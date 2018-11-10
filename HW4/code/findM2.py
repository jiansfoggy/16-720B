'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import submission as sub
import helper as hp
import matplotlib.image as mpimg

def bestM2(pts1, pts2, M, K1, K2):
    F = sub.eightpoint(pts1, pts2, M)
    E = sub.essentialMatrix(F, K1, K2)

    M1 = np.zeros((3,4))
    M1[0,0] = 1
    M1[1,1] = 1
    M1[2,2] = 1

    C1 = np.matmul(K1, M1)

    P = []
    error = []
    num_pos = []
    Marray = hp.camera2(E)
    h,w,d = Marray.shape
    for i in range(0, d):
        C2 = np.matmul(K2, Marray[:,:,i])
        Pi, erri = sub.triangulate(C1, pts1, C2, pts2)
        P.append(Pi)
        error.append(erri)
        ind = np.where(Pi[:,2] > 0)
        num_pos.append(len(ind[0]))

    P = np.stack(P, axis=-1)
    correct = np.equal(num_pos, P.shape[0])
    ind = np.where(correct)[0][0]
    M2 = Marray[:,:,ind]
    M2 = np.reshape(M2, (M2.shape[0],M2.shape[1]))
    P = P[:,:,ind]
    P = np.reshape(P, (P.shape[0],P.shape[1]))
    C2 = np.matmul(K2,M2)

    return M1,C1,M2,C2,P

if __name__ == "__main__":
    im1 = mpimg.imread('../data/im1.png')
    im2 = mpimg.imread('../data/im2.png')
    h,w,d = im1.shape
    M = max(h,w)
    
    data = np.load('../data/some_corresp.npz')
    pts1 = data['pts1']
    pts2 = data['pts2']

    data = np.load('../data/intrinsics.npz')
    K1 = data['K1']
    K2 = data['K2']

    M1, C1, M2, C2, P = bestM2(pts1, pts2, M, K1, K2)
    np.savez('../results/files/q3_3.npz', M2=M2, C2=C2, P=P)