"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import sys
import numpy as np
import helper as hp
import findM2 as fm
from mpl_toolkits import mplot3d
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pts1 = np.divide(pts1,M)
    pts2 = np.divide(pts2,M)
    N = pts1.shape[0]

    pts1_x = pts1[:,0]
    pts1_y = pts1[:,1]
    pts2_x = pts2[:,0]
    pts2_y = pts2[:,1]

    A = np.vstack((np.multiply(pts1_x, pts2_x), np.multiply(pts1_x, pts2_y), pts1_x, np.multiply(pts1_y, pts2_x), np.multiply(pts1_y, pts2_y), pts1_y, pts2_x, pts2_y, np.ones(N)))
    A = A.T

    u,s,v = np.linalg.svd(A)
    F = np.reshape(v[8,:], (3,3))

    F_refined = hp.refineF(F, pts1, pts2)
    t = 1.0/M

    T = np.array([[t,0,0],[0,t,0],[0,0,1]])
    F = np.matmul(T.T, np.matmul(F_refined,T))

    return F

'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pts1 = np.divide(pts1,M)
    pts2 = np.divide(pts2,M)
    N = pts1.shape[0]

    pts1_x = pts1[:,0]
    pts1_y = pts1[:,1]
    pts2_x = pts2[:,0]
    pts2_y = pts2[:,1]

    A = np.vstack((np.multiply(pts1_x, pts2_x), np.multiply(pts1_x, pts2_y), pts1_x, np.multiply(pts1_y, pts2_x), np.multiply(pts1_y, pts2_y), pts1_y, pts2_x, pts2_y, np.ones(N)))
    A = A.T

    u,s,v = np.linalg.svd(A)
    f1 = v[8,:]
    f2 = v[7,:]

    F1 = np.reshape(f1, (3,3))
    F2 = np.reshape(f2, (3,3))

    fun = lambda a: np.linalg.det(a*F1 + (1-a)*F2)
    a0 = fun(0)
    a1 = (2*(fun(1)-fun(-1))/3.0) - ((fun(2)-fun(-2))/12.0)
    a2 = ((fun(1)+fun(-1))/2.0) - fun(0)
    a3 = ((fun(1)-fun(-1))/2.0) - a1
    coeff = np.array([a3,a2,a1,a0])
    roots = np.roots(coeff)

    t = 1.0/M
    T = np.array([[t,0,0],[0,t,0],[0,0,1]])

    Farray = []
    for i in range(0, len(roots)):
            x = roots[i]
            if x.imag == 0:
                x = x.real
                F = x*F1 + (1-x)*F2
                F_refined = hp.refineF(F, pts1, pts2)
                F = np.matmul(T.T, np.matmul(F_refined,T))
                Farray.append(F)
            else:
                continue
    
    Farray = np.stack(Farray, axis=-1)
    return Farray

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    return np.matmul(K2.T, np.matmul(F,K1))

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    N = pts1.shape[0]
    P = np.zeros((N,3))

    for i in range(0,N):
            a1 = np.multiply(pts1[i,0], C1[2,:]) - C1[0,:]
            a2 = np.multiply(pts1[i,1], C1[2,:]) - C1[1,:]
            a3 = np.multiply(pts2[i,0], C2[2,:]) - C2[0,:]
            a4 = np.multiply(pts2[i,1], C2[2,:]) - C2[1,:]
            A = np.vstack((a1,a2,a3,a4))

            u,s,v = np.linalg.svd(A)
            f1 = v[3,:]
            f1 = f1/f1[3]
            P[i,:] = f1[:3]

    a = np.ones((1,N))
    p_temp = np.vstack((P.T, a))
    pts1_new = np.matmul(C1, p_temp)
    pts1_new = pts1_new / pts1_new[2,:]
    
    pts2_new = np.matmul(C2, p_temp)
    pts2_new = pts2_new / pts2_new[2,:]

    pts1_new = pts1_new[:2,:]
    pts2_new = pts2_new[:2,:]

    err = np.power(np.subtract(pts1.T,pts1_new),2) + np.power(np.subtract(pts2.T,pts2_new),2)
    err = np.sum(err)

    return P,err

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def createGaussianFilter(shape, sigma):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    P1 = np.vstack((x1,y1,1))
    epline = np.matmul(F,P1)
    epline = epline/np.linalg.norm(epline)
    a = epline[0][0]
    b = epline[1][0]
    c = epline[2][0]

    step = 10
    sigma = 5
    mindis = np.inf

    #filter code here
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = 0
    y2 = 0
    patch1 = im1[y1-step:y1+step+1, x1-step:x1+step+1]
    kernel = createGaussianFilter((2*step+1, 2*step+1), sigma)

    for i in range(y1-sigma*step, y1+sigma*step):
        x2_curr = (-b*i-c)/a
        x2_temp = round(x2_curr)
        x2_curr = int(round(x2_curr))

        s_h = i-step
        e_h = i+step+1
        s_w = x2_curr-step
        e_w = x2_curr+step+1
        if s_w > 0 and e_w < im2.shape[1] and s_h > 0 and e_h < im2.shape[0]:
            patch2 = im2[s_h:e_h, s_w:e_w]

            weightedDist = []
            for l in range(0, patch2.shape[2]):
                dist = np.subtract(patch1[:,:,l], patch2[:,:,l])
                weightedDist.append(np.linalg.norm(np.matmul(kernel, dist)))
            error = sum(weightedDist)

            if error < mindis:
                mindis = error
                x2 = x2_curr
                y2 = i

    return x2, y2
'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation
    N = pts1.shape[0]
    maxInliers = -1
    bestInliers = np.zeros((1,1))

    num_iter = 50
    tol = 0.02
    deter = np.zeros((1,N))

    for i in range(0, num_iter):
        idx = np.random.choice(N,7)
        pts1_rand = pts1[idx,:]
        pts2_rand = pts2[idx,:]

        Farray = sevenpoint(pts1_rand, pts2_rand, M)

        for j in range(0, Farray.shape[2]):
            F = Farray[:,:,j]

            for k in range(0,N):
                pts1_temp = np.vstack((pts1[k,0], pts1[k,1], 1))
                epline = np.matmul(F,pts1_temp)
                epline = epline/np.linalg.norm(epline)
                pts2_temp = np.hstack((pts2[k,0], pts2[k,1], 1))
                z = np.matmul(pts2_temp, epline)
                deter[:,k] = z[0]
            
            deter = abs(deter)
            inliers = deter < tol
            numInliers = sum(inliers[0])
            if numInliers > maxInliers:
                maxInliers = numInliers
                bestInliers = inliers[0]

    pts1_refined = pts1[bestInliers,:]
    pts2_refined = pts2[bestInliers,:]
    F = eightpoint(pts1_refined, pts2_refined, M)
    return F, bestInliers

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    d = r.shape[0]
    theta = np.linalg.norm(r)
    if theta == 0:
        R = np.identity(d)
    else:
        u = r/theta
        u1 = u[0]
        u2 = u[1]
        u3 = u[2]

        u_x = np.array([[0,-u3,u2],[u3,0,-u1],[-u2,u1,0]])
        R = np.identity(d)*np.cos(theta) + (1-np.cos(theta))*np.matmul(u,u.T) + np.sin(theta)*u_x

    return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def sHalf(r):
    theta = np.linalg.norm(r)

    if theta == np.pi and ( (r[0] == r[1] and r[0] == 0 and r[1] == 0 and r[2] < 0) or  (r[0] == 0 and r[2] < 0) or (r[0] < 0) ):
        return -1*r
    return r

def arcTan2(y,x):
    if x > 0:
        return np.arctan(y/x)
    elif x<0:
        return np.pi + np.arctan(y/x)
    elif x == 0 and y < 0:
        return -np.pi/2
    elif x ==0 and y > 0:
        return np.pi/2

def invRodrigues(R):
    # Replace pass by your implementation
    A = np.subtract(R,R.T)/2
    rho = np.array([A[2,1], A[0,2], A[1,0]])
    s = np.linalg.norm(rho)
    c = (R[0,0]+R[1,1]+R[2,2]-1)/2
    r = []
    if s == 0 and c == 1:
        r = np.zeros((3,1))
    elif s == 0 and c == -1:
        Z = np.add(R, np.identity(3))
        r1 = Z[:,0]
        r2 = Z[:,1]
        r3 = Z[:,2]
        if len(np.nonzero(r1)) > 0:
            v = r1
        elif len(np.nonzero(r2)) > 0:
            v = r2
        elif len(np.nonzero(r3)) > 0:
            v = r3
        u = v/np.linalg.norm(v)
        r_hat = u*np.pi
        r = sHalf(r_hat)
    elif s != 0:
        u = rho/s
        theta = arcTan2(s,c)
        r = u*theta

    return r

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    
    N = p1.shape[0]
    pts = x[:3*N]
    r = x[3*N:3*N+3]
    t = x[3*N+3:]
    t = np.reshape(t, (3,1))
    
    P = np.reshape(pts, (N,3))
    a = np.ones((1,N))
    P = np.vstack((P.T, a))
    
    R = rodrigues(r)
    M2 = np.hstack((R, t))

    C1 = np.matmul(K1, M1)
    C2 = np.matmul(K2, M2)

    pts1_new = np.matmul(C1, P)
    pts1_new = pts1_new / pts1_new[2,:]
    
    pts2_new = np.matmul(C2, P)
    pts2_new = pts2_new / pts2_new[2,:]

    p1_hat = pts1_new[:2,:].T
    p2_hat = pts2_new[:2,:].T

    residuals = np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])])
    return residuals

def residualsWrapper(x, K1, M1, p1, K2, p2):
    return rodriguesResidual(K1, M1, p1, K2, p2, x)

def reProjectionError(K1,M1,K2,M2,p1,p2,P):
    N = p1.shape[0]
    C1 = np.matmul(K1,M1)
    C2 = np.matmul(K2,M2)

    a = np.ones((1,N))
    p_temp = np.vstack((P.T, a))
    pts1_new = np.matmul(C1, p_temp)
    pts1_new = pts1_new / pts1_new[2,:]
    
    pts2_new = np.matmul(C2, p_temp)
    pts2_new = pts2_new / pts2_new[2,:]

    pts1_new = pts1_new[:2,:]
    pts2_new = pts2_new[:2,:]

    err = np.power(np.subtract(p1.T,pts1_new),2) + np.power(np.subtract(p2.T,pts2_new),2)
    err = np.sqrt(np.sum(err))
    return err

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    
    P = P_init.reshape([-1])
    R = M2_init[:,:3]
    t = M2_init[:,3]

    r = invRodrigues(R)
    x0 = np.hstack((P,r,t))

    error_init = reProjectionError(K1,M1,K2,M2_init,p1,p2,P_init)
    print ("Initial Re-Projection Error", error_init)

    res = least_squares(residualsWrapper, x0, args=(K1, M1, p1, K2, p2))

    N = p1.shape[0]
    pts = res.x[:3*N]
    r = res.x[3*N:3*N+3]
    t = res.x[3*N+3:]
    t = np.reshape(t, (3,1))

    P = np.reshape(pts, (N,3))
    R = rodrigues(r)
    M2 = np.hstack((R, t))

    error_final = reProjectionError(K1,M1,K2,M2,p1,p2,P)
    print ("Final Re-Projection Error", error_final)

    return M2, P

def runEighPoint(im1,im2,pts1,pts2,M):
    F = eightpoint(pts1, pts2, M)
    try:
        hp.displayEpipolarF(im1, im2, F)
    except Exception:
        pass
    np.savez('../results/files/q2_1.npz', F=F, M=M)

def runSevenPoint(im1,im2,pts1,pts2,M):
    dist = np.power(np.subtract(pts1[:,0], pts2[:,0]), 2) + np.power(np.subtract(pts1[:,1], pts2[:,1]), 2)
    points = dist.argsort()[:7]
    pts1_temp = pts1[points,:]
    pts2_temp = pts2[points,:]
    Farray = sevenpoint(pts1, pts2, M)
    for i in range(0, Farray.shape[2]):
        F = Farray[:,:,i]
        try:
            hp.displayEpipolarF(im1, im2, F)
        except Exception:
            continue
    # saving the best F
    np.savez('../results/files/q2_2.npz', F=Farray[:,:,0], M=M, pts1=pts1_temp, pts2=pts2_temp)

def runEssentialMatrix(pts1,pts2,M,K1,K2):
    F = eightpoint(pts1, pts2, M)
    E = essentialMatrix(F,K1,K2)
    print (E)

def runEpipolarCorrespondence(im1, im2, pts1, pts2, M):
    F8 = eightpoint(pts1, pts2, M)

    try:
        p1, p2 = hp.epipolarMatchGUI(im1,im2,F8)
        np.savez('../results/files/q4_1.npz', F=F8, pts1=p1, pts2=p2)
    except Exception:
        pass

def runRansac(pts1noisy, pts2noisy, M, im1, im2):
    F = eightpoint(pts1noisy, pts2noisy, M)
    F_ransac, _ = ransacF(pts1noisy, pts2noisy, M)
    
    try:
        hp.displayEpipolarF(im1, im2, F)
    except Exception:
        pass
    
    try:
        hp.displayEpipolarF(im1, im2, F_ransac)
    except Exception:
        pass

def runRodrigues():
    r = np.ones([3, 1])
    R = invRodrigues(r)
    print (R)

    R = np.identity(3)
    r = rodrigues(R)
    print (r)

def projectPoints(P, name):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.scatter(P[:,0], P[:,1], P[:,2], color='b', marker='.')
    plt.show()
    # plt.savefig(name)

def runBA(pts1, pts2, M, K1, K2):
    _, inliers = ransacF(pts1, pts2, M)
    p1 = pts1[inliers,:]
    p2 = pts2[inliers,:]

    M1, C1, M2_init, C2, P_init = fm.bestM2(p1, p2, M, K1, K2)
    projectPoints(P_init, '../results/images/initial_points.png')
    
    M2, P = bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init)
    projectPoints(P, '../results/images/final_points.png')

    np.savez('../results/files/q5_3.npz', M2i = M2_init, M2f = M2, Pi = P_init, Pf=P)

if __name__ == "__main__":
        data = np.load('../data/some_corresp.npz')
        pts1 = data['pts1']
        pts2 = data['pts2']

        im1 = mpimg.imread('../data/im1.png')
        im2 = mpimg.imread('../data/im2.png')
        h,w,d = im1.shape
        M = max(h,w)

        data = np.load('../data/intrinsics.npz')
        K1 = data['K1']
        K2 = data['K2']

        data = np.load('../data/some_corresp_noisy.npz')
        pts1noisy = data['pts1']
        pts2noisy = data['pts2']

        inp = int(sys.argv[1])
        if inp == 1:
            runEighPoint(im1,im2,pts1,pts2,M)
        elif inp == 2:
            runSevenPoint(im1,im2,pts1,pts2,M)
        elif inp == 3:
            runEssentialMatrix(pts1,pts2,M,K1,K2)
        elif inp == 4:
            runEpipolarCorrespondence(im1, im2, pts1, pts2, M)
        elif inp == 5:
            runRansac(pts1noisy, pts2noisy, M, im1, im2)
        elif inp == 6:
            runRodrigues()
        elif inp == 7:
            runBA(pts1noisy, pts2noisy, M, K1, K2)
        