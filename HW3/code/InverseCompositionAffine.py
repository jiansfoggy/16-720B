import numpy as np
from scipy.interpolate import RectBivariateSpline

def validate_coords(y, x, ny, nx):
	a = np.logical_and(np.logical_and(x>=0, x<=nx-1), np.logical_and(y>=0, y<=ny-1))
	return a.nonzero()[0]

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
	M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

	h,w = It.shape
	x = np.arange(w)
	y = np.arange(h)
	
	It_y, It_x = np.gradient(It)

	It_spline = RectBivariateSpline(y, x, It)
	It1_spline = RectBivariateSpline(y, x, It1)
	It_x_spline = RectBivariateSpline(y, x, It_x)
	It_y_spline = RectBivariateSpline(y, x, It_y)

	
	xt,yt = np.meshgrid(x, y)
	xt = np.reshape(xt, (-1,1))
	yt = np.reshape(yt, (-1,1))
	
	template = np.array(It_spline.ev(yt,xt).tolist())
	template = template.ravel()
	# print ("Template shape is {0}".format(template.shape))

	patch_x = np.reshape(np.array(It_x_spline.ev(yt,xt).tolist()), (-1,1))
	patch_y = np.reshape(np.array(It_y_spline.ev(yt,xt).tolist()), (-1,1))
	A = np.hstack((np.multiply(yt,patch_y), np.multiply(xt,patch_y), patch_y, np.multiply(yt,patch_x), np.multiply(xt,patch_x), patch_x))
	a = np.ones((xt.shape[0],1))
	xy1 = np.hstack((yt,xt,a))

	tol = 0.1
	iter1 = 0
	while (True):
		affine_homogenous = np.matmul(M,xy1.T)
		valid_coords = validate_coords(affine_homogenous[0], affine_homogenous[1],  h, w)
		C = A[valid_coords,:]
		H = np.matmul(C.T,C)

		yi = affine_homogenous[0, valid_coords]
		xi = affine_homogenous[1, valid_coords]

		image = np.array(It1_spline.ev(yi, xi).tolist())
		temp_template = template[valid_coords]
		b = image - temp_template
		b = np.matmul(C.T, b)

		deltap = np.linalg.lstsq(H,b,rcond=None)[0]
		deltaM = np.reshape(deltap, (2,3))

		M = M - deltaM
		a = np.linalg.norm(deltaM)
		if a < tol:
			# print (iter1)
			break

		iter1 += 1
	return M