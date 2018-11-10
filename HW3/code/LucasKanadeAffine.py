import numpy as np
from scipy.interpolate import RectBivariateSpline

def validate_coords(y, x, ny, nx):
	a = np.logical_and(np.logical_and(x>=0, x<=nx-1), np.logical_and(y>=0, y<=ny-1))
	return a.nonzero()[0]

def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
	M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
	tol = 0.08
	h, w = It1.shape
	x = np.arange(w)
	y = np.arange(h)
	
	# get gradient of image
	It_y, It_x = np.gradient(It1)

	# interpolating code to get I_t
	It_spline = RectBivariateSpline(y,x,It)
	It1_spline = RectBivariateSpline(y,x,It1)
	It_x_spline = RectBivariateSpline(y,x,It_x)
	It_y_spline = RectBivariateSpline(y,x,It_y)

	# get the template
	x_temp = np.arange(0,w-1)
	y_temp = np.arange(0,h-1)
	x,y = np.meshgrid(x_temp, y_temp)
	x = np.reshape(x, (-1,1))
	y = np.reshape(y, (-1,1))

	iter1 = 0
	while (True):
		a = np.ones((x.shape[0],1))
		homogenous = np.hstack((y,x,a))
		affine_homogenous = np.matmul(M, homogenous.T)
		
		# check affine coordinates in range
		valid_coords = validate_coords(affine_homogenous[0], affine_homogenous[1], h, w)

		yi = affine_homogenous[0, valid_coords]
		xi = affine_homogenous[1, valid_coords]
		patch_x = np.reshape(np.array(It_x_spline.ev(yi, xi).tolist()), (-1,1))
		patch_y = np.reshape(np.array(It_y_spline.ev(yi, xi).tolist()), (-1,1))

		x = x[valid_coords]
		y = y[valid_coords]
		A = np.hstack((np.multiply(y,patch_y), np.multiply(x,patch_y), patch_y, np.multiply(y,patch_x), np.multiply(x,patch_x), patch_x))
		image = np.array(It1_spline.ev(yi, xi).tolist())
		template = np.array(It_spline.ev(y.flatten(), x.flatten()).tolist())
		b = template - image
		deltap = np.linalg.lstsq(A,b,rcond=None)[0]
		deltaM = np.reshape(deltap, (2,3))
		M = M + deltaM
		a = np.linalg.norm(deltaM)
		# print (a)
		if a < tol:
			# print (M)
			# print (iter1)
			break
		iter1 += 1

	return M