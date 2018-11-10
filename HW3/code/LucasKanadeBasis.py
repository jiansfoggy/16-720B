import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeBasis(It, It1, rect, bases):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here
	p0 = np.zeros(2)
	p = p0
	# get rectangle coordinates
	x1 = rect[0]
	x2 = rect[2]
	y1 = rect[1]
	y2 = rect[3]
	tol = 0.5

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
	x_temp = np.arange(x1,x2+0.5)
	y_temp = np.arange(y1,y2+0.5)
	x,y = np.meshgrid(x_temp, y_temp)
	template = np.array(It_spline.ev(y.flatten(), x.flatten()).tolist())

	num_bases = bases.shape[2]
	shape1 = bases[:,:,0].shape[0]*bases[:,:,0].shape[1]
	B = np.zeros((shape1, num_bases))
	for i in range(0, num_bases):
		base = bases[:,:,i]
		base = base.flatten()
		B[:,i] = base

	M = np.subtract(np.identity(shape1), np.matmul(B, B.T))
	iter1 = 0
	while (True):
		xi_temp = np.arange(x1+p[0], x2+p[0]+0.5)
		yi_temp = np.arange(y1+p[1], y2+p[1]+0.5)
		xi, yi = np.meshgrid(xi_temp, yi_temp)
		patch_x = np.reshape(np.array(It_x_spline.ev(yi.flatten(), xi.flatten()).tolist()), (-1,1))
		patch_y = np.reshape(np.array(It_y_spline.ev(yi.flatten(), xi.flatten()).tolist()), (-1,1))
		A = np.hstack((patch_x, patch_y))
		A_new = np.matmul(M,A)

		image = np.array(It1_spline.ev(yi.flatten(), xi.flatten()).tolist())
		b = template - image
		b_new = np.matmul(M,b)
		deltap = np.linalg.lstsq(A_new,b_new,rcond=None)[0]
		p = p + deltap
		a = np.linalg.norm(deltap)
		# print (a)
		if a < tol:
			# print (iter)
			break
		iter1 += 1
		
	return p