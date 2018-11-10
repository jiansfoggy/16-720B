import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import shift
import cv2

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	
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

	iter1 = 0
	while (True):
		xi_temp = np.arange(x1+p[0], x2+p[0]+0.5)
		yi_temp = np.arange(y1+p[1], y2+p[1]+0.5)
		xi, yi = np.meshgrid(xi_temp, yi_temp)
		patch_x = np.reshape(np.array(It_x_spline.ev(yi.flatten(), xi.flatten()).tolist()), (-1,1))
		patch_y = np.reshape(np.array(It_y_spline.ev(yi.flatten(), xi.flatten()).tolist()), (-1,1))
		A = np.hstack((patch_x, patch_y))

		image = np.array(It1_spline.ev(yi.flatten(), xi.flatten()).tolist())
		b = template - image
		deltap = np.linalg.lstsq(A,b,rcond=None)[0]
		p = p + deltap
		a = np.linalg.norm(deltap)
		if a < tol:
			# print (iter1)
			break
		iter1 += 1
		
	return p