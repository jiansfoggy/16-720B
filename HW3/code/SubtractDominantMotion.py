import numpy as np
from scipy.interpolate import RectBivariateSpline
from LucasKanadeAffine import LucasKanadeAffine
from scipy import ndimage
from skimage import morphology

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
	M = LucasKanadeAffine(image1, image2)
	tol = 0.35
	
	h,w = image1.shape
	x = np.arange(w)
	y = np.arange(h)
	mask = np.ones((h,w), dtype=bool)
	
	x_temp = np.arange(0,w)
	y_temp = np.arange(0,h)
	xi,yi = np.meshgrid(x_temp, y_temp)
	ones = np.ones(w*h)
	xy1 = np.array([yi.flatten(), xi.flatten(), ones])

	xy1_new = np.matmul(np.vstack((M, np.array([0,0,1]))), xy1)
	y_new = xy1_new[0,:]
	x_new = xy1_new[1,:]

	image2_spline = RectBivariateSpline(y,x,image2)
	temp = np.array(image2_spline.ev(y_new, x_new).tolist())
	image1_new = np.reshape(temp, (h,w))

	abs_diff = np.absolute(image1_new, image1)
	# print (np.max(abs_diff), np.min(abs_diff), np.mean(abs_diff))
	ind = (abs_diff >= tol)

	abs_diff[ind] = 1
	abs_diff[~ind] = 0
	mask = abs_diff

	# print (mask)
	# ser = morphology.disk(5)
	# mask = morphology.binary_dilation(mask, ser)

	# ser1 = morphology.disk(1)
	# mask = morphology.binary_dilation(mask, ser1)
	# mask = ndimage.binary_dilation(mask)
	# mask = ndimage.binary_erosion(mask)
	# mask = morphology.remove_small_objects(mask, 50)
	return mask