import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from InverseCompositionAffine import InverseCompositionAffine
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
	M = InverseCompositionAffine(image1, image2)
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
	ind = (abs_diff >= tol)

	abs_diff[ind] = 1
	abs_diff[~ind] = 0
	mask = abs_diff
	return mask

# write your script here, we recommend the above libraries for making your animation
sequences = np.load('../data/aerialseq.npy')
num_frames = sequences.shape[2]
for i in range(0,num_frames-1):
    img1 = sequences[:,:,i]
    img2 = sequences[:,:,i+1]

    mask = SubtractDominantMotion(img1, img2)

    if i==30 or i==60 or i==90 or i==120:
        fig,ax = plt.subplots(1)
        ax.imshow(img2)
        C = np.dstack((img2, img2, img2, mask))
        ax.imshow(C)
        plt.axis('off')
        name = '../temp_results/affine/aerial-frame-new'+str(i)+'.png'
        plt.savefig(name, bbox_inches='tight')