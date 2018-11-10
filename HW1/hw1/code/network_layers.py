import numpy as np
import scipy.ndimage
import os,time
from util import get_VGG16_weights
import skimage

def extract_deep_feature(x, vgg16_weights):
	'''
	Extracts deep features from the given VGG-16 weights.
	
	[input]
	* x: numpy.ndarray of shape (H,W,3)
	* vgg16_weights: numpy.ndarray of shape (L,3)

	[output]
	* feat: numpy.ndarray of shape (K)
	x,vgg16_weights
	'''

	# print (vgg16_weights)
	# vgg16_weights = get_VGG16_weights()
	# image = skimage.io.imread("../data/kitchen/sun_aasmevtpkslccptd.jpg").astype('float')
	
	image = skimage.transform.resize(x, (224,224,3), mode='reflect')
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	response = np.divide(np.subtract(image, mean), std)

	for layer in vgg16_weights:
		# print (layer[0])
		if (layer[0] == "conv2d"):
			response = multichannel_conv2d(response, layer[1], layer[2])
		elif (layer[0] == "relu"):
			response = relu(response)
		elif (layer[0] == "maxpool2d"):
			response = max_pool2d(response, layer[1])
		elif (layer[0] == "linear"):
			if response.ndim != 1:
				response = np.moveaxis(response, -1, 0).flatten()
			response = linear(response, layer[1], layer[2])

	return response

def multichannel_conv2d(x,weight,bias):
	'''
	Performs multi-channel 2D convolution.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* feat: numpy.ndarray of shape (H,W,output_dim)
	'''
	input_dim = x.shape[2]
	feat = []
	for i in range(0, bias.shape[0]):
		response = []
		for j in range(0, input_dim):
			kernel = np.fliplr(np.flipud(weight[i,j,:,:]))
			response.append(scipy.ndimage.convolve(x[:,:,j], kernel))
		
		response = np.sum(np.asarray(response), 0)
		feat.append(response)
	
	return np.add(np.moveaxis(np.asarray(feat), 0, -1), bias)

def relu(x):
	'''
	Rectified linear unit.

	[input]
	* x: numpy.ndarray

	[output]
	* y: numpy.ndarray
	'''
	x[x<0] = 0
	return x

def max_pool2d(x,size):
	'''
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* size: pooling receptive field

	[output]
	* y: numpy.ndarray of shape (H/size,W/size,input_dim)
	'''

	y = np.zeros((int(np.floor(x.shape[0]/size)), int(np.floor(x.shape[1]/size)), x.shape[2]))

	for channel in range(0, x.shape[2]):
		r = 0
		for rows in np.arange(0, x.shape[0]-size+1, size):
			c = 0
			for cols in np.arange(0, x.shape[1]-size+1, size):
				y[r,c,channel] = np.max(x[rows:rows+size, cols:cols+size, channel])
				c = c+1
			r = r+1

	return y

def linear(x,W,b):
	'''
	Fully-connected layer.

	[input]
	* x: numpy.ndarray of shape (input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* y: numpy.ndarray of shape (output_dim)
	'''
	return np.add(np.matmul(W, x), b)