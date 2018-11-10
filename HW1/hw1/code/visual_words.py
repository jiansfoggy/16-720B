import numpy as np
import multiprocessing as mp
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import matplotlib.pyplot as plt
import util
import random

def extract_filter_responses(image):
	'''
	Extracts the filter responses for the given image.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	[output]
	* filter_responses: numpy.ndarray of shape (H,W,3F)
	'''

	scales = [1,2,4,8,8*np.sqrt(2)]
	F = 20
	filter_responses = np.zeros((image.shape[0], image.shape[1], image.shape[2]*F))
	
	image = image.astype('float')/255
	# make grayscale to color
	if not image.shape[2]:
		image = np.repeat(image[:,:,np.newaxis], 3, axis=2)
	
	# convert to Lab space
	image = skimage.color.rgb2lab(image)
	
	i = -1
	for scale in scales:
		f1 = scipy.ndimage.gaussian_filter(image[:,:,0], sigma=scale)
		i+=1
		filter_responses[:,:,i] = f1
		f2 = scipy.ndimage.gaussian_filter(image[:,:,1], sigma=scale)
		i+=1
		filter_responses[:,:,i] = f2
		f3 = scipy.ndimage.gaussian_filter(image[:,:,2], sigma=scale)
		i+=1
		filter_responses[:,:,i] = f3

		f1 = scipy.ndimage.gaussian_laplace(image[:,:,0], sigma=scale)
		i+=1
		filter_responses[:,:,i] = f1
		f2 = scipy.ndimage.gaussian_laplace(image[:,:,1], sigma=scale)
		i+=1
		filter_responses[:,:,i] = f2
		f3 = scipy.ndimage.gaussian_laplace(image[:,:,2], sigma=scale)
		i+=1
		filter_responses[:,:,i] = f3

		f1 = scipy.ndimage.gaussian_filter(image[:,:,0], sigma=scale, order = [0,1])
		i+=1
		filter_responses[:,:,i] = f1
		f2 = scipy.ndimage.gaussian_filter(image[:,:,1], sigma=scale, order = [0,1])
		i+=1
		filter_responses[:,:,i] = f2
		f3 = scipy.ndimage.gaussian_filter(image[:,:,2], sigma=scale, order = [0,1])
		i+=1
		filter_responses[:,:,i] = f3

		f1 = scipy.ndimage.gaussian_filter(image[:,:,0], sigma=scale, order = (1,0))
		i+=1
		filter_responses[:,:,i] = f1
		f2 = scipy.ndimage.gaussian_filter(image[:,:,1], sigma=scale, order = (1,0))
		i+=1
		filter_responses[:,:,i] = f2
		f3 = scipy.ndimage.gaussian_filter(image[:,:,2], sigma=scale, order = (1,0))
		i+=1
		filter_responses[:,:,i] = f3

	return filter_responses

def get_visual_words(image,dictionary):
	'''
	Compute visual words mapping for the given image using the dictionary of visual words.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	
	[output]
	* wordmap: numpy.ndarray of shape (H,W)
	'''
	if image.shape[2] >= 3:
		image = image[:,:,:3]
	response = extract_filter_responses(image)
	width, height, depth = response.shape
	image_height, image_width, image_depth = image.shape
	response_new = np.reshape(response, (width*height, response.shape[-1]))
	distances = scipy.spatial.distance.cdist(response_new, dictionary)
	distances = np.argmin(distances, axis=1)
	wordmap = np.reshape(distances, (image_height, image_width))
	return wordmap


def compute_dictionary_one_image(args):
	'''
	Extracts random samples of the dictionary entries from an image.
	This is a function run by a subprocess.

	[input]
	* i: index of training image
	* alpha: number of random samples
	* image_path: path of image file
	* time_start: time stamp of start time

	[saved]
	* sampled_response: numpy.ndarray of shape (alpha,3F)
	'''

	i,alpha,image_path = args
	image = skimage.io.imread('../data/' + image_path)
	image = image.astype('float')/255
	if image.shape[2] >= 3:
		image = image[:,:,:3]
	response = extract_filter_responses(image)
	filter_responses = np.random.permutation(response.reshape(image.shape[0]*image.shape[1], -1))[:alpha]
	return filter_responses

def compute_dictionary(num_workers=2):
	'''
	Creates the dictionary of visual words by clustering using k-means.

	[input]
	* num_workers: number of workers to process in parallel
	
	[saved]
	* dictionary: numpy.ndarray of shape (K,3F)
	'''

	train_data = np.load("../data/train_data.npz")
	F = 20
	T = train_data['image_names'].shape[0]
	alpha = 200
	k = 100

	pool = mp.Pool(num_workers)
	# get all responses
	filter_responses = []
	for i in range(0, T):
		# print (i)
		args = [(i, alpha, train_data['image_names'][i][0])]
		filter_responses.append(pool.apply_async(compute_dictionary_one_image, args))

	# stack them to get a filtered reponses matrix of size (alpha*T,3*F)
	features = []
	for result in filter_responses:
		features.append(result.get())

	a = features[0]
	for i in range(1, len(features)):
		a = np.concatenate((a, features[i]), axis=0)

	# save output features
	np.save('../outputs/filtered_responses.npy', a)

	# perform k-means clustering
	kmeans = sklearn.cluster.KMeans(n_clusters=k, n_jobs=-1).fit(a)
	dictionary = kmeans.cluster_centers_
	print (dictionary.shape)
	np.save('../outputs/dictionary.npy', dictionary)