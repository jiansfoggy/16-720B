import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import multiprocessing as mp

def build_recognition_system(num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,M)
	* labels: numpy.ndarray of shape (N)
	* dictionary: numpy.ndarray of shape (K,3F)
	* SPM_layer_num: number of spatial pyramid layers
	'''

	train_data = np.load("../data/train_data.npz")
	dictionary = np.load("../outputs/dictionary.npy")

	data = train_data['image_names']
	SPM_layer_num = 2
	K = 100
	size_Feature = int(K*(4**(SPM_layer_num+1) -1)/3)
	
	pool = mp.Pool(num_workers)

	results = []
	for i in range(0, len(data)):
		print (i)
		args = [data[i][0], dictionary, SPM_layer_num, K]
		results.append(pool.apply_async(get_image_feature, args))
	
	features = []
	for result in results:
		features.append(result.get())

	final_features = np.reshape(features, (len(data), size_Feature))
	labels = np.asarray(train_data['labels'])
	np.savez('../outputs/trained_system.npz', features = final_features, labels = labels, SPM_layer_num = SPM_layer_num, dictionary = dictionary)


def test_label(args):
	file_path,dictionary,layer_num,K, features, labels = args
	feature = get_image_feature(file_path, dictionary, layer_num, K)
	distance = distance_to_set(feature, features)
	i = np.argmax(distance)
	label = labels[i]
	return label

def evaluate_recognition_system(num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	test_data = np.load("../data/test_data.npz")
	trained_system = np.load("../outputs/trained_system.npz")

	features = trained_system['features']
	dictionary = trained_system['dictionary']
	SPM_layer_num = trained_system['SPM_layer_num']
	labels = trained_system['labels']
	K = dictionary.shape[0]

	data = test_data['image_names']
	pool = mp.Pool(num_workers)

	features_test = []
	for i in range(0, len(data)):
		args = [(data[i][0], dictionary, SPM_layer_num, K, features, labels)]
		features_test.append(pool.apply_async(test_label, args))

	test_labels = []
	for feature in features_test:
		test_labels.append(feature.get())

	testActualLabels = test_data['labels']
	size_confusion = len(np.unique(testActualLabels))
	C = np.zeros((size_confusion, size_confusion))

	for a,p in zip(testActualLabels, test_labels):
		C[a][p] += 1

	accuracy = np.diag(C).sum()/C.sum()
	return C, accuracy

def get_image_feature(file_path,dictionary,layer_num,K):
	'''
	Extracts the spatial pyramid matching feature.

	[input]
	* file_path: path of image file to read
	* dictionary: numpy.ndarray of shape (K,3F)
	* layer_num: number of spatial pyramid layers
	* K: number of clusters for the word maps

	[output]
	* feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
	'''

	image = imageio.imread('../data/' + file_path)
	wordmap = visual_words.get_visual_words(image, dictionary)
	hist_all = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
	return hist_all

def distance_to_set(word_hist,histograms):
	'''
	Compute similarity between a histogram of visual words with all training image histograms.

	[input]
	* word_hist: numpy.ndarray of shape (K)
	* histograms: numpy.ndarray of shape (N,K)

	[output]
	* sim: numpy.ndarray of shape (N)
	'''

	min_compare = np.minimum(histograms, word_hist)
	return np.sum(min_compare, axis=1)


def get_feature_from_wordmap(wordmap,dict_size):
	'''
	Compute histogram of visual words.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* dict_size: dictionary size K

	[output]
	* hist: numpy.ndarray of shape (K)
	'''

	flatten_wordmap = wordmap.flatten()
	hist = np.histogram(flatten_wordmap, bins = dict_size, range = (0,dict_size))
	hist = hist[0]/np.linalg.norm(hist[0], ord = 1)
	return np.asarray(hist)

def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
	'''
	Compute histogram of visual words using spatial pyramid matching.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* layer_num: number of spatial pyramid layers
	* dict_size: dictionary size K

	[output]
	* hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
	'''

	i_h, i_w = wordmap.shape
	hist_all = []

	for layer in range(0, layer_num+1):
		D = 2**layer
		if layer == 0 or layer == 1:
			weight = 1/(2**(layer_num))
		else:
			weight = 1/(2**(layer_num+1-layer))

		height_indices = np.round(np.arange(0, i_h+1, i_h/D)).astype('int')
		width_indices = np.round(np.arange(0, i_w+1, i_w/D)).astype('int')
		divisions = height_indices.shape[0]-1


		for i in range(0, divisions):
			for j in range (0, divisions):
				s_h, s_w = height_indices[i], width_indices[j]
				e_h, e_w = height_indices[i+1], width_indices[j+1]
				imageSection = wordmap[s_h:e_h, s_w:e_w]
				imageDictionary = get_feature_from_wordmap(imageSection, dict_size)
				imageDictionary = imageDictionary*weight
				hist_all.append(imageDictionary)

	hist_all = np.asarray(hist_all)
	hist_all = hist_all.flatten()
	hist_all = hist_all/np.linalg.norm(hist_all, ord = 1)
	return hist_all