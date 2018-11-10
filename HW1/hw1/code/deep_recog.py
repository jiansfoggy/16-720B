import numpy as np
import multiprocessing as mp
import threading
import queue
import imageio
import os,time
import torch
import skimage.transform
from torchvision.transforms import transforms
import util
import network_layers
import scipy

def build_recognition_system(vgg16,num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,K)
	* labels: numpy.ndarray of shape (N)
	'''

	train_data = np.load("../data/train_data.npz")

	data = train_data['image_names']
	pool = mp.Pool(num_workers)

	results = []
	for i in range(0, len(data)):
		print (i)
		args = [(i, data[i][0], vgg16)]
		results.append(pool.apply_async(get_image_feature, args))

	features = []
	for result in results:
		feature = result.get()
		features.append(feature.detach().numpy())

	final_features = np.reshape(features, (len(data), len(features[0])))
	# print (final_features.shape)
	labels = np.asarray(train_data['labels'])
	np.savez('../outputs/trained_system_deep.npz', features = final_features, labels = labels)
	
def testClassification(args):
	i, file_path, vgg_weights, features, labels = args
	args = (i, file_path, vgg_weights)
	feature = get_image_feature(args)
	actual_feature = feature.detach().numpy()
	feature_to_use = np.reshape(actual_feature, (actual_feature.shape[0], 1))
	distance = distance_to_set(feature_to_use.T, features)
	label = labels[np.argmax(distance)]
	return label

def evaluate_recognition_system(vgg16,num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	test_data = np.load("../data/test_data.npz")
	trained_system = np.load('../outputs/trained_system-vgg.npz')

	features = trained_system['features']
	train_labels = trained_system['labels']
	test_labels = test_data['labels']
	test_images = test_data['image_names']

	pool = mp.Pool(num_workers)

	results = []
	for i in range(0,len(test_images)):
		args = [(i, test_images[i][0], vgg16, features, train_labels)]
		results.append(pool.apply_async(testClassification, args))

	features = []
	for result in results:
		features.append(result.get())

	size_confusion = len(np.unique(test_labels))
	C = np.zeros((size_confusion, size_confusion))

	for a,p in zip(test_labels, features):
		C[a][p] += 1

	accuracy = np.diag(C).sum()/C.sum()
	return C, accuracy

def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: torch.Tensor of shape (3,H,W)
	'''

	image_temp = skimage.transform.resize(image, (224,224,3), mode='reflect')
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	image_normalize = np.divide(np.subtract(image_temp, mean), std)
	transform = transforms.Compose([transforms.ToTensor()])
	image_composed = transform(image_normalize)
	return image_composed.unsqueeze(0)

def get_image_feature(args):
	'''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
 	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.
	* time_start: time stamp of start time
 	[saved]
	* feat: evaluated deep feature
	'''

	i,image_path,vgg16 = args
	image_path = '../data/' + image_path
	image = skimage.io.imread(image_path)
	image = preprocess_image(image)

	conv_layers = torch.nn.Sequential(*list(vgg16.children())[0])
	fc7_layer = torch.nn.Sequential(*list(vgg16.children())[1][:5])
	conv_output = conv_layers(image)
	return fc7_layer(conv_output.flatten())

def distance_to_set(feature,train_features):
	'''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N,K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''

	return -1 * scipy.spatial.distance.cdist(feature, train_features, metric='euclidean')