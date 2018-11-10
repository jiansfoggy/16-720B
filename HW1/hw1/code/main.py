import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import network_layers
import skimage.io
import time

if __name__ == '__main__':

	num_cores = util.get_num_CPU()

	# path_img_1 = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
	# path_img_2 = "../data/windmill/sun_aszoiltmpeptbfwt.jpg"
	# path_img_3 = "../data/highway/sun_aacqsbumiuidokeh.jpg"
	# dictionary = np.load('dictionary.npy')
	# trained_system = np.load("../outputs/trained_system.npz")
	# dictionary = trained_system['dictionary']

	# image = skimage.io.imread(path_img_1)
	# img = visual_words.get_visual_words(image,dictionary)
	# util.save_wordmap(img, "image_wordmap_kitchen.png")

	# image = skimage.io.imread(path_img_2)
	# img = visual_words.get_visual_words(image,dictionary)
	# util.save_wordmap(img, "image_wordmap_windmill.png")

	# image = skimage.io.imread(path_img_3)
	# img = visual_words.get_visual_words(image,dictionary)
	# util.save_wordmap(img, "../outputs/image_wordmap_highway.png")

	# filter_responses = visual_words.extract_filter_responses(image)
	# util.display_filter_responses(filter_responses)

	# visual_words.compute_dictionary(num_workers=num_cores)
	
	# dictionary = np.load('dictionary.npy')
	# print (dictionary)
	# img = visual_words.get_visual_words(image,dictionary)
	# util.save_wordmap(img, "image_wordmap_kitchen.png")
	# img = visual_recog.get_feature_from_wordmap(img, len(dictionary))
	# img = visual_recog.get_feature_from_wordmap_SPM(img, 4, len(dictionary))
	# visual_recog.get_image_feature("../data/windmill/sun_aszoiltmpeptbfwt.jpg", dictionary, 2, 100)
	
	# visual_recog.build_recognition_system(num_workers=num_cores)
	# conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
	# print(conf)
	# print(np.diag(conf).sum()/conf.sum())
	# print(accuracy)

	# vgg16 = torchvision.models.vgg16(pretrained=True).double()
	# vgg16.eval()
	# network_layers.extract_deep_feature()
	# deep_recog.build_recognition_system(vgg16,num_workers=num_cores-1)
	# conf, accuracy = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores)
	# print(conf)
	# print(np.diag(conf).sum()/conf.sum())
	pass