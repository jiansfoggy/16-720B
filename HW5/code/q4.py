import numpy as np

import skimage
import skimage.measure as skmeasure
import skimage.color as skcolor
import skimage.restoration as skrestore
import skimage.filters as skfilter
import skimage.morphology as skmorph
import skimage.segmentation as sksegment

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None

    sigma_est = skrestore.estimate_sigma(image, multichannel=True, average_sigmas=True)
    denoised_image = skrestore.denoise_bilateral(image, sigma_color=sigma_est, win_size=5)    
    
    bw = skcolor.rgb2gray(denoised_image)

    thresh = skfilter.threshold_otsu(bw)
    bw = bw < thresh
    
    bw = skmorph.dilation(bw, skmorph.square(7))
    bw = skmorph.erosion(bw, skmorph.square(3))

    label_image = skmeasure.label(bw, neighbors=8, background=0.0)
    image_label_overlay = skcolor.label2rgb(label_image, image = bw)

    bboxes = []
    for region in skmeasure.regionprops(label_image):
        if region.area >= 850:
            minr, minc, maxr, maxc = region.bbox
            bboxes.append(np.array([minr, minc, maxr, maxc]))
    
    return bboxes, bw