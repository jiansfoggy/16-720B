import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion

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
        name = '../temp_results/aerial-frame'+str(i)+'.png'
        plt.savefig(name, bbox_inches='tight')