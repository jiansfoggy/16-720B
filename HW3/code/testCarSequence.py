import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation
sequences = np.load('../data/carseq.npy')
num_frames = sequences.shape[2]

rects = np.zeros((num_frames, 4))
rects[0,:] = [59,116,145,151]

for i in range(0, num_frames-1):
    It = sequences[:,:,i]
    It1 = sequences[:,:,i+1]
    rect = rects[i,:]
    p = LucasKanade(It, It1, rect)
    rects[i+1,:] = [rects[i,0]+p[0],rects[i,1]+p[1],rects[i,2]+p[0],rects[i,3]+p[1]]
    if i==1 or i==100 or i==200 or i==300 or i==400:
        height = rects[i,3] - rects[i,1]
        width = rects[i,2] - rects[i,0]
        x,y = rects[i,0],rects[i,1]
        rectangle = patches.Rectangle((x,y),width,height,linewidth=1,edgecolor='r',facecolor='none')

        fig,ax = plt.subplots(1)
        ax.imshow(It1,cmap='gray')
        ax.add_patch(rectangle)
        name = '../temp_results/nocorr-frame'+str(i)+'.png'
        plt.axis('off')
        plt.savefig(name, bbox_inches='tight')

np.save('./carseqrects.npy', rects)