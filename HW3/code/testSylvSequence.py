import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanadeBasis import LucasKanadeBasis
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation
sequences = np.load('../data/sylvseq.npy')
bases = np.load('../data/sylvbases.npy')
num_frames = sequences.shape[2]

rects_bases = np.zeros((num_frames, 4))
rects_bases[0,:] = [101,61,155,107]

rects = np.zeros((num_frames,4))
rects[0,:] = [101,61,155,107]

for i in range(0, num_frames-1):
    It = sequences[:,:,i]
    It1 = sequences[:,:,i+1]
    rect_bases = rects_bases[i,:]
    p = LucasKanadeBasis(It, It1, rect_bases, bases)
    rects_bases[i+1,:] = [rects_bases[i,0]+p[0],rects_bases[i,1]+p[1],rects_bases[i,2]+p[0],rects_bases[i,3]+p[1]]

    rect = rects[i,:]
    p = LucasKanade(It, It1, rect)
    rects[i+1,:] = [rects[i,0]+p[0],rects[i,1]+p[1],rects[i,2]+p[0],rects[i,3]+p[1]]
    if i==1 or i==200 or i==300 or i==350 or i==400:
        height = rects_bases[i,3] - rects_bases[i,1]
        width = rects_bases[i,2] - rects_bases[i,0]
        x,y = rects_bases[i,0],rects_bases[i,1]
        rectangle = patches.Rectangle((x,y),width,height,linewidth=1,edgecolor='y',facecolor='none')

        height1 = rects[i,3] - rects[i,1]
        width1 = rects[i,2] - rects[i,0]
        x,y = rects[i,0],rects[i,1]
        rectangle1 = patches.Rectangle((x,y),width,height,linewidth=1,edgecolor='g',facecolor='none')

        fig,ax = plt.subplots(1)
        ax.imshow(It,cmap='gray')
        ax.add_patch(rectangle)
        ax.add_patch(rectangle1)
        name = '../results/bases/frame'+str(i)+'.png'
        plt.axis('off')
        plt.savefig(name, bbox_inches='tight')

np.save('./sylvseqrects.npy', rects_bases)