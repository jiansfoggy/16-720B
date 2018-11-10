import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation
sequences = np.load('../data/carseq.npy')
num_frames = sequences.shape[2]

rects = np.zeros((num_frames, 4))
rects_correct = np.zeros((num_frames, 4))
rects[0,:] = [59,116,145,151]
rects_correct[0,:] = [59,116,145,151]
tol = 3

p0 = np.zeros(2)
for i in range(0, num_frames-1):
    It = sequences[:,:,i]
    It1 = sequences[:,:,i+1]
    rect = rects_correct[i,:]
    rect1 = rects[i,:]
    p = LucasKanade(It, It1, rect)
    p_dash = LucasKanade(It, It1, rect1)
    rects[i+1,:] = [rects[i,0]+p_dash[0],rects[i,1]+p_dash[1],rects[i,2]+p_dash[0],rects[i,3]+p_dash[1]]
    rects_correct[i+1,:] = [rects_correct[i,0]+p[0],rects_correct[i,1]+p[1],rects_correct[i,2]+p[0],rects_correct[i,3]+p[1]]

    p0 = rects_correct[i+1,:] - rects_correct[0,:]
    p0 = np.array([p0[0], p0[1]])
    p_new = LucasKanade(sequences[:,:,0], It1, rects_correct[0,:],p0)
    temp = p_new-p0
    delta = temp - p

    if np.linalg.norm(delta) < tol:
        p = temp
    else:
        p = p
    
    rects_correct[i+1,:] = [rects_correct[i,0]+p[0],rects_correct[i,1]+p[1],rects_correct[i,2]+p[0],rects_correct[i,3]+p[1]]
    
    if i==1 or i==100 or i==200 or i==300 or i==400:
        height = rects_correct[i,3] - rects_correct[i,1]
        width = rects_correct[i,2] - rects_correct[i,0]
        x,y = rects_correct[i,0],rects_correct[i,1]
        rectangle = patches.Rectangle((x,y),width,height,linewidth=1,edgecolor='k',facecolor='none')

        height = rects[i,3] - rects[i,1]
        width = rects[i,2] - rects[i,0]
        x,y = rects[i,0],rects[i,1]
        rectangle1 = patches.Rectangle((x,y),width,height,linewidth=1,edgecolor='r',facecolor='none')

        fig,ax = plt.subplots(1)
        ax.imshow(It1,cmap='gray')
        ax.add_patch(rectangle)
        ax.add_patch(rectangle1)
        name = '../temp_results/corr-frame'+str(i)+'.png'
        plt.axis('off')
        plt.savefig(name, bbox_inches='tight')
    
np.save('./carseqrects-wcrt.npy', rects_correct)