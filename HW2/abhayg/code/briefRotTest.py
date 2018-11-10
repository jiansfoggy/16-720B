import cv2
import numpy as np
import matplotlib.pyplot as plt
from BRIEF import briefLite, briefMatch

def rotate_bound(im, angle):
    (h,w) = im.shape
    (cX,cY) = (w//2, h//2)
    M = cv2.getRotationMatrix2D((cX,cY),-angle,1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0,2] += (nW / 2) - cX
    M[1,2] += (nH / 2) - cY
    return cv2.warpAffine(im, M, (nW, nH))

def plotGraph(angles, matches):
    x_pos = np.arange(len(angles))
    plt.bar(x_pos, matches, align='center', alpha=0.5, width=0.35)
    plt.xticks(x_pos, angles, rotation=90)
    plt.ylabel('Numer of Matches')
    plt.title('Numer of matches vs Angle of Rotations')
    plt.xlabel('Angle of Rotation')
    plt.savefig('../results/rotations_brief_small.png', bbox_inches='tight')

if __name__ == "__main__":
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    if len(im1.shape)==3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if im1.max()>10:
        im1 = np.float32(im1)/255
    match_count = []
    angles = []
    locs1, desc1 = briefLite(im1)
    for i in range(0,360,10):
        print ('{} angles rotated'.format(i))
        im2 = rotate_bound(im1, i)
        locs2, desc2 = briefLite(im2)
        matches = briefMatch(desc1, desc2)
        match_count.append(matches.shape[0])
        angles.append(i)

    plotGraph(angles,match_count)