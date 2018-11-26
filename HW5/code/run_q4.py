import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import pickle
import string
import skimage
import skimage.measure as skmeasure
import skimage.color as skcolor
import skimage.restoration as skrestore
import skimage.io as sio
import skimage.filters as skfilter
import skimage.morphology as skmorph
import skimage.segmentation as sksegment

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def fetchRows(bboxes):
    rows = []
    row = []
    for i in range(bboxes.shape[0] - 1):
        if abs(bboxes[i][0] - bboxes[i+1][0]) < 100:
            row.append(bboxes[i])
        else:
            row.append(bboxes[i])
            rows.append(row)
            row = []
    if i == bboxes.shape[0] - 2:
        row.append(bboxes[i+1])
        rows.append(row)

    rows = np.array(rows)
    return rows

def get_batch(row, bw, pad_dist=2):
    batch = []
    for j in range(row.shape[0]):
        minr, minc, maxr, maxc = row[j]
        
        minr -= pad_dist
        minc -= pad_dist
        maxr += pad_dist
        maxc += pad_dist

        rowcenter = round((maxr + minr + 1) / 2)
        colcenter = round((maxc + minc + 1) / 2)

        lengthToCrop = max(maxr - minr + 1, maxc - minc + 1)
        min_r = int(round(rowcenter - (lengthToCrop / 2)))
        max_r = int(round(rowcenter + (lengthToCrop / 2)))
        min_c = int(round(colcenter - (lengthToCrop / 2)))
        max_c = int(round(colcenter + (lengthToCrop / 2)))

        crop_img = bw[min_r: max_r, min_c: max_c]
        crop_img = skfilter.gaussian(crop_img)
        crop_img = skimage.transform.resize(crop_img, (28, 28))
        crop_img = 1-np.pad(crop_img, pad_dist, 'constant')
        crop_img = crop_img.T
        batch.append(crop_img.flatten())
    return np.array(batch)

letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
params = pickle.load(open('q3_weights.pickle','rb'))

predicted_txt = []
gt = ['DEEPLEARNINGDEEPERLEARNINGDEEPESTLEARNING', 'TODOLIST1MAKEATODOLIST2CHECKOFFTHEFIRSTTHINGONTODOLIST3REALIZEYOUHAVEALREADYCOMPLETED2THINGS4REWARDYOURSELFWITHANAP', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890', 'HAIKUSAREEASYBUTSOMETIMESTHEYDONTMAKESENSEREFRIGERATOR']

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    bw = np.invert(bw)
    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    bboxes = np.array(bboxes)
    rows = fetchRows(bboxes)

    bw = np.invert(bw)
    predicted_text = ""
    for i in range(0,rows.shape[0]):
        row = np.array(rows[i])
        row = row[row[:, 1].argsort()]
        batch = get_batch(row, bw, pad_dist=2)

        h1 = forward(batch, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        predicted_labels = np.argmax(probs, axis=1)
        temp_pred_text = "".join(letters[predicted_labels])
        print (temp_pred_text)
        predicted_text += temp_pred_text

    predicted_txt.append(predicted_text)

inv_count = [0,0,0,0]
count = 0
for i, (str1, str2) in enumerate(zip(gt, predicted_txt)):
    for a,b in zip(str1, str2):
        if a == b:
            inv_count[i] += 1

str_len = [len(x) for x in gt]
total_len = sum(str_len)
total_count = sum(inv_count)
inv_acc = []
for a,b in zip(str_len, inv_count):
    inv_acc.append(b/a)
total_acc = total_count/total_len
print ("Total Accuracy {:.03f}".format(total_acc))
print ("Individual Accuracies", inv_acc)
print ("Predicted Strings: ", predicted_txt)