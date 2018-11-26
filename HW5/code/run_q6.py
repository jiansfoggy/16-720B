import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

train_mean = np.mean(train_x, axis=0)
train_mean = np.reshape(train_mean, (1,train_mean.shape[0]))
train_x = np.subtract(train_x, train_mean)

dim = 32
# do PCA
u,s,v = np.linalg.svd(train_x)
temp_P = v[:dim,:].T
P = np.matmul(temp_P, temp_P.T)

# rebuild a low-rank version
lrank = np.linalg.matrix_rank(P)

# rebuild it
recon = np.add(np.matmul(train_x, P), train_mean)
train_x = np.add(train_x, train_mean)
train_psnr = psnr(train_x, recon)

# for i in range(5):
#     plt.subplot(2,1,1)
#     plt.imshow(train_x[i].reshape(32,32).T)
#     plt.subplot(2,1,2)
#     plt.imshow(recon[i].reshape(32,32).T)
#     plt.show()

# build valid dataset
valid_x = np.subtract(valid_x, train_mean)
recon_valid = np.add(np.matmul(valid_x, P), train_mean)
valid_x = np.add(valid_x, train_mean)

total = []
for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
print (np.array(total).mean())