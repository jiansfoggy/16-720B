import numpy as np
import scipy.io
from nn import *
from collections import Counter
from skimage.measure import compare_psnr as psnr
import matplotlib.pyplot as plt
import pickle

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# initialize layers here
initialize_weights(train_x.shape[1],hidden_size,params,'layer1')
initialize_weights(hidden_size,hidden_size,params,'hidden')
initialize_weights(hidden_size,hidden_size,params,'hidden2')
initialize_weights(hidden_size,train_x.shape[1],params,'output')

def apply_gradient(params, name, learning_rate):
        W = params['W' + name]
        b = params['b' + name]

        grad_W = params['grad_W' + name]
        grad_b = params['grad_b' + name]

        m_W = params['m_W'+name]
        m_b = params['m_b'+name]

        m_W = 0.9*m_W - learning_rate*grad_W
        m_b = 0.9*m_b - learning_rate*grad_b

        W += m_W
        b += m_b

        params['m_W'+name] = m_W
        params['m_b'+name] = m_b

        params['W'+name] = W
        params['b'+name] = b

plot_train_loss = []
# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # apply forward pass
        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'hidden', relu)
        h3 = forward(h2, params, 'hidden2', relu)
        probs = forward(h3, params, 'output', sigmoid)
        
        # get loss
        # loss = (np.square(xb - probs)).mean(axis=None)
        loss = np.sum(np.square(xb - probs))
        total_loss += loss
        
        # apply back-prop
        delta1 = -2*(xb-probs)
        delta2 = backwards(delta1,params,'output',sigmoid_deriv)
        delta3 = backwards(delta2,params,'hidden2',relu_deriv)
        delta4 = backwards(delta3,params,'hidden',relu_deriv)
        backwards(delta4,params,'layer1',relu_deriv)

        # apply gradient descent
        apply_gradient(params, 'output', learning_rate)
        apply_gradient(params, 'hidden2', learning_rate)
        apply_gradient(params, 'hidden', learning_rate)
        apply_gradient(params, 'layer1', learning_rate)
        
    plot_train_loss.append(total_loss)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        # print ("Learning Rate New: ", learning_rate)
        learning_rate *= 0.9

x = np.arange(0,max_iters)
plt.plot(x, plot_train_loss)
plt.xlabel('Num. Epochs')
plt.ylabel('Total squared loss')
plt.title('Loss vs Num. Epochs')
plt.show()

with open('q5_weights.pickle', 'wb') as outputfile:
    pickle.dump(params, outputfile)

# params = pickle.load(open('q5_weights.pickle', 'rb'))

# visualize some results
# Q5.3.1
idx = [2837, 2835, 2961, 2952, 1353, 1360, 1295, 1241, 252, 256]
temp_x = valid_x[idx,:]

h1 = forward(temp_x,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)
loss = np.sum(np.square(temp_x - out))
print (loss)

for i in range(10):
    plt.subplot(2,1,1)
    plt.imshow(temp_x[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(out[i].reshape(32,32).T)
    plt.show()

# # evaluate PSNR
# # Q5.3.2
h1 = forward(valid_x,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)
total_psnr = psnr(valid_x, out)
print (total_psnr)