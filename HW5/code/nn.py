import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    a = np.sqrt(6)/np.sqrt(in_size+out_size)
    W, b = np.random.uniform(-a,a,(in_size, out_size)), np.zeros(out_size)
    
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1/(1+np.exp(-x))
    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # your code here
    pre_act = np.matmul(X, W) + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    s = np.max(x, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(x-s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x/div

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None
    actual_classes = np.argmax(probs, axis=1)
    b = np.zeros((actual_classes.shape[0], probs.shape[1]))
    b[np.arange(actual_classes.shape[0]), actual_classes] = 1
    correct_count = np.sum(np.all(np.equal(y, b), axis=1))
    acc = correct_count/y.shape[0]

    loss = -1*np.einsum('ij,ij->i', y, np.log(probs))
    loss = np.sum(loss)
    return loss, acc

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X

    act_deriv = activation_deriv(post_act)*delta

    grad_W = np.matmul(X.T, act_deriv)
    grad_b = np.sum(act_deriv, axis=0)
    grad_X = np.matmul(W, act_deriv.T).T

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    idx = np.random.permutation(x.shape[0])
    x_temp = x[idx,:]
    y_temp = y[idx,:]
    num_batches = x.shape[0]/batch_size
    batch_x = np.split(x_temp, num_batches)
    batch_y = np.split(y_temp, num_batches)

    for a,b in zip(batch_x,batch_y):
        batches.append((a,b))

    return batches