import numpy as np
import scipy.io
import string
import pickle
from nn import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 5
learning_rate = 1e-2
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(1024,hidden_size,params,'layer1')
initialize_weights(hidden_size,36,params,'output')

saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights_init.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

def apply_gradient(params, name, learning_rate):
        W = params['W' + name]
        b = params['b' + name]

        grad_W = params['grad_W' + name]
        grad_b = params['grad_b' + name]

        W = W - learning_rate*grad_W
        b = b - learning_rate*grad_b

        params['W'+name] = W
        params['b'+name] = b

plot_train_loss = []
plot_train_acc = []
plot_valid_loss = []
plot_valid_acc = []
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    
    for xb,yb in batches:
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        loss, acc = compute_loss_and_acc(yb, probs)
        
        total_loss += loss
        total_acc += acc
        
        delta1 = probs
        yb_idx = np.argmax(yb, axis=1)
        delta1[np.arange(probs.shape[0]),yb_idx] -= 1
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)

        apply_gradient(params, 'output', learning_rate)
        apply_gradient(params, 'layer1', learning_rate)
    
    total_acc = total_acc/batch_num
    total_loss = total_loss/len(train_x)
    
    plot_train_loss.append(total_loss)
    plot_train_acc.append(total_acc)

    h1 = forward(valid_x, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
    vloss, vacc = compute_loss_and_acc(valid_y, probs)
    vloss = vloss/len(valid_x)
    plot_valid_acc.append(vacc)
    plot_valid_loss.append(vloss)

    if itr % 2 == 0:
        print("Train itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
        print("Valid itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,vloss,vacc))

x = np.arange(0,max_iters)
f, (ax1, ax2) = plt.subplots(1,2)
# plotting losses
f.suptitle('Number of epochs vs Loss and Accuracy')
ax1.plot(x, plot_train_loss)
ax1.plot(x, plot_valid_loss)
ax1.legend(['Train Loss', 'Valid Loss'])
ax1.set(xlabel='Num. Epochs', ylabel='Loss')
# plotting accuracies
ax2.plot(x, plot_train_acc)
ax2.plot(x, plot_valid_acc)
ax2.legend(['Train Accuracy', 'Valid Accuracy'])
ax2.set(xlabel='Num. Epochs', ylabel='Accuracy')
plt.show()

if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()

saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

#  Q3.1.2
params = pickle.load(open('q3_weights.pickle', 'rb'))
h1 = forward(test_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)
test_loss, test_acc = compute_loss_and_acc(test_y, probs)

# Q 3.1.3
params_init = pickle.load(open('q3_weights_init.pickle', 'rb'))
W_init = params_init['Wlayer1']
W_final = params['Wlayer1']

W_init = np.reshape(W_init, (32,32,64))
W_final = np.reshape(W_final, (32,32,64))

fig = plt.figure(1)
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 axes_pad=0,  # pad between axes in inch.
                 )

for i in range(W_init.shape[2]):
    grid[i].imshow(W_init[:,:,i], cmap='gray')  # The AxesGrid object work as a list of axes.
    grid[i].axis('off')
    grid[i].set_xticks([])
    grid[i].set_yticks([])

plt.show()

# Q3.1.4
params = pickle.load(open('q3_weights.pickle', 'rb'))
confusion_matrix = np.zeros((test_y.shape[1],test_y.shape[1]))
h1 = forward(train_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)
train_loss, train_acc = compute_loss_and_acc(train_y, probs)

for a,p in zip(train_y, probs):
    pred_class = np.argmax(p)
    true_class = np.argmax(a)
    confusion_matrix[true_class][pred_class] += 1

plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()