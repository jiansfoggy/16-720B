from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import scipy.io
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 36)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}, Loss: {:.6f}'.format(epoch, loss.item()))
    
    acc = correct / len(train_loader.dataset)
    return total_loss, acc

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {:.02f}%'.format(test_loss, 100. * correct / len(test_loader.dataset)))

class Nist36Dataset(Dataset):
    def __init__(self, path, train=True):
        if train:
            val = 'train'
        else:
            val = 'test'

        data = scipy.io.loadmat(path)
        data_x, data_y =  data[val+'_data'], data[val+'_labels']
        data_x = np.reshape(data_x, (data_x.shape[0],1,32,32))
        self.data_x = torch.from_numpy(data_x).float()
        self.data_y = torch.argmax(torch.from_numpy(data_y), dim=1).long()

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return (self.data_x[idx], self.data_y[idx])

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch NIST36 Example')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N')
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    train_data = scipy.io.loadmat('../data/nist36_train.mat')
    valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
    test_data = scipy.io.loadmat('../data/nist36_test.mat')

    train_x, train_y = train_data['train_data'], train_data['train_labels']
    valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
    test_x, test_y = test_data['test_data'], test_data['test_labels']

    train_loader = torch.utils.data.DataLoader(
        Nist36Dataset(path='../data/nist36_train.mat', train=True),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        Nist36Dataset(path='../data/nist36_test.mat', train=False),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    plot_loss = []
    plot_acc = []
    for epoch in range(1, 50):
        iteration_loss, iteration_acc = train(args, model, device, train_loader, optimizer, epoch)
        plot_loss.append(iteration_loss)
        plot_acc.append(iteration_acc)
        test(args, model, device, test_loader)

    x = np.arange(1,50)
    plt.plot(x, plot_loss)
    plt.xlabel('Num. Epochs')
    plt.ylabel('Average Training Loss')
    plt.title('Loss vs Num. Epochs')
    plt.savefig('../results/q7-nist36-cnn-loss.png')
    
    plt.clf()
    
    x = np.arange(1,50)
    plt.plot(x, plot_acc)
    plt.xlabel('Num. Epochs')
    plt.ylabel('Training Accuracy')
    plt.title('Accuracy vs Num. Epochs')
    plt.savefig('../results/q7-nist36-cnn-acc.png')

if __name__ == '__main__':
    main()