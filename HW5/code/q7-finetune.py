from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

dtype = torch.FloatTensor

def run_epoch(model, loss_fn, loader, optimizer, dtype):
  model.train()
  for x, y in loader:
    x_var = Variable(x.type(dtype))
    y_var = Variable(y.type(dtype).long())
    scores = model(x_var)
    loss = loss_fn(scores, y_var)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def check_accuracy(model, loader, dtype):
  # Set the model to eval mode
  model.eval()
  num_correct, num_samples = 0, 0
  for x, y in loader:
    x_var = Variable(x.type(dtype), volatile=True)

    scores = model(x_var)
    _, preds = scores.data.cpu().max(1)
    num_correct += (preds == y).sum()
    num_samples += x.size(0)

  acc = float(num_correct) / num_samples
  return acc

def finetuneSqueeNet(train_dset, train_loader, val_loader, args):
    model = models.squeezenet1_1(pretrained=True)
    # print (model)
    num_classes = len(train_dset.classes)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model.num_classes = num_classes

    model.type(dtype)
    loss_fn = nn.CrossEntropyLoss().type(dtype)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.epochs):
        # Run an epoch over the training data.
        print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
        run_epoch(model, loss_fn, train_loader, optimizer, dtype)

        # Check accuracy on the train and val sets.
        train_acc = check_accuracy(model, train_loader, dtype)
        val_acc = check_accuracy(model, val_loader, dtype)
        print('Train accuracy: ', train_acc)
        print('Val accuracy: ', val_acc)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=3)
        self.fc1 = nn.Linear(27040, 256)
        self.fc2 = nn.Linear(256, 17)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 27040)
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
        total_loss += loss.item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
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

def scratchTrain(args, device, train_loader, test_loader):
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    plot_loss = []
    plot_acc = []
    for epoch in range(1, args.epochs):
        iteration_loss, iteration_acc = train(args, model, device, train_loader, optimizer, epoch)
        plot_loss.append(iteration_loss)
        plot_acc.append(iteration_acc)
        test(args, model, device, test_loader)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Flowers Example')
    parser.add_argument('--train_dir', default='../data/oxford-flowers17/train')
    parser.add_argument('--val_dir', default='../data/oxford-flowers17/val')
    parser.add_argument('--test_dir', default='../data/oxford-flowers17/test')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N')
    parser.add_argument('--epochs', type=int, default=24, metavar='N')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N')
    
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform_data = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()])

    train_dset = ImageFolder(args.train_dir, transform = transform_data)
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, **kwargs)

    val_dset = ImageFolder(args.val_dir, transform = transform_data)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=True, **kwargs)

    test_dset = ImageFolder(args.test_dir, transform = transform_data)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # finetuneSqueeNet(train_dset, train_loader, val_loader, args)
    scratchTrain(args, device, train_loader, test_loader)

if __name__ == '__main__':
    main()