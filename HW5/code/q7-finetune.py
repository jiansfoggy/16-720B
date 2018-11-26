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

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Flowers Example')
    parser.add_argument('--train_dir', default='../data/oxford-flowers17/train')
    parser.add_argument('--val_dir', default='../data/oxford-flowers17/val')
    parser.add_argument('--test_dir', default='../data/oxford-flowers17/test')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N')
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--num_epochs1', default=24, type=int)
    
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform_data = transforms.Compose([
        transforms.Scale(224),
        transforms.RandomSizedCrop(224),
        transforms.ToTensor()])

    train_dset = ImageFolder(args.train_dir, transform = transform_data)
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, **kwargs)

    val_dset = ImageFolder(args.val_dir, transform = transform_data)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=True, **kwargs)

    test_dset = ImageFolder(args.test_dir, transform = transform_data)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=True, **kwargs)

    model = models.squeezenet1_1(pretrained=True)
    print (model)
    num_classes = len(train_dset.classes)

    model.type(dtype)
    loss_fn = nn.CrossEntropyLoss().type(dtype)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.num_epochs1):
        # Run an epoch over the training data.
        print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
        run_epoch(model, loss_fn, train_loader, optimizer, dtype)

        # Check accuracy on the train and val sets.
        train_acc = check_accuracy(model, train_loader, dtype)
        val_acc = check_accuracy(model, val_loader, dtype)
        print('Train accuracy: ', train_acc)
        print('Val accuracy: ', val_acc)
        print()

def run_epoch(model, loss_fn, loader, optimizer, dtype):
  """
  Train the model for one epoch.
  """
  # Set the model to training mode
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

if __name__ == '__main__':
    main()