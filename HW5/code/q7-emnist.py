from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from q4 import *
import os
import skimage.io
import string

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

def fetchString(row, bw, pad_dist=2):
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
        crop_img = np.pad(crop_img, pad_dist, 'constant')
        crop_img = crop_img.T
        batch.append(crop_img.flatten())
    return np.array(batch)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 47)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}, Loss: {:.6f}'.format(epoch, loss.item()))

if __name__ == "__main__":
    predicted_txt = []
    gt1 = [['DEEPLEARNING','DEEPERLEARNING','DEEPESTLEARNING'], ['TODOLIST','1MAKEATODOLIST','2CHECKOFFTHEFIRST','THINGONTODOLIST','3REALIZEYOUHAVEALREADY','COMPLETED2THINGS','4REWARDYOURSELFWITH','ANAP'], ['ABCDEFG','HIJKLMN','OPQRSTU','VWXYZ','1234567890'], ['HAIKUSAREEASY','BUTSOMETIMESTHEYDONTMAKESENSE','REFRIGERATOR']]
    letters = {'0': 48, '1': 49, '2': 50, '3': 51, '4': 52, '5': 53, '6': 54, '7': 55, '8': 56, '9': 57, '10': 65, '11': 66, '12': 67, '13': 68, '14': 69, '15': 70, '16': 71, '17': 72, '18': 73, '19': 74, '20': 75, '21': 76, '22': 77, '23': 78, '24': 79, '25': 80, '26': 81, '27': 82, '28': 83, '29': 84, '30': 85, '31': 86, '32': 87, '33': 88, '34': 89, '35': 90, '36': 97, '37': 98, '38': 100, '39': 101, '40': 102, '41': 103, '42': 104, '43': 110, '44': 113, '45': 114, '46': 116}

    final_letters = []
    for k,v in letters.items():
        final_letters.append(chr(v))
    
    parser = argparse.ArgumentParser(description='PyTorch EMNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N')
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
    
    train_loader = torch.utils.data.DataLoader(
        datasets.EMNIST('../data/emnist', split='balanced', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs+1):
        train(args, model, device, train_loader, optimizer, epoch)

    for img in os.listdir('../images'):
        im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
        bboxes, bw = findLetters(im1)
        bw = np.invert(bw)
        bboxes = np.array(bboxes)
        rows = fetchRows(bboxes)

        bw = np.invert(bw)
        predicted_text = []
        
        for i in range(0,rows.shape[0]):
            row = np.array(rows[i])
            row = row[row[:, 1].argsort()]
            batch = fetchString(row, bw, 2)

            strings = [skimage.transform.resize(batch[i].reshape(32,32), (28,28)) for i in range(batch.shape[0])]
            new_strings = np.array(strings)
            strings = np.reshape(new_strings, (-1,1,28,28))

            model.eval()
            output = model(torch.from_numpy(strings).float())
            predicted_labels = torch.argmax(output, dim=1)
            predicted_labels = predicted_labels.numpy()
            text = ""
            for i in predicted_labels:
                # print (i)
                text += final_letters[i]
            print (text)
            predicted_text.append(text)

        predicted_txt.append(predicted_text)

    count = np.zeros((4,8))
    str_len = np.zeros((4,8))
    for i, (row1,row2) in enumerate(zip(gt1, predicted_txt)):
        for j, (a, b) in enumerate(zip(row1, row2)):
            str_len[i][j] = len(a)
            for c,d in zip(a,b):
                if c == d:
                    count[i][j] += 1

    acc = np.divide(count, str_len)
    acc = np.nan_to_num(acc, 0)
    print (np.true_divide(acc.sum(1),(acc!=0).sum(1)))

    # letter_pred = ['DEEPLEARNINGDEEPERLEARNINGDEEPESTLEARNING','T0D0LIST2MAKEATODOLIST2CHECK0FFTHEFIRSTTHING0NTODOLIST3RBALIZEYOUHWVEALREADYCOMPLETED2THINGS4REWARDY0URSELFWITHANAP', 'ABCDEFGHIJKLMN0PQRSTUVWXYZ1234567890','HAIKUSAREEnASYBUTSOMETIMESTKEYDONTMAKESENSEREFRIGERATOR']

    # gt = ['DEEPLEARNINGDEEPERLEARNINGDEEPESTLEARNING','TODOLIST1MAKEATODOLIST2CHECKOFFTHEFIRSTTHINGONTODOLIST3REALIZEYOUHAVEALREADYCOMPLETED2THINGS4REWARDYOURSELFWITHANAP', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890', 'HAIKUSAREEASYBUTSOMETIMESTHEYDONTMAKESENSEREFRIGERATOR']

    # inv_count = [0,0,0,0]
    # count = 0
    # for i, (str1, str2) in enumerate(zip(gt, letter_pred)):
    #     for a,b in zip(str1, str2):
    #         if a == b:
    #             inv_count[i] += 1

    # str_len = [len(x) for x in gt]
    # total_len = sum(str_len)
    # total_count = sum(inv_count)
    # inv_acc = []
    # for a,b in zip(str_len, inv_count):
    #     inv_acc.append(b/a)
    # total_acc = total_count/total_len
    # print ("Total Accuracy {:.03f}".format(total_acc))
    # print ("Individual Accuracies", np.round(inv_acc, decimals=4))
    # print (acc)