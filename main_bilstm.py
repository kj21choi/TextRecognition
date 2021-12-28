import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

import resnet
import myResnet
import myInceptionNet
from Attention import Attention

torch.cuda.empty_cache()
"""Load datasets"""

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
# NONE = ['NONE']  # label for empty space
NONE = ['-']  # label for empty space
ALL_CHAR_SET = NUMBER + ALPHABET + NONE
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 7

# print(ALL_CHAR_SET.index('NONE'))


def encode(a):
    onehot = [0] * ALL_CHAR_SET_LEN
    idx = ALL_CHAR_SET.index(a)
    onehot[idx] += 1
    return onehot


# modified dataset class
class Mydataset(Dataset):
    def __init__(self, img_path, label_path, is_train=True, transform=None):
        self.path = img_path
        self.label_path = label_path
        if is_train:
            self.img = os.listdir(self.path)[:10000]
            self.labels = open(self.label_path, 'r').read().split('\n')[:-1][:10000]
        else:
            self.img = os.listdir(self.path)[:1000]
            self.labels = open(self.label_path, 'r').read().split('\n')[:-1][:1000]

        self.transform = transform
        self.max_length = MAX_CAPTCHA

    def __getitem__(self, idx):
        img_path = self.img[idx]
        img = Image.open(f'{self.path}/{self.img[idx]}')
        img = img.convert('L')
        label = self.labels[idx]
        label_oh = []
        # one-hot for each character
        for i in range(self.max_length):
            if i < len(label):
                label_oh += encode(label[i])
            else:
                # label_oh += [0]*ALL_CHAR_SET_LEN
                # label_oh += encode('NONE')
                label_oh += encode('-')

        if self.transform is not None:
            img = self.transform(img)
        return img, np.array(label_oh), label

    def __len__(self):
        return len(self.img)


transform = transforms.Compose([
    transforms.Resize([160, 60]),
    transforms.ToTensor(),
    ##############################################################################
    transforms.Normalize((0.8958,), (0.1360,)),
    ##############################################################################
])

"""Loading DATA"""
# Change to your own data foler path!
gPath = ''

train_ds = Mydataset(gPath + 'Data/train/', gPath + 'Data/train.txt', transform=transform)
test_ds = Mydataset(gPath + 'Data/train/', gPath + 'Data/train.txt', False, transform)
train_dl = DataLoader(train_ds, batch_size=128, num_workers=4)
test_dl = DataLoader(test_ds, batch_size=1, num_workers=4)
# train_dl = DataLoader(train_ds, batch_size=128)
# test_dl = DataLoader(test_ds, batch_size=1)

# mean = 0.
# std = 0.
# for step, i in enumerate(train_dl):
#     img, label_oh, label = i
#     img = Variable(img).cuda()
#     mean += img.mean()
#     std += img.std()
# print(step)
# mean /= step
# std /= step
# print("mean=", mean, " std=", std)


"""To CUDA for local run"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GPUID = '1'  # define GPUID
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

"""Problem 1: Design LSTM model for catcha image recognition. (10 points)"""
class LSTM(nn.Module):
    def __init__(self, cnn_dim, hidden_size, vocab_size, num_layers=2):
        super(LSTM, self).__init__()

        # define the properties
        self.cnn_dim = cnn_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # lstm
        self.bilstm = nn.GRU(input_size=self.hidden_size, hidden_size=hidden_size, num_layers=2,
                              bidirectional=True, bias=True, batch_first=True)

        # output fully connected layer
        self.fc_in = nn.Linear(in_features=self.cnn_dim, out_features=self.vocab_size)
        self.fc_out = nn.Linear(in_features=self.hidden_size * 2, out_features=self.vocab_size)

        # embedding layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.vocab_size)

        # activations
        self.softmax = nn.Softmax(dim=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, features, isTraining=True):

        ##############################################################################
        hidden = self.init_hidden(batch_size, next(self.parameters()).is_cuda)
        outputs, hidden = self.bilstm(features, hidden)
        outputs = self.fc_out(outputs)
        if not isTraining:
            outputs = self.softmax(outputs)
        ##############################################################################
        return outputs

    def init_hidden(self, batch_size, gpu=False):
        h0 = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)).cuda()
        return h0

"""Problem 2: 

*   1.Connect CNN model to the desinged LSTM model.
*   2.Replace ResNet to your own CNN model from Assignment3.
"""

##############################################################################
"""ResNet"""
# CNN
cnn_model_path = './trained_myResnet_only2.pth'
betternet = myResnet.betternet()
betternet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
betternet.fc = nn.Linear(in_features=512, out_features=ALL_CHAR_SET_LEN * MAX_CAPTCHA, bias=True)
betternet.load_state_dict(torch.load(cnn_model_path))
betternet.to(device)
betternet.eval()

##############################################################################

# LSTM
cnn_dim = 512  # resnet18-512
hidden_size = 37  # 8->37
vocab_size = 37  # ALL_CHAR_SET_LEN
lstm = LSTM(cnn_dim=cnn_dim, hidden_size=hidden_size, vocab_size=hidden_size).to(device)
lstm.train()
# loss, optimizer
##############################################################################
# params = list(betternet.parameters()) + list(lstm.parameters())
loss_func = nn.MultiLabelSoftMarginLoss()
nn.utils.clip_grad_norm_(lstm.parameters(), 5)
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001, weight_decay=0.0001)
lr_scheduler = StepLR(optimizer, step_size=500)
##############################################################################


def get_char_count(arg1):
    c0 = ALL_CHAR_SET[np.argmax(arg1.cpu().tolist()[0:ALL_CHAR_SET_LEN])]
    c1 = ALL_CHAR_SET[np.argmax(arg1.cpu().tolist()[ALL_CHAR_SET_LEN:ALL_CHAR_SET_LEN * 2])]
    c2 = ALL_CHAR_SET[np.argmax(arg1.cpu().tolist()[ALL_CHAR_SET_LEN * 2:ALL_CHAR_SET_LEN * 3])]
    c3 = ALL_CHAR_SET[np.argmax(arg1.cpu().tolist()[ALL_CHAR_SET_LEN * 3:ALL_CHAR_SET_LEN * 4])]
    c4 = ALL_CHAR_SET[np.argmax(arg1.cpu().tolist()[ALL_CHAR_SET_LEN * 4:ALL_CHAR_SET_LEN * 5])]
    c5 = ALL_CHAR_SET[np.argmax(arg1.cpu().tolist()[ALL_CHAR_SET_LEN * 5:ALL_CHAR_SET_LEN * 6])]
    c6 = ALL_CHAR_SET[np.argmax(arg1.cpu().tolist()[ALL_CHAR_SET_LEN * 6:ALL_CHAR_SET_LEN * 7])]
    return c0, c1, c2, c3, c4, c5, c6
"""step3: Find hyper-parameters."""

"""TRAINING"""
print_interval = 50
# max_epoch = 1
max_epoch = 50
threshold = 1e-4
isConverged = False
for epoch in range(max_epoch):
    if isConverged:
        break
    for step, i in enumerate(train_dl):
        img, label_oh, label = i
        img = Variable(img).cuda()
        text = Variable(label_oh.float()).cuda()
        label_oh = Variable(label_oh.long()).cuda()
        batch_size, _ = label_oh.shape
        cnn_pred, features = betternet(img)
        ##############################################################################
        optimizer.zero_grad()
        # cnn_pred = cnn_pred.reshape(batch_size, MAX_CAPTCHA, ALL_CHAR_SET_LEN).to(device)

        pred = lstm(text, isTraining=True)
        pred = torch.flatten(pred, 1)  # 128, 259
        # for end-to-end learning
        loss = loss_func(pred, label_oh)  # softmax(128, 259) vs. one-hot(128, 259)

        if loss < threshold:
            isConverged = True

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        ##############################################################################
        if (step + 1) % print_interval == 0:
            print('epoch:', epoch + 1, 'step:', step + 1, 'loss:', loss.item())
        if (epoch + 1) % 101 == 0:
            print("predict:", nn.Softmax(pred[0, :]))
            print("target:", label_oh[0, :])

# cnn_model_path = './trained_cnn_model_bidirect_end_to_end.pth'
lstm_model_path = './trained_lstm_model_bidirect_end_to_end.pth'
print('Finished Training')
# torch.save(betternet.state_dict(), cnn_model_path)
torch.save(lstm.state_dict(), lstm_model_path)
print('Saved Trained Model')

"""TEST"""

char_correct = 0
word_correct = 0
total = 0

# betternet.eval()
lstm.eval()

with torch.no_grad():
    for step, (img, label_oh, label) in enumerate(test_dl):
        char_count = 0
        img = Variable(img).cuda()
        label_oh = Variable(label_oh.long()).cuda()
        cnn_pred, feature = betternet(img)
        batch_size, _ = label_oh.shape
        cnn_pred = cnn_pred.reshape(batch_size, MAX_CAPTCHA, ALL_CHAR_SET_LEN).to(device)
        pred = lstm(cnn_pred, isTraining=False)
        outputs = torch.flatten(pred, 1)  # 128, 259
        # for end-to-end learning

        label_len = label[0]
        outputs = outputs.squeeze(0)
        label_oh = label_oh.squeeze(0)

        c0, c1, c2, c3, c4, c5, c6 = get_char_count(outputs)
        d0, d1, d2, d3, d4, d5, d6 = get_char_count(label_oh)

        c = '%s%s%s%s%s%s%s' % (c0, c1, c2, c3, c4, c5, c6)
        d = '%s%s%s%s%s%s%s' % (d0, d1, d2, d3, d4, d5, d6)

        char_count += (c0 == d0) + (c1 == d1) + (c2 == d2) + (c3 == d3) + (c4 == d4) + (c5 == d5) + (c6 == d6)
        char_correct += char_count

        print("predict:", c)
        print("label:", d)

        if bool(str(label[0]) in str(c)):
            word_correct += 1

        total += 1

print(100/7*char_correct/total)
print(100*word_correct/total)

"""END TEST"""
