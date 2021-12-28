import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import myResnet
import resnet

torch.cuda.empty_cache()
"""Load datasets"""

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
NONE = ['NONE']  # label for empty space
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
            self.labels = open(self.label_path, 'r').read().split('\n')[:-1][:1000]
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
                label_oh += encode('NONE')

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
train_dl = DataLoader(train_ds, batch_size=256, num_workers=4)
test_dl = DataLoader(test_ds, batch_size=1, num_workers=4)
# train_dl = DataLoader(train_ds, batch_size=128)
# test_dl = DataLoader(test_ds, batch_size=1)

"""To CUDA for local run"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GPUID = '1'  # define GPUID
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

"""Problem 1: Design LSTM model for catcha image recognition. (10 points)"""
class LSTM(nn.Module):
    def __init__(self, cnn_dim, hidden_size, vocab_size, num_layers=1):
        super(LSTM, self).__init__()

        # define the properties
        self.cnn_dim = cnn_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=self.vocab_size, hidden_size=hidden_size)

        # output fully connected layer
        self.fc_in = nn.Linear(in_features=self.cnn_dim, out_features=self.vocab_size)
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

        # embedding layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.vocab_size)

        # activations
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, captions):

        batch_size = features.size(0)
        cnn_dim = features.size(1)  # for what?

        hidden_state = torch.zeros((batch_size, self.hidden_size)).cuda()
        cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()

        # define the output tensor placeholder
        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).cuda()  # 128 * 7 * 37

        # embed the captions
        # captions_embed = self.embed(captions)
        ##############################################################################
        # force_teach_threshold = 0.3
        # avgpool = nn.AdaptiveAvgPool2d(1, 1)
        # features = avgpool(features)
        features = torch.flatten(features, 1)
        features = self.fc_in(features)

        # for each word: 128(batch) * 7 (chars) * 37 (prob)
        for t in range(captions.size(1)):  # 0 ~ 6
            if t == 0:
                # t=0, input = features
                hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))
            else:
                # t > 1, input = embedded label one hot(t-1) or output(t-1)
                # if teach_ratio > force_teach_threshold:
                # hidden_state, cell_state = self.lstm_cell(out, (hidden_state, cell_state))
                # else:
                hidden_state, cell_state = self.lstm_cell(captions[:, t - 1, :], (hidden_state, cell_state))
            out = self.fc_out(hidden_state)  # 8 -> 37
            outputs[:, t, :] = self.softmax(out)
        ##############################################################################
        return outputs


"""Problem 2: 

*   1.Connect CNN model to the desinged LSTM model.
*   2.Replace ResNet to your own CNN model from Assignment3.
"""

##############################################################################
"""ResNet"""
# CNN
# cnn_model_path = './trainied_resnet18_only.pth'
# lstm_model_path = './trained_lstm_model.pth'
cnn_model_path = './trained_myResnet_only2.pth'
device = torch.device("cuda")
model = myResnet.betternet()
# model = resnet.resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=512, out_features=ALL_CHAR_SET_LEN * MAX_CAPTCHA, bias=True)
model.load_state_dict(torch.load(cnn_model_path))
model.to(device)
model.eval()

"""TEST"""


def get_char_count(arg1):
    c0 = ALL_CHAR_SET[np.argmax(arg1.cpu().tolist()[0:ALL_CHAR_SET_LEN])]
    c1 = ALL_CHAR_SET[np.argmax(arg1.cpu().tolist()[ALL_CHAR_SET_LEN:ALL_CHAR_SET_LEN * 2])]
    c2 = ALL_CHAR_SET[np.argmax(arg1.cpu().tolist()[ALL_CHAR_SET_LEN * 2:ALL_CHAR_SET_LEN * 3])]
    c3 = ALL_CHAR_SET[np.argmax(arg1.cpu().tolist()[ALL_CHAR_SET_LEN * 3:ALL_CHAR_SET_LEN * 4])]
    c4 = ALL_CHAR_SET[np.argmax(arg1.cpu().tolist()[ALL_CHAR_SET_LEN * 4:ALL_CHAR_SET_LEN * 5])]
    c5 = ALL_CHAR_SET[np.argmax(arg1.cpu().tolist()[ALL_CHAR_SET_LEN * 5:ALL_CHAR_SET_LEN * 6])]
    c6 = ALL_CHAR_SET[np.argmax(arg1.cpu().tolist()[ALL_CHAR_SET_LEN * 6:ALL_CHAR_SET_LEN * 7])]
    return c0, c1, c2, c3, c4, c5, c6


char_correct = 0
word_correct = 0
total = 0

# betternet_model.eval()
# lstm_model.eval()

with torch.no_grad():
    for step, (img, label_oh, label) in enumerate(test_dl):
        char_count = 0
        img = Variable(img).cuda()
        label_oh = Variable(label_oh.float()).cuda()
        pred_cnn, feature = model(img)
        # pred_cnn = betternet(img)
        batch_size, _ = label_oh.shape
        # pred_cnn = pred_cnn.reshape(batch_size, ALL_CHAR_SET_LEN, MAX_CAPTCHA)  # 1, 37, 7 reshape
        # captions = torch.argmax(label_oh, 1)  # most probable 7 words
        # feature = torch.flatten(pred_cnn, 1)
        # captions = label_oh.permute(0, 2, 1)  # (1, 7, 37)
        # pred_lstm = lstm(feature, captions)  # (1, 7, 37)
        # pred_lstm = torch.flatten(pred_lstm, 1)  # (1, 259)

        label_len = label[0]
        outputs = pred_cnn.squeeze(0)
        # outputs = pred_lstm.squeeze(0)
        label_oh = label_oh.squeeze(0)

        c0, c1, c2, c3, c4, c5, c6 = get_char_count(outputs)
        d0, d1, d2, d3, d4, d5, d6 = get_char_count(label_oh)

        c = '%s%s%s%s%s%s%s' % (c0, c1, c2, c3, c4, c5, c6)
        d = '%s%s%s%s%s%s%s' % (d0, d1, d2, d3, d4, d5, d6)

        char_count += (c0 == d0) + (c1 == d1) + (c2 == d2) + (c3 == d3) + (c4 == d4) + (c5 == d5) + (c6 == d6)
        char_correct += char_count
        #
        print("predict:", c)
        print("label:", d)

        if bool(str(label[0]) in str(c)):
            word_correct += 1

        total += 1

print(100/7*char_correct/total)
print(100*word_correct/total)

"""END TEST"""
