import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import resnet
import myResnet
import myInceptionNet

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
test_ds = Mydataset(gPath + 'Data/test/', gPath + 'Data/test.txt', False, transform)
# train_dl = DataLoader(train_ds, batch_size=256, num_workers=4)
# test_dl = DataLoader(test_ds, batch_size=1, num_workers=4)
train_dl = DataLoader(train_ds, batch_size=128)
test_dl = DataLoader(test_ds, batch_size=1)

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
    def __init__(self, cnn_dim, hidden_size, vocab_size, num_layers=1):
        super(LSTM, self).__init__()

        # define the properties
        self.cnn_dim = cnn_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=self.vocab_size, hidden_size=hidden_size)
        # self.rnn = nn.RNN(input_size=self.vocab_size, hidden_size=hidden_size, batch_first=True)
        # output fully connected layer
        self.fc_in = nn.Linear(in_features=self.cnn_dim, out_features=self.vocab_size)
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

        # embedding layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.vocab_size)

        # activations
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, captions, isTraining):

        batch_size = features.size(0)
        cnn_dim = features.size(1)

        hidden_state = torch.zeros((batch_size, self.hidden_size)).cuda()
        cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()

        # define the output tensor placeholder
        outputs = torch.zeros((batch_size, captions.size(1), self.vocab_size)).cuda()  # 128 * 7 * 37

        # embed the captions
        # captions_embed = self.embed(captions)  # when training, i will use label one-hot vector
        ##############################################################################
        avgpool = nn.AdaptiveAvgPool2d(1)  # 128 x 512 x 5 x 2
        features = avgpool(features)  # 128 x 512 x 1 x 1
        features = torch.flatten(features, 1)  # 128 x 512
        features = self.fc_in(features)  # 512 -> 37 can you ...?
        # features = self.softmax(features)

        # for each word: 128(batch) * 7 (chars) * 37 (prob)
        for t in range(captions.size(1)):  # 0 ~ 6
            if t == 0:
                # t=0, input = features
                hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))
            else:
                # t > 1, input = embedded label one hot(t-1) or output(t-1)
                # by using Embed lookup table, the same characters are updated simultaneously
                # teacher forcer
                if isTraining:
                    hidden_state, cell_state = self.lstm_cell(captions[:, t - 1, :], (hidden_state, cell_state))
                else:
                    hidden_state, cell_state = self.lstm_cell(outputs[:, t - 1, :], (hidden_state, cell_state))
            out = self.fc_out(hidden_state)  # 8 -> 37 predict next state
            # out = self.softmax(out)  # 0 ~ 1 activate
            outputs[:, t, :] = out
        outputs = self.softmax(outputs)
        ##############################################################################
        return outputs


"""Problem 2: 

*   1.Connect CNN model to the desinged LSTM model.
*   2.Replace ResNet to your own CNN model from Assignment3.
"""

##############################################################################
"""ResNet"""
# CNN
# cnn_model_path = './trained_cnn_model_threshold_myresnet_testset.pth'
betternet = myResnet.betternet()
betternet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
betternet.fc = nn.Linear(in_features=512, out_features=ALL_CHAR_SET_LEN * MAX_CAPTCHA, bias=True)
# betternet.load_state_dict(torch.load(cnn_model_path))
betternet.to(device)
# betternet.eval()

##############################################################################

# LSTM
cnn_dim = 512  # resnet18-512
hidden_size = 8  # 8->37
vocab_size = 37  # ALL_CHAR_SET_LEN
lstm = LSTM(cnn_dim=cnn_dim, hidden_size=hidden_size, vocab_size=vocab_size)
lstm = lstm.to(device)


# loss, optimizer
##############################################################################
# optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.9)
params = list(betternet.parameters()) + list(lstm.parameters())
loss_func = nn.MultiLabelSoftMarginLoss()
nn.utils.clip_grad_norm_(params, 5)
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
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
print_interval = 30
max_epoch = 1000
# max_epoch = 1
threshold = 1e-4
isConverged = False
for epoch in range(max_epoch):
    if isConverged:
        break
    for step, i in enumerate(train_dl):
        img, label_oh, label = i
        img = Variable(img).cuda()
        label_oh = Variable(label_oh.float()).cuda()
        batch_size, _ = label_oh.shape
        pred_cnn, feature = betternet(img)
        ##############################################################################
        optimizer.zero_grad()

        label_oh_reshape = label_oh.reshape(batch_size, MAX_CAPTCHA, ALL_CHAR_SET_LEN)  # (128, 259) -> (128, 7, 37)
        # caption_oh = torch.argmax(label_oh_reshape, 2)  # one-hot word index
        # caption_target = caption_oh[:, 1:].to(device)  # 1~6
        # caption_target = label_oh[:, vocab_size:].to(device)  # (128, 259) -> (128, 222)

        caption_train = label_oh_reshape.to(device)
        # caption_train = label_oh_reshape[:, :label_oh_reshape.shape[1]-1, :].to(device)  # (128, 6, 37) ... 0~5

        pred_lstm = lstm(feature, caption_train, True)  # (128, 512, 5, 2), (128, 7, 37), float
        # outputs = lstm(feature, caption_train, True)  # (128, 512, 5, 2), (128, 7), float
        outputs = torch.flatten(pred_lstm, 1)  # 128, 222
        # for end-to-end learning
        loss = loss_func(outputs, label_oh)  # softmax(128, 259) vs. one-hot(128, 259)

        if loss < threshold:
            isConverged = True

        loss.backward()
        optimizer.step()
        ##############################################################################
        if (step + 1) % print_interval == 0:
            print('epoch:', epoch + 1, 'step:', step + 1, 'loss:', loss.item())
        if (epoch + 1) % 100 == 0:
            print("predict:", outputs[0, :])
            print("target:", label_oh[0, :])

cnn_model_path = './trained_cnn_model_end-to-end.pth'
lstm_model_path = './trained_lstm_model.pth'
print('Finished Training')
torch.save(betternet.state_dict(), cnn_model_path)
torch.save(lstm.state_dict(), lstm_model_path)
print('Saved Trained Model')

"""TEST"""

char_correct = 0
word_correct = 0
total = 0

betternet.eval()
lstm.eval()

with torch.no_grad():
    for step, (img, label_oh, label) in enumerate(test_dl):
        char_count = 0
        img = Variable(img).cuda()
        label_oh = Variable(label_oh.float()).cuda()
        pred_cnn, feature = betternet(img)
        batch_size, _ = label_oh.shape
        caption = pred_cnn.reshape(batch_size, MAX_CAPTCHA, ALL_CHAR_SET_LEN)  # 1, 7, 37 reshape
        # caption = torch.argmax(caption, 2)
        # use predicted words by CNN
        pred_lstm = lstm(feature, caption, False)  # (1, 7, 37)  #but captions will not be used
        pred_lstm = torch.flatten(pred_lstm, 1)  # (1, 259)

        label_len = label[0]
        outputs = pred_lstm.squeeze(0)
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

print(char_correct)
print(word_correct)
print((word_correct / total) * 100.0, '%')

"""END TEST"""
