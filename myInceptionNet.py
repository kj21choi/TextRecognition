import torch
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv -> 3x3 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


# Define a CNN model
class BetterNet(nn.Module):
    def __init__(self):
        super(BetterNet, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.stem_network = nn.Sequential(
            nn.Conv2d(1, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        # in-channel, n1x1(b1), n1x1(b2). n3x3(b2), n3x3(b3-1), n3x3(b3-2), n1x1(b4)
        # next-in-channel = n1x1(b1) + n3x3(b2) + n3x3(b3-2), n1x1(b4)
        self.inc_1 = Inception(192,  64,  96, 128, 16,  32,  32)  
        self.inc_2 = Inception(256, 128,  96, 128, 16,  32,  32)
        self.inc_3 = Inception(320, 128,  96, 128, 32,  64,  64)

        self.inc_4 = Inception(384, 192,  96, 192, 32,  64,  64)
        self.inc_5 = Inception(512, 160, 112, 224, 32,  64,  64)
        self.inc_6 = Inception(512, 256, 128, 384, 48,  96,  96)

        self.inc_7 = Inception(832, 272, 128, 368, 48,  96,  96)
        self.inc_8 = Inception(832, 256, 192, 384, 48,  96,  96)
        self.inc_9 = Inception(832, 368, 200, 400, 64, 128, 128)

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(1024, 10)
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, x):
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        out = self.stem_network(x)
        out = self.inc_1(out)
        out = self.inc_2(out)
        out = self.inc_3(out)
        out = self.maxpool3(out)

        out = self.inc_4(out)
        out = self.inc_5(out)
        out = self.inc_6(out)
        out = self.maxpool(out)

        out = self.inc_7(out)
        out = self.inc_8(out)
        out = self.inc_9(out)
        out = self.avgpool(out)

        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.softmax(out)
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return out