'''
model_archive.py

A file that contains neural network models.
You can also implement your own model here.
'''
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self, hparams):
        super(Baseline, self).__init__()

        self.conv0 = nn.Sequential(
                nn.Conv1d(hparams.num_mels, 32, kernel_size=8, stride=1, padding=0),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(8, stride=8)
                )

        self.conv1 = nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=0),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(8, stride=8)
                )

        self.conv2 = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(4, stride=4)
                )

        self.linear = nn.Linear(192, len(hparams.genres))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), x.size(1)*x.size(2))
        x = self.linear(x)

        return x




class Conv_1D(nn.Module):
    def __init__(self, hparams):
        super(Conv_1D, self).__init__()

        self.conv0 = nn.Sequential(
                nn.Conv1d(hparams.num_mels, 32, kernel_size=3, stride=1, padding=1),
                # nn.Conv1d(hparams.num_mels, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2)
                )

        self.conv1 = nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2)
                )

        self.conv2 = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2)
                )

        self.fc0 = nn.Linear(192, 128)
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, len(hparams.genres))
        self.linear = nn.Linear(1024, len(hparams.genres))
        self.dropout_conv = nn.Dropout(p=0.2)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), x.size(1)*x.size(2))
        '''
        x = self.fc0(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        '''
        x = self.linear(x)
        x = self.softmax(x)

        return x

class Conv_2D(nn.Module):
    def __init__(self, hparams):
        super(Conv_2D, self).__init__()

        self.conv_128_64 = nn.Sequential(
                nn.Conv2d(hparams.num_mels, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                )

        self.conv_64_128 = nn.Sequential(
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                )


        self.maxpool = nn.MaxPool2d(4, 4)

        self.linear = nn.Linear(128, len(hparams.genres))

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        # [4, num_mels, feature_length]
        x = x.view(x.size(0), x.size(1), x.size(2)//8, 8)

        x = self.conv_128_64(x)
        x = self.maxpool(x)
        x = self.conv_64_128(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))
        x = self.linear(x)
        x = self.softmax(x)

        return x

class Choi(nn.Module):
    def __init__(self, hparams):
        super(Choi, self).__init__()

        self.conv1= nn.Sequential(
                nn.Conv2d(hparams.num_mels, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2,4),
                )
        self.conv2= nn.Sequential(
                nn.Conv2d(128, 384, 3, 1, 1),
                nn.BatchNorm2d(384),
                nn.ReLU(),
                nn.MaxPool2d(4,5)
                )

        self.conv3= nn.Sequential(
                nn.Conv2d(384, 768, 3, 1, 1),
                nn.BatchNorm2d(768),
                nn.ReLU(),
                nn.MaxPool2d(3,8),
                )

        self.conv4 = nn.Sequential(
                nn.Conv2d(768, 2048, 3, 1, 1),
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.MaxPool2d(4,8),
                )

        self.maxpool = nn.MaxPool2d(4, 4)

        self.linear = nn.Linear(64, len(hparams.genres))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), x.size(2)//8, 8)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))
        # x = self.linear(x)
        # x = self.softmax(x)
        return x

