"""
Simple model from Livio Calivers
"""
import numpy as np
import torch.nn as nn

from blip.models import generic_model

class CNNLivio(nn.Module):
    """
    """
    def __init__(self, dropout=0.2, kernel_size = 2,n_filters=32):
        super(CNNLivio, self).__init__()
        self.n_filters = n_filters
        self.padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding=self.padding)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=3, stride=1, padding=self.padding)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=3, stride=1, padding=self.padding)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 1), stride=(2, 2))

        self.fc1 = nn.Linear(in_features=n_filters*4 * 1 * 3, out_features=4096)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(in_features=4096, out_features=1024)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)

        self.fc3 = nn.Linear(in_features=1024, out_features=2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(batch_size,self.n_filters * 4 * 1 * 3)
        x = self.relu4(self.dropout1(self.fc1(x)))
        x = self.relu5(self.dropout2(self.fc2(x)))
        x = self.fc3(x)
        return x