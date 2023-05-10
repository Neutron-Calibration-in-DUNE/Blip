"""
Common dictionarys, lists and functions.
"""
from torch import nn

activations = {
    'relu':         nn.ReLU,
    'tanh':         nn.Tanh,
    'sigmoid':      nn.Sigmoid,
    'softmax':      nn.Softmax,
    'leaky_relu':   nn.LeakyReLU,
}

normalizations = {
    'batch_norm',
    'bias',
}

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, input):
        return input