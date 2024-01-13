"""
Common dictionarys, lists and functions.
"""
from torch import nn
import MinkowskiEngine as ME

activations = {
    'relu':         nn.ReLU,
    'tanh':         nn.Tanh,
    'sigmoid':      nn.Sigmoid,
    'softmax':      nn.Softmax,
    'leaky_relu':   nn.LeakyReLU,
}

sparse_activations = {
    'relu':     ME.MinkowskiReLU(),
    'prelu':    ME.MinkowskiPReLU(),
    'selu':     ME.MinkowskiSELU(),
    'celu':     ME.MinkowskiCELU(),
    'sigmoid':  ME.MinkowskiSigmoid(),
    'tanh':     ME.MinkowskiTanh(),
    'softmax':  ME.MinkowskiSoftmax(),
    'leaky_relu':   ME.MinkowskiLeakyReLU(),
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
