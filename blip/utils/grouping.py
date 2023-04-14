"""
Implementation of the blip model using pytorch
"""
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch_geometric.transforms as T
from torch.nn import Linear
import torch.nn.functional as F


def query_ball_point(
    centroids,
    position,
    radius,
    number_of_samples
):
    if radius == None:
        return torch.arange(len(centroids))