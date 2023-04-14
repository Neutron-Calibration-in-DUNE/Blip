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


def farthest_point_sampling(
    position, 
    number_of_points
):
    if number_of_points == None:
        return position
    
