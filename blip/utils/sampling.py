"""
Implementation of the blip model using pytorch
"""
import numpy as np
import torch
import torch.nn                   as nn
import torch_geometric.transforms as T
import torch.nn.functional        as F
from torch.nn    import Linear
from collections import OrderedDict


def farthest_point_sampling(
    position, 
    number_of_points
):
    if number_of_points == None:
        return position
    
