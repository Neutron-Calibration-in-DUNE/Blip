"""
Generic losses for blip.
"""
import torch

class GenericLoss:
    """
    """
    def __init__(self,
        name:   str='generic',
        alpha:  float=0.0,
        device: str='cpu'
    ):
        self.name = name
        self.alpha = alpha
        self.classes = []
        self.device = device

    def set_device(self,
        device
    ):  
        self.device = device
    
    def loss(self,
        outputs,
        data,
    ):
        pass