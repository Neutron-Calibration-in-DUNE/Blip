"""
Generic losses for blip.
"""
import torch

class GenericLoss:
    """
    """
    def __init__(self,
        name:   str='generic',
        device: str='cpu'
    ):
        self.name = name
        self.alpha = 0.0
        self.classes = []
        # set device to cpu for now
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