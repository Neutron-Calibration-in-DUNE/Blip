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
        meta:   dict={}
    ):
        self.name = name
        self.alpha = alpha
        self.classes = []
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']

    def set_device(self,
        device
    ):  
        self.device = device
    
    def loss(self,
        outputs,
        data,
    ):
        pass