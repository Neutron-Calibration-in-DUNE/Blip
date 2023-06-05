"""
Wrapper for NTXent loss
"""
import torch
import torch.nn as nn
from pytorch_metric_learning.losses import NTXentLoss

from blip.losses import GenericLoss

class MultiClassNTXEntropyLoss(GenericLoss):
    """
    """
    def __init__(self,
        alpha: float=1.0,
        name:   str='ntxent_loss',
        classes:    list=[], 
        temperature:  float=0.10,
        device:     str='cpu'
    ):
        super(MultiClassNTXEntropyLoss, self).__init__(name, device)
        self.alpha = alpha
        self.temperature = temperature
        self.classes = classes
        self.ntx_entropy_loss = {
            NTXentLoss(temperature=temperature)
            for key in self.classes
        }

    def loss(self,
        outputs,
        data,
    ):
        """Computes and returns/saves loss information"""
        embeddings = outputs['reductions']
        indices = torch.arange(0, len(data.category), device=outputs['reductions'].device)
        labels = torch.cat([
            indices
            for ii in range(int(len(outputs['reductions'])/len(data.category)))
        ])
        loss = self.ntxent_loss(embeddings, labels)
        return self.alpha * loss