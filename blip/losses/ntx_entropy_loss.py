"""
Wrapper for NTXent loss
"""
import torch
import torch.nn as nn
from pytorch_metric_learning.losses import NTXentLoss

from blip.losses import GenericLoss

class NTXEntropyLoss(GenericLoss):
    """
    """
    def __init__(self,
        alpha: float=1.0,
        name:   str='ntxent_loss',
        temperature:  float=0.10,
        classes:    list=[],
        reduction:  str='mean',
        class_weights:  dict={},
        device:     str='cpu'
    ):
        super(NTXEntropyLoss, self).__init__(name, device)
        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction
        self.classes = classes
        self.class_weights = class_weights
        self.ntxent_loss = NTXentLoss(temperature=temperature)

    def loss(self,
        outputs,
        data,
    ):
        """Computes and returns/saves loss information"""
        embeddings = outputs['reductions']
        indices = torch.arange(0, len(torch.unique(data.batch)), device=outputs['reductions'].device)
        labels = torch.cat([
            indices
            for ii in range(int(len(outputs['reductions'])/len(torch.unique(data.batch))))
        ])
        loss = self.ntxent_loss(embeddings, labels)
        return self.alpha * loss