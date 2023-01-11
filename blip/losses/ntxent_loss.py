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
    ):
        super(NTXEntropyLoss, self).__init__(name)
        self.alpha = alpha
        self.temperature = temperature
        self.ntxent_loss = NTXentLoss(temperature=temperature)

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
        self.batch_loss = torch.cat((self.batch_loss, torch.tensor([[loss]], device=self.device)), dim=0)
        return self.alpha * loss