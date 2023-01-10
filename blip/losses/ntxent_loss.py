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
        print(len(outputs))
        print(outputs[0].shape)
        print(outputs[1].shape)
        embeddings = torch.cat(outputs[1])
        print(embeddings)
        indices = torch.arange(0, outputs[1][0].size(0), device=outputs[1][0].device)
        print(indices)
        labels = torch.cat([indices for ii in range(len(outputs[1]))])
        print(labels)
        loss = self.ntxent_loss(embeddings, labels)
        self.batch_loss = torch.cat((self.batch_loss, torch.tensor([[loss]], device=self.device)), dim=0)
        return self.alpha * loss