"""
Wrapper for NTXent loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from blip.losses import GenericLoss

class NegativeLogLikelihoodLoss(GenericLoss):
    """
    """
    def __init__(self,
        alpha: float=1.0,
        name:   str='nll_loss',
        meta:   dict={}
    ):
        super(NegativeLogLikelihoodLoss, self).__init__(name, meta)
        self.alpha = alpha
        self.nll_loss = F.nll_loss

    def loss(self,
        outputs,
        data,
    ):
        """Computes and returns/saves loss information"""
        loss = self.nll_loss(outputs, data.category)
        return self.alpha * loss