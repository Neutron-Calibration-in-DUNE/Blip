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
    ):
        super(NegativeLogLikelihoodLoss, self).__init__(name)
        self.alpha = alpha
        self.nll_loss = F.nll_loss

    def loss(self,
        outputs,
        data,
    ):
        """Computes and returns/saves loss information"""
        loss = self.nll_loss(outputs, data.category)
        self.batch_loss = torch.cat((self.batch_loss, torch.tensor([[loss]], device=self.device)), dim=0)
        return self.alpha * loss