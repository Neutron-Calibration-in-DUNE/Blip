"""
Wrapper for L1 loss
"""
import torch
import torch.nn as nn

from blip.losses import GenericLoss

class L1Loss(GenericLoss):
    """
    """
    def __init__(self,
        alpha: float=1.0,
        name:   str='l1_loss',
        reduction:  str='mean',
        device:     str='cpu'
    ):
        super(L1Loss, self).__init__(name, device)
        self.alpha = alpha
        self.reduction = reduction
        self.l1_loss = nn.L1Loss(reduction=reduction)

    def loss(self,
        outputs,
        data,
    ):
        """Computes and returns/saves loss information"""
        loss = self.l1_loss(outputs, data[1].to(self.device))
        self.batch_loss = torch.cat((self.batch_loss, torch.tensor([[loss]], device=self.device)), dim=0)
        return self.alpha * loss