"""
Wrapper for CrossEntropy loss
"""
import torch
import torch.nn as nn

from blip.losses import GenericLoss

class MultiClassCrossEntropyLoss(GenericLoss):
    """
    """
    def __init__(self,
        alpha: float=1.0,
        name:   str='cross_entropy_loss',
        reduction:  str='mean'
    ):
        super(MultiClassCrossEntropyLoss, self).__init__(name)
        self.alpha = alpha
        self.reduction = reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=self.reduction)

    def loss(self,
        outputs,
        data,
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, classes in enumerate(outputs.keys()):
            loss += self.cross_entropy_loss(outputs[classes], data.category[:,ii].to(self.device))
        self.batch_loss = torch.cat((self.batch_loss, torch.tensor([[loss]], device=self.device)), dim=0)
        return self.alpha * loss