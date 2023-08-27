"""
Wrapper for NTXent loss
"""
import torch
import torch.nn as nn
from pytorch_metric_learning.losses import AngularLoss as angular_loss

from blip.losses import GenericLoss

class AngularLoss(GenericLoss):
    """
    """
    def __init__(self,
        name:           str='ntxent_loss',
        alpha:          float=0.0,
        target_type:    str='classes',
        targets:        list=[],
        outputs:        list=[],
        augmentations:  int=0,
        reduction:      str='mean',
        beta:           float=40,
        meta:           dict={}
    ):
        super(AngularLoss, self).__init__(
            name, alpha, target_type, targets, outputs, augmentations, meta
        )
        self.beta = beta
        self.ntx_entropy_loss = {
            key: angular_loss(alpha=self.beta)
            for key in self.targets
        }

    def _loss(self,
        target,
        outputs,
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, output in enumerate(self.outputs):
            temp_loss = self.ntx_entropy_loss[self.targets[ii]](
                outputs[output].to(self.device), 
                target[self.targets[ii]].to(self.device)
            )
            loss += temp_loss
            self.batch_loss[self.targets[ii]] = torch.cat(
                (self.batch_loss[self.targets[ii]], torch.tensor([[temp_loss]], device=self.device)), dim=0
            )
        return self.alpha * loss