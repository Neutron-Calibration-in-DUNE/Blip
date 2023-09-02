"""
Wrapper for Contrastive loss
"""
import torch
import torch.nn as nn
from pytorch_metric_learning.losses import ContrastiveLoss as contrastive_loss

from blip.losses import GenericLoss

class ContrastiveLoss(GenericLoss):
    """
    """
    def __init__(self,
        name:           str='contrastive_loss',
        alpha:          float=0.0,
        target_type:    str='classes',
        targets:        list=[],
        outputs:        list=[],
        augmentations:  int=0,
        reduction:      str='mean',
        pos_margin:     float=0,
        neg_margin:     float=1,
        meta:           dict={}
    ):
        super(ContrastiveLoss, self).__init__(
            name, alpha, target_type, targets, outputs, augmentations, meta
        )
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.contrastive_loss = {
            key: contrastive_loss(pos_margin=self.pos_margin, neg_margin=self.neg_margin)
            for key in self.targets
        }

    def _loss(self,
        target,
        outputs,
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, output in enumerate(self.outputs):
            temp_loss = self.alpha[ii] * self.contrastive_loss[self.targets[ii]](
                outputs[output].to(self.device), 
                target[self.targets[ii]].to(self.device)
            )
            loss += temp_loss
            self.batch_loss[self.targets[ii]] = torch.cat(
                (self.batch_loss[self.targets[ii]], torch.tensor([[temp_loss]], device=self.device)), dim=0
            )
        return loss