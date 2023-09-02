"""
Wrapper for Focal loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from blip.losses import GenericLoss

class FocalLoss(GenericLoss):
    """
    """
    def __init__(self,
        name:           str='focal_loss',
        alpha:          float=0.0,
        target_type:    str='classes',
        targets:        list=[],
        outputs:        list=[],
        augmentations:  int=0,
        reduction:      str='mean',
        gamma:          float=2,
        meta:           dict={}
    ):
        super(FocalLoss, self).__init__(
            name, alpha, target_type, targets, outputs, augmentations, meta
        )
        self.reduction = reduction
        self.gamma = gamma
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        if self.reduction == 'mean':
            self.focal_loss = {
                key: self.mean_focal
                for key in self.targets
            }
        elif self.reduction == 'sum':
            self.focal_loss = {
                key: self.sum_focal
                for key in self.targets
            }
        else:
            self.focal_loss = {
                key: self.focal
                for key in self.targets
            }
        
    def focal(self,
        output,
        target
    ):
        cross_entropy_loss = self.cross_entropy_loss(output, target)
        pt = torch.exp(-cross_entropy_loss)
        focal_loss = (1 - pt) ** self.gamma * cross_entropy_loss
        return focal_loss   

    def mean_focal(self,
        output,
        target
    ):
        cross_entropy_loss = self.cross_entropy_loss(output, target)
        pt = torch.exp(-cross_entropy_loss)
        focal_loss = (1 - pt) ** self.gamma * cross_entropy_loss
        return focal_loss.mean()

    def sum_focal(self,
        output,
        target
    ):
        cross_entropy_loss = self.cross_entropy_loss(output, target)
        pt = torch.exp(-cross_entropy_loss)
        focal_loss = (1 - pt) ** self.gamma * cross_entropy_loss
        return focal_loss.sum()

    def _loss(self,
        target,
        outputs,
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, output in enumerate(self.outputs):
            temp_loss = self.alpha[ii] * self.focal_loss[self.targets[ii]](
                outputs[output].to(self.device), 
                target[self.targets[ii]].to(self.device)
            )
            loss += temp_loss
            self.batch_loss[self.targets[ii]] = torch.cat(
                (self.batch_loss[self.targets[ii]], torch.tensor([[temp_loss]], device=self.device)), dim=0
            )
        return loss