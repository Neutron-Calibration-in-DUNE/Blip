"""
Wrapper for L2 loss
"""
import torch
import torch.nn as nn

from blip.losses import GenericLoss

class L2Loss(GenericLoss):
    """
    """
    def __init__(self,
        name:           str='multi_class_ce_loss',
        alpha:          float=0.0,
        target_type:    str='classes',
        targets:        list=[],
        outputs:        list=[],
        augmentations:  int=0,
        reduction:      str='mean',
        meta:           dict={}
    ):
        super(L2Loss, self).__init__(
            name, alpha, target_type, targets, outputs, augmentations, meta
        )
        self.alpha = alpha
        self.reduction = reduction
        self.l2_loss = {
            key: nn.MSELoss(reduction=reduction)
            for key in self.targets
        }

    def _loss(self,
        target,
        outputs
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, output in enumerate(self.outputs):
            loss += self.l2_loss[self.targets[ii]](
                outputs[output].to(self.device), 
                target[self.targets[ii]].to(self.device)
            )
        return self.alpha * loss