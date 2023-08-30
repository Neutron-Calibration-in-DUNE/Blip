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
        name:           str='l1_loss',
        alpha:          float=0.0,
        target_type:    str='classes',
        targets:        list=[],
        outputs:        list=[],
        augmentations:  int=0,
        reduction:      str='mean',
        meta:           dict={}
    ):
        super(L1Loss, self).__init__(
            name, alpha, target_type, targets, outputs, augmentations, meta
        )
        self.alpha = alpha
        self.reduction = reduction
        self.l1_loss = {
            key: nn.L1Loss(reduction=reduction)
            for key in self.targets
        }

    def _loss(self,
        target,
        outputs
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, output in enumerate(self.outputs):
            temp_loss = self.l1_loss[self.targets[ii]](
                outputs[output].to(self.device), 
                target[self.targets[ii]].unsqueeze(1).to(self.device)
            )
            loss += temp_loss
            self.batch_loss[self.targets[ii]] = torch.cat(
                (self.batch_loss[self.targets[ii]], torch.tensor([[temp_loss]], device=self.device)), dim=0
            )
        return self.alpha * loss