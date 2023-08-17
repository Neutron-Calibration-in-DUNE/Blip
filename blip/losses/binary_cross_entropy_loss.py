"""
Wrapper for BinaryCrossEntropy loss
"""
import torch
import torch.nn as nn

from blip.losses import GenericLoss

class BinaryCrossEntropyLoss(GenericLoss):
    """
    """
    def __init__(self,
        name:           str='binary_cross_entropy_loss',
        alpha:          float=0.0,
        target_type:    str='classes',
        targets:        list=[],
        outputs:        list=[],
        augmentations:  int=0,
        reduction:      str='mean',
        sigmoid:        bool=True,
        meta:           dict={}
    ):
        super(BinaryCrossEntropyLoss, self).__init__(
            name, alpha, target_type, targets, outputs, augmentations, meta
        )
        self.reduction = reduction
        self.sigmoid = sigmoid
        if sigmoid:
            self.cross_entropy_loss = {
                key: nn.BCEWithLogitsLoss(reduction=self.reduction)
                for key in self.targets
            }
        else:
            self.cross_entropy_loss = {
                key: nn.BCELoss(reduction=self.reduction)
                for key in self.targets
            }

    def _loss(self,
        target,
        outputs,
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, output in enumerate(self.outputs):
            loss += self.cross_entropy_loss[self.targets[ii]](
                outputs[output].to(self.device), 
                target[self.targets[ii]].to(self.device)
            )
        return self.alpha * loss