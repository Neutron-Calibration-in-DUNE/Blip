"""
Wrapper for CrossEntropy loss
"""
import numpy as np
import torch
import torch.nn as nn

from blip.losses import GenericLoss

class CrossEntropyLoss(GenericLoss):
    """
    """
    def __init__(self,
        name:           str='cross_entropy_loss',
        alpha:          float=0.0,
        target_type:    str='classes',
        targets:        list=[],
        outputs:        list=[],
        augmentations:  int=0,
        reduction:      str='mean',
        meta:           dict={}
    ):
        super(CrossEntropyLoss, self).__init__(
            name, alpha, target_type, targets, outputs, augmentations, meta
        )
        self.reduction = reduction
        # if len(class_weights.keys()) > 0:
        #     self.cross_entropy_loss = {
        #         key: nn.CrossEntropyLoss(
        #             weight=self.class_weights[key].to(self.device), 
        #             reduction=self.reduction
        #         )
        #         for key in self.class_weights.keys()
        #     }
        # else:
        self.cross_entropy_loss = {
            key: nn.CrossEntropyLoss(reduction=self.reduction)
            for key in self.targets
        }

    def _loss(self,
        target,
        outputs,
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, output in enumerate(self.outputs):
            temp_loss = self.cross_entropy_loss[self.targets[ii]](
                outputs[output].to(self.device), 
                target[self.targets[ii]].to(self.device)
            )
            loss += temp_loss
            self.batch_loss[self.targets[ii]] = torch.cat(
                (self.batch_loss[self.targets[ii]], torch.tensor([[temp_loss]], device=self.device)), dim=0
            )
        return self.alpha * loss