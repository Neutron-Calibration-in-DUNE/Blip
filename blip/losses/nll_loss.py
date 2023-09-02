"""
Wrapper for NLL loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from blip.losses import GenericLoss

class NegativeLogLikelihoodLoss(GenericLoss):
    """
    """
    def __init__(self,
        name:           str='nll_loss',
        alpha:          float=0.0,
        target_type:    str='classes',
        targets:        list=[],
        outputs:        list=[],
        augmentations:  int=0,
        reduction:      str='mean',
        meta:           dict={}
    ):
        super(NegativeLogLikelihoodLoss, self).__init__(
            name, alpha, target_type, targets, outputs, augmentations, meta
        )
        self.reduction = reduction
        self.nll_loss = {
            key: nn.NLLLoss(reduction=self.reduction)
            for key in self.targets
        }

    def _loss(self,
        target,
        outputs,
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, output in enumerate(self.outputs):
            temp_loss = self.alpha[ii] * self.nll_loss[self.targets[ii]](
                outputs[output].to(self.device), 
                target[self.targets[ii]].to(self.device)
            )
            loss += temp_loss
            self.batch_loss[self.targets[ii]] = torch.cat(
                (self.batch_loss[self.targets[ii]], torch.tensor([[temp_loss]], device=self.device)), dim=0
            )
        return loss