"""
Wrapper for NTXent loss
"""
import torch
import torch.nn as nn
from pytorch_metric_learning.losses import NTXentLoss

from blip.losses import GenericLoss

class NTXEntropyLoss(GenericLoss):
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
        temperature:    float=0.10,
        meta:           dict={}
    ):
        super(NTXEntropyLoss, self).__init__(
            name, alpha, target_type, targets, outputs, augmentations, meta
        )
        self.temperature = temperature
        self.ntx_entropy_loss = {
            key: NTXentLoss(temperature=temperature)
            for key in self.targets
        }

    def _loss(self,
        target,
        outputs,
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        print(target)
        print(outputs)
        for ii, output in enumerate(self.outputs):
            print(ii, output)
            loss += self.ntx_entropy_loss[self.targets[ii]](
                outputs[output].to(self.device), 
                target[self.targets[ii]].to(self.device)
            )
        return self.alpha * loss