"""
Wrapper for NTXent loss
"""
import torch
from pytorch_metric_learning.losses import NTXentLoss

from blip.losses import GenericLoss


class NTXEntropyLoss(GenericLoss):
    """
    """
    def __init__(
        self,
        name:           str = 'ntxent_loss',
        alpha:          float = 0.0,
        target_type:    str = 'classes',
        targets:        list = [],
        outputs:        list = [],
        augmentations:  int = 0,
        reduction:      str = 'mean',
        temperature:    float = 0.10,
        meta:           dict = {}
    ):
        super(NTXEntropyLoss, self).__init__(
            name, alpha, target_type, targets, outputs, augmentations, meta
        )
        self.temperature = temperature
        self.ntx_entropy_loss = {}
        for key in self.targets:
            if key in self.meta["class_weights"]:
                self.ntx_entropy_loss[key] = NTXentLoss(
                    temperature=temperature, weight=self.meta['class_weights'][key]
                )
            else:
                self.ntx_entropy_loss[key] = NTXentLoss(temperature=temperature)

    def _loss(
        self,
        target,
        outputs,
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, output in enumerate(self.outputs):
            temp_loss = self.alpha[ii] * self.ntx_entropy_loss[self.targets[ii]](
                outputs[output].to(self.device),
                target[self.targets[ii]].to(self.device)
            )
            loss += temp_loss
            self.batch_loss[self.targets[ii]] = torch.cat(
                (self.batch_loss[self.targets[ii]], torch.tensor([[temp_loss]], device=self.device)), dim=0
            )
        return loss
