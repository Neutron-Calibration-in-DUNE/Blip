"""
Wrapper for BlipSegmentation loss
"""
import numpy as np
import torch
import torch.nn as nn

from blip.losses.generic_loss import GenericLoss


class BlipSegmentationLoss(GenericLoss):
    """
    """
    def __init__(
        self,
        name:           str = 'cross_entropy_loss',
        alpha:          float = 1.0,
        meta:           dict = {}
    ):
        super(BlipSegmentationLoss, self).__init__(
            name, alpha, meta
        )
        self.reduction = 'mean'
        self.cross_entropy_loss = {
            'topology': nn.CrossEntropyLoss(reduction=self.reduction),
            'physics': nn.CrossEntropyLoss(reduction=self.reduction)
        }

    def loss(
        self,
        data
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, output in enumerate(self.cross_entropy_loss.keys()):
            temp_loss = self.alpha * self.cross_entropy_loss[output](
                data['outputs'][output].to(self.device),
                data['labels'].squeeze(0)[:, ii].long().to(self.device)
            )
            loss += temp_loss
            self.batch_loss[output] = torch.cat(
                (self.batch_loss[output], torch.tensor([[temp_loss]], device=self.device)), dim=0
            )
        return loss
