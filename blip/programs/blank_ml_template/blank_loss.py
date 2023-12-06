
"""
Generic model code.
"""
import torch
from torch import nn

from blip.losses import GenericLoss

place_holder_loss_config = {
    "no_params":    "no_values"
}


class PlaceHolderLoss(GenericLoss):
    """
    """
    def __init__(
        self,
        name:           str = 'place_holder_loss',
        alpha:          float = 0.0,
        target_type:    str = 'classes',
        targets:        list = [],
        outputs:        list = [],
        augmentations:  int = 0,
        meta:           dict = {}
    ):
        super(PlaceHolderLoss, self).__init__(
            name, alpha, target_type, targets, outputs, augmentations, meta
        )
        self.place_holder_loss = {
            key: None
            for key in self.targets
        }

    def _loss(
        self,
        target,
        outputs
    ):
        loss = 0
        for ii, output in enumerate(self.outputs):
            temp_loss = self.alpha[ii] * self.place_holder_loss[self.targets[ii]](
                outputs[output].to(self.device),
                target[self.targets[ii]].unsqueeze(1).to(self.device)
            )
            loss += temp_loss
            self.batch_loss[self.targets[ii]] = torch.cat(
                (self.batch_loss[self.targets[ii]], torch.tensor([[temp_loss]], device=self.device)), dim=0
            )
        return loss
