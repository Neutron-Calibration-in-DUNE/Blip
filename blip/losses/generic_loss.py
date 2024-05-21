"""
Generic losses for blip.
"""
import torch

from blip.utils.logger import BlipError


class GenericLoss:
    """
    Abstract base class for Blip losses.  The inputs are
        1. name - a unique name for the loss function.
        2. alpha - a coefficient (should be from 0.0-1.0) for the strength of the influence of the loss.
        3. target_type - specifies whether the targets are features/classes/clusters/hits/etc.
        4. targets - list of names for the targets.
        5. outputs - list of names of the associated outputs for each target.
        6. augmentations - specified whether augmentations are created for the dataset
        7. meta - meta information from the module.
    """
    def __init__(
        self,
        name:           str = 'generic_loss',
        alpha:          float = 1.0,
        meta:           dict = {}
    ):
        self.name = name        
        self.alpha = alpha
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']

        # construct batch loss dictionaries
        self.batch_loss = {
            key: torch.empty(size=(0, 1), dtype=torch.float, device=self.device)
            for key in self.meta['dataset'].labels
        }

    def reset_batch(self):
        for key in self.batch_loss.keys():
            self.batch_loss[key] = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)

    def set_device(
        self,
        device
    ):
        self.device = device
        for key in self.batch_loss.keys():
            self.batch_loss[key] = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)

    def loss(
        self,
        data
    ):
        raise BlipError('"loss" not implemented in Loss!')
