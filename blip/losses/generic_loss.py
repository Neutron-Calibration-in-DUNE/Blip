"""
Generic losses for blip.
"""
import torch

from blip.utils.logger import BlipError


class GenericLoss:
    """
    Abstract base class for Blip losses.
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
