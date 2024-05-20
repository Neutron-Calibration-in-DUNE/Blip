"""
Generic metrics for blip.
"""
import torch

from blip.utils.logger import BlipError


class GenericMetric:
    """
    Abstract base class for Blip metrics.  The inputs are
        1. name - a unique name for the metric function.
        2. meta - meta information from the module.
    """
    def __init__(
        self,
        name:           str = 'generic_metric',
        meta:           dict = {}
    ):
        self.name = name
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']

        # construct batch metric dictionaries
        self.batch_metric = {
            key: torch.empty(size=(0, 1), dtype=torch.float, device=self.device)
            for key in ['topology', 'physics']
        }

    def reset_batch(self):
        for key in self.batch_metric.keys():
            self.batch_metric[key] = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)

    def set_device(
        self,
        device
    ):
        self.device = device
        for key in self.batch_metric.keys():
            self.batch_metric[key] = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)

    def report_tensorboard(
        self,
        iterations,
        train_type
    ):
        pass

    def update(
        self,
        data
    ):
        raise BlipError(f'"update" not implemented in {self.name}!')

    def compute(
        self,
    ):
        raise BlipError(f'"compute" not implemented in {self.name}!')
