"""
Generic metrics for blip.
"""
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchmetrics.classification import AUROC

from blip.utils.logger import BlipError
from blip.utils.utils import fig_to_array
from blip.metrics.generic_metric import GenericMetric


class BlipSegmentationAUC(GenericMetric):
    """
    """
    def __init__(
        self,
        name:           str = 'blip_segmentation_auc_metric',
        meta:           dict = {}
    ):
        self.name = name
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        self.labels = ['vertex', 'tracklette_begin', 'tracklette_end', 'fragment_begin', 'fragment_end']
        self.label_indices = [2, 3, 4, 5, 6]

        # construct batch metric dictionaries
        self.auroc = {
            key: AUROC().to(self.device)
            for key in self.labels
        }

    def reset_batch(self):
        for key in self.auroc.keys():
            self.auroc[key] = AUROC().to(self.device)

    def set_device(
        self,
        device
    ):
        self.device = device
        for key in self.auroc.keys():
            self.auroc[key].to(self.device)

    def report_tensorboard(
        self,
        iterations,
        train_type
    ):
        for ii, output in enumerate(self.auroc.keys()):
            self.meta['tensorboard'].add_scalar(
                f'{self.name}: {output} ({train_type})',
                self.auroc[output].compute(),
                iterations
            )

    def update(
        self,
        data
    ):
        for ii, output in enumerate(self.auroc.keys()):
            self.auroc[output].update(
                nn.functional.softmax(data['outputs'][output].to(self.device), dim=1, dtype=torch.float),
                data['labels'].squeeze(0)[:, self.label_indices[ii]].long().to(self.device)
            )

    def compute(
        self,
    ):
        return {
            output: self.auroc[output].compute()
            for output in self.auroc.keys()
        }
