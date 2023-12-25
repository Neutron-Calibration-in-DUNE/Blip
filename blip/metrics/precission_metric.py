"""
Confusion matrix metric.
"""
import torch
import torch.nn as nn
from torchmetrics.classification import Precision

from blip.metrics import GenericMetric


class PrecisionMetric(GenericMetric):

    def __init__(
        self,
        name:           str = 'precision',
        target_type:        str = 'classes',
        when_to_compute:    str = 'all',
        targets:        list = [],
        outputs:        list = [],
        augmentations:  int = 0,
        meta:           dict = {}
    ):
        """
        """
        super(PrecisionMetric, self).__init__(
            name, target_type, when_to_compute, targets, outputs, augmentations, meta
        )
        self.precision_tasks = {
            key: ('multiclass' if len(self.meta['dataset'].meta['blip_labels_values'][key]) > 2 else 'binary')
            for key in self.targets
        }
        self.precision_metric = {
            key: Precision(
                task=self.precision_tasks[key],
                num_classes=len(self.meta['dataset'].meta['blip_labels_values'][key])
            ).to(self.device)
            for key in self.targets
        }

    def _metric_update(
        self,
        target,
        outputs
    ):
        for ii, output in enumerate(self.outputs):
            self.precision_metric[output].update(
                nn.functional.softmax(outputs[output].to(self.device), dim=1, dtype=torch.float),
                target[self.targets[ii]].to(self.device)
            )

    def _metric_compute(self):
        return {
            output: self.precision_metric[output].compute()
            for output in self.outputs
        }
