"""
Confusion matrix metric.
"""
import torch
import torch.nn as nn
from torchmetrics.classification import AveragePrecision

from blip.dataset.common import *
from blip.metrics import GenericMetric

class AveragePrecisionMetric(GenericMetric):
    
    def __init__(self,
        name:           str='average_precision',
        target_type:        str='classes',
        when_to_compute:    str='all',
        targets:        list=[],
        outputs:        list=[],
        augmentations:  int=0,
        meta:           dict={}
    ):
        """
        """
        super(AveragePrecisionMetric, self).__init__(
            name, target_type, when_to_compute, targets, outputs, augmentations, meta
        )
        self.average_precision_tasks = {
            key: ('multiclass' if len(self.meta['dataset'].meta['blip_labels_values'][key]) > 2 else 'binary')
            for key in self.targets
        }
        self.average_precision_metric = {
            key: AveragePrecision(
                task=self.average_precision_tasks[key],
                num_classes=len(self.meta['dataset'].meta['blip_labels_values'][key])
            ).to(self.device)
            for key in self.targets
        }
        
    def _metric_update(self,
        target,
        outputs
    ):
        for ii, output in enumerate(self.outputs):
            self.average_precision_metric[output].update(
                nn.functional.softmax(outputs[output].to(self.device), dim=1, dtype=torch.float),
                target[self.targets[ii]].to(self.device)
            )

    def _metric_compute(self):
        return {
            output: self.average_precision_metric[output].compute()
            for output in self.outputs
        }