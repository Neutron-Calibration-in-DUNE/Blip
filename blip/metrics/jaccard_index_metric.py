"""
Confusion matrix metric.
"""
import torch
import torch.nn as nn
from torchmetrics.classification import JaccardIndex

from blip.metrics import GenericMetric


class JaccardIndexMetric(GenericMetric):

    def __init__(
        self,
        name:           str = 'jaccard_index',
        target_type:        str = 'classes',
        when_to_compute:    str = 'all',
        targets:        list = [],
        outputs:        list = [],
        augmentations:  int = 0,
        meta:           dict = {}
    ):
        """
        """
        super(JaccardIndexMetric, self).__init__(
            name, target_type, when_to_compute, targets, outputs, augmentations, meta
        )
        self.jaccard_index_tasks = {
            key: ('multiclass' if len(self.meta['dataset'].meta['blip_labels_values'][key]) > 2 else 'binary')
            for key in self.targets
        }
        self.jaccard_index_metric = {
            key: JaccardIndex(
                task=self.jaccard_index_tasks[key],
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
            self.jaccard_index_metric[output].update(
                nn.functional.softmax(outputs[output].to(self.device), dim=1, dtype=torch.float),
                target[self.targets[ii]].to(self.device)
            )

    def _metric_compute(self):
        return {
            output: self.jaccard_index_metric[output].compute()
            for output in self.outputs
        }
