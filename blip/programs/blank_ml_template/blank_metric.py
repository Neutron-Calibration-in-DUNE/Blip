
"""
Generic model code.
"""
import torch
from torch import nn

from blip.metrics import GenericMetric

place_holder_metric_config = {
    "no_params":    "no_values"
}


class PlaceHolderMetric(GenericMetric):
    """
    """
    def __init__(
        self,
        name:           str = 'generic',
        target_type:        str = 'classes',
        when_to_compute:    str = 'all',
        targets:        list = [],
        outputs:        list = [],
        augmentations:  int = 0,
        meta:           dict = {}
    ):
        super(PlaceHolderMetric, self).__init__(
            name, target_type, when_to_compute, targets, outputs, augmentations, meta
        )
        self.place_holder_metric = {
            key: None
            for key in self.targets
        }
        
    def _metric_update(
        self,
        target,
        outputs
    ):
        for ii, output in enumerate(self.outputs):
            self.place_holder_metric[output].update(
                outputs[output].to(self.device),
                target[self.targets[ii]].to(self.device)
            )

    def _metric_compute(self):
        return {
            output: self.place_holder_metric[output].compute()
            for output in self.outputs
        }
