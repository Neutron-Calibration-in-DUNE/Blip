"""
Confusion matrix metric.
"""
import torch
import torch.nn as nn
from torchmetrics.classification import ConfusionMatrix
from torchmetrics.classification import MulticlassConfusionMatrix

from blip.dataset.common import *
from blip.metrics import GenericMetric

class SaverMetric(GenericMetric):
    
    def __init__(self,
        name:       str='saver_metric',
        inputs:             list=[],
        dimensions:         list=[],
        when_to_compute:    str="all",
        meta:   dict={}
    ):
        """
        """
        super(SaverMetric, self).__init__(
            name, inputs, when_to_compute, meta
        )
        self.inputs = inputs
        self.dimensions = dimensions
        self.batch_output = {}
        self.num_classes = []
        self.labels = {}

        for ii, input in enumerate(self.inputs):
            # setup saver for output with number of classes
            self.batch_output[input] = torch.empty(
                size=(0, self.dimensions[ii]),
                dtype=torch.float, device=self.device
            )

    def reset_saver(self):
        for ii, input in enumerate(self.inputs):
            # setup saver for output with number of classes
            self.batch_output[input] = torch.empty(
                size=(0, self.dimensions[ii]),
                dtype=torch.float, device=self.device
            )
    
    def reset(self,
    ):
        for ii, input in enumerate(self.inputs):
            self.metrics[input].reset()

    def update(self,
        outputs,
        data,
    ):
        # get output probabilities for each class
        for ii, input in enumerate(self.inputs):
            self.batch_output[input] = torch.cat(
                (self.batch_output[input], outputs[input]), 
                dim=0
            )

    # def compute(self):
    #     outputs = {}
    #     for ii, input in enumerate(self.inputs):
    #         outputs[input] = self.metrics[input].compute()
    #     return outputs