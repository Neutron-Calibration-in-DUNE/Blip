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
        outputs:            list=[],
        output_dimensions:  list=[],
        when_to_compute:    str="all",
        meta:   dict={}
    ):
        """
        """
        super(SaverMetric, self).__init__(
            name, inputs, when_to_compute, meta
        )
        self.inputs = inputs
        self.outputs = outputs
        self.output_dimensions = output_dimensions
        self.batch_output = {}
        self.batch_input = {}
        self.num_classes = []
        self.labels = {}

        for ii, input in enumerate(self.inputs):
            # setup saver for input with number of classes
            self.batch_input[input] = torch.empty(
                size=(0, 1),
                dtype=torch.float, device=self.device
            )
        for ii, output in enumerate(self.outputs):
            # setup saver for output with number of classes
            self.batch_output[output] = torch.empty(
                size=(0, self.output_dimensions[ii]),
                dtype=torch.float, device=self.device
            )
        

    def reset_saver(self):
        for ii, input in enumerate(self.inputs):
            # setup saver for input with number of classes
            self.batch_input[input] = torch.empty(
                size=(0, 1),
                dtype=torch.float, device=self.device
            )
        for ii, output in enumerate(self.outputs):
            # setup saver for output with number of classes
            self.batch_output[output] = torch.empty(
                size=(0, self.output_dimensions[ii]),
                dtype=torch.float, device=self.device
            )
    
    def reset(self,
    ):
        pass

    def update(self,
        outputs,
        data,
    ):
        for ii, input in enumerate(self.inputs):
            # setup saver for input with number of classes
            self.batch_input[input] = torch.cat(
                (self.batch_input[input], data.category[self.meta['dataset'].meta['blip_classes_indices_by_name'][input]]), 
                dim=0
            )
        # get output probabilities for each class
        for ii, output in enumerate(self.outputs):
            self.batch_output[output] = torch.cat(
                (self.batch_output[output], outputs[output]), 
                dim=0
            )

    # def compute(self):
    #     outputs = {}
    #     for ii, input in enumerate(self.inputs):
    #         outputs[input] = self.metrics[input].compute()
    #     return outputs