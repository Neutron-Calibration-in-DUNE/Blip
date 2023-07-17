"""
Confusion matrix metric.
"""
import torch
import torch.nn as nn
from torchmetrics.classification import Dice

from blip.dataset.common import *
from blip.metrics import GenericMetric

class DiceScoreMetric(GenericMetric):
    
    def __init__(self,
        name:       str='dice_score',
        shape:      tuple=(),
        inputs:             list=[],
        number_of_classes:  list=[],
        when_to_compute:    str="all",
        consolidate_classes:   dict=None,
        meta:   dict={}
    ):
        """
        """
        super(DiceScoreMetric, self).__init__(
            name, shape, input, when_to_compute, meta
        )
        self.number_of_classes = number_of_classes
        self.inputs = inputs

        self.metrics = {}
        self.batch_probabilities = {}
        self.batch_summed_adc = {}
        if consolidate_classes is not None:
            self.consolidate_classes = True
        else:
            self.consolidate_classes = False
        self.labels = {}

        for ii, input in enumerate(self.inputs):
            self.metrics[input] = Dice(
                num_classes=self.number_of_classes[ii]
            )
            if consolidate_classes is not None:
                self.labels[input] = consolidate_classes[input]
            else:
                self.labels[input] = classification_labels[input].values()
            self.batch_probabilities[input] = torch.empty(
                size=(0, self.number_of_classes[ii] + 1),
                dtype=torch.float, device=self.device
            )

    def reset_probabilities(self):
        for ii, input in enumerate(self.inputs):
            self.batch_probabilities[input] = torch.empty(
                size=(0, self.number_of_classes[ii] + 1),
                dtype=torch.float, device=self.device
            )
    
    def reset(self,
    ):
        for ii, input in enumerate(self.inputs):
            self.metrics[input].reset()
        self.reset_probabilities()

    def set_device(self,
        device
    ):
        for ii, input in enumerate(self.inputs):
            self.metrics[input].to(device)
        self.device = device

    def update(self,
        outputs,
        data,
    ):
        # get output probabilities for each class
        for ii, input in enumerate(self.inputs):
            softmax = nn.functional.softmax(
                outputs[input], 
                dim=1, dtype=torch.float
            )
            predictions = torch.cat(
                (softmax, data.category[:, ii].unsqueeze(1).to(self.device)),
                dim=1
            ).to(self.device)
            self.batch_probabilities[input] = torch.cat(
                (self.batch_probabilities[input], predictions),
                dim=0
            )
            
            self.metrics[input].update(
                outputs[input], data.category[:,ii].to(self.device)
            )

    def compute(self):
        outputs = {}
        for ii, input in enumerate(self.inputs):
            outputs[input] = self.metrics[input].compute()
        return outputs