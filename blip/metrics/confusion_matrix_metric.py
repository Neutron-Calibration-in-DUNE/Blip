"""
Confusion matrix metric.
"""
import torch
import torch.nn as nn
from torchmetrics.classification import ConfusionMatrix
from torchmetrics.classification import MulticlassConfusionMatrix

from blip.dataset.common import *
from blip.metrics import GenericMetric

class ConfusionMatrixMetric(GenericMetric):
    
    def __init__(self,
        name:       str='confusion_matrix',
        shape:      tuple=(),
        inputs:             list=[''],
        number_of_classes:  list=[],
    ):
        """
        """
        super(ConfusionMatrixMetric, self).__init__(
            name,
            shape,
            inputs
        )
        self.number_of_classes = number_of_classes
        self.inputs = inputs
        self.number_of_classes = number_of_classes

        self.metrics = {}
        self.batch_probabilities = {}
        self.batch_summed_adc = {}
        self.labels = {}

        for ii, input in enumerate(self.inputs):
            if self.number_of_classes[ii] == 2:
                self.metrics[input] = ConfusionMatrix(task="binary", number_of_classes=2)
            else:
                self.metrics[input] = MulticlassConfusionMatrix(
                    num_classes=self.number_of_classes[ii]
                )
            self.labels[input] = classification_labels[input].values()
            self.batch_probabilities[input] = torch.empty(
                size=(0, self.number_of_classes[ii] + 1),
                dtype=torch.float, device=self.device
            )
            self.batch_summed_adc[input] = torch.empty(
                size=(0, 1),
                dtype=torch.float, device=self.device
            )

    def reset_probabilities(self):
        for ii, input in enumerate(self.inputs):
            self.batch_probabilities[input] = torch.empty(
                size=(0, self.number_of_classes[ii] + 1),
                dtype=torch.float, device=self.device
            )
            self.batch_summed_adc[input] = torch.empty(
                size=(0, 1),
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
                (softmax, data.category[:,ii].unsqueeze(1)),
                dim=1
            ).to(self.device)

            self.batch_probabilities[input] = torch.cat(
                (self.batch_probabilities[input], predictions),
                dim=0
            )
            
            # get summed adc from inputs
            self.batch_summed_adc[input] = torch.cat(
                (self.batch_summed_adc[input], data.summed_adc.unsqueeze(1).to(self.device)),
                dim=0
            )
            self.metrics[input].update(outputs[input], data.category[:,ii])

    def compute(self):
        outputs = {}
        for ii, input in enumerate(self.inputs):
            outputs[input] = self.metrics[input].compute()
        return outputs