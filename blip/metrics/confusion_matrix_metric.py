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
        mode:       str="view",
        inputs:             list=[],
        when_to_compute:    str="all",
        meta:   dict={}
    ):
        """
        """
        super(ConfusionMatrixMetric, self).__init__(
            name, inputs, when_to_compute, meta
        )
        self.mode = mode
        self.inputs = inputs

        self.metrics = {}
        self.batch_probabilities = {}
        self.batch_summed_adc = {}
        self.num_classes = []
        self.consolidate_classes = False
        self.labels = {}

        for ii, input in enumerate(self.inputs):
            # setup confusion matrix with number of classes
            self.num_classes.append(len(self.meta['dataset'].meta['blip_labels_values'][input]))
            self.metrics[input] = MulticlassConfusionMatrix(
                num_classes=len(self.meta['dataset'].meta['blip_labels_values'][input])
            )
            # set label names
            if self.meta['dataset'].meta['consolidate_classes'] is not None:
                self.consolidate_classes = True
                self.labels[input] = self.meta['dataset'].meta['consolidate_classes'][input]
            else:
                self.labels[input] = self.meta['dataset'].meta['blip_labels_values'][input]

            if self.mode == "view":
                self.batch_probabilities[input] = torch.empty(
                    size=(0, self.num_classes[ii] + 1),
                    dtype=torch.float, device=self.device
                )
                
            elif self.mode == "view_cluster":
                self.batch_probabilities[input] = torch.empty(
                    size=(0, self.num_classes[ii] * 2),
                    dtype=torch.float, device=self.device
                )
        self.batch_summed_adc = torch.empty(
            size=(0, 1),
            dtype=torch.float, device=self.device
        )

    def reset_probabilities(self):
        for ii, input in enumerate(self.inputs):
            if self.mode == "view":
                self.batch_probabilities[input] = torch.empty(
                    size=(0, self.num_classes[ii] + 1),
                    dtype=torch.float, device=self.device
                )
            elif self.mode == "view_cluster":
                self.batch_probabilities[input] = torch.empty(
                    size=(0, self.num_classes[ii] * 2),
                    dtype=torch.float, device=self.device
                )
        self.batch_summed_adc = torch.empty(
            size=(0, 1),
            dtype=torch.float, device=self.device
        )
    
    def reset(self,
    ):
        for ii, input in enumerate(self.inputs):
            self.metrics[input].reset()
        # self.reset_probabilities()

    # def set_device(self,
    #     device
    # ):
    #     for ii, input in enumerate(self.inputs):
    #         self.metrics[input].to(device)
    #     self.device = device

    def update(self,
        outputs,
        data,
    ):
        # get output probabilities for each class
        batch = data.batch.to(self.device)
        data.to(self.device)
        for ii, input in enumerate(self.inputs):
            softmax = nn.functional.softmax(
                outputs[input], 
                dim=1, dtype=torch.float
            )
            if self.mode == "view":
                predictions = torch.cat(
                    (softmax, data.category.unsqueeze(1)),
                    dim=1
                ).to(self.device)

                self.batch_probabilities[input] = torch.cat(
                    (self.batch_probabilities[input], predictions),
                    dim=0
                )
                self.metrics[input].update(
                    softmax, data.category.to(self.device)
                )
                self.batch_summed_adc = torch.cat(
                    (self.batch_summed_adc, data.summed_adc.unsqueeze(1).to(self.device)),
                    dim=0
                )
            elif self.mode == "view_cluster":
                # convert categories to probabilities.
                answer = torch.zeros(outputs[input].shape).to(self.device)
                for jj, batches in enumerate(torch.unique(data.batch)):
                    labels, counts = torch.unique(
                        data.category[(batch == batches), ii], return_counts=True, dim=0
                    )
                    for kk, label in enumerate(labels):
                        answer[jj][label] = counts[kk] / len(data.category[(batch == batches), ii])
                predictions = torch.cat(
                    (softmax, data.category.unsqueeze(1)),
                    dim=1
                ).to(self.device)

                self.batch_probabilities[input] = torch.cat(
                    (self.batch_probabilities[input], predictions),
                    dim=0
                )
                self.metrics[input].update(
                    softmax, torch.argmax(answer, axis=1)
                )

    # def compute(self):
    #     outputs = {}
    #     for ii, input in enumerate(self.inputs):
    #         outputs[input] = self.metrics[input].compute()
    #     return outputs