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
        mode:       str="voxel",
        inputs:             list=[],
        number_of_classes:  list=[],
        when_to_compute:    str="all",
        consolidate_classes:   dict=None,
        device: str='cpu'
    ):
        """
        """
        super(ConfusionMatrixMetric, self).__init__(
            name, inputs, when_to_compute, device
        )
        self.mode = mode
        self.inputs = inputs
        self.number_of_classes = number_of_classes

        self.metrics = {}
        self.batch_predictions = {}
        self.batch_summed_adc = {}
        if consolidate_classes is not None:
            self.consolidate_classes = True
        else:
            self.consolidate_classes = False
        self.labels = {}

        for ii, input in enumerate(self.inputs):
            self.metrics[input] = MulticlassConfusionMatrix(
                num_classes=self.number_of_classes[ii]
            )
            if consolidate_classes is not None:
                self.labels[input] = consolidate_classes[input]
            else:
                #self.labels[input] = classification_labels[input].values()
                self.labels['particle'] = [
                    "capture_gamma", 
                    "capture_gamma_474", 
                    "capture_gamma_336",
                    "capture_gamma_256",
                    "capture_gamma_118",
                    "capture_gamma_083",
                    "capture_gamma_051",
                    "capture_gamma_016",
                    "capture_gamma_other",
                    "ar39",
                    "ar42",
                    "kr85",
                    "rn222",
                    "nuclear_recoil",
                    "electron_recoil"
                ]
    #         if self.mode == "voxel":
    #             self.batch_predictions[input] = torch.empty(
    #                 size=(0, self.number_of_classes[ii] + 1),
    #                 dtype=torch.float, device=self.device
    #             )
    #         elif self.mode == "cluster":
    #             self.batch_predictions[input] = torch.empty(
    #                 size=(0, self.number_of_classes[ii] * 2),
    #                 dtype=torch.float, device=self.device
    #             )

    # def reset_probabilities(self):
    #     for ii, input in enumerate(self.inputs):
    #         if self.mode == "voxel":
    #             self.batch_predictions[input] = torch.empty(
    #                 size=(0, self.number_of_classes[ii] + 1),
    #                 dtype=torch.float, device=self.device
    #             )
    #         elif self.mode == "cluster":
    #             self.batch_predictions[input] = torch.empty(
    #                 size=(0, self.number_of_classes[ii] * 2),
    #                 dtype=torch.float, device=self.device
    #             )
    
    # def reset(self,
    # ):
    #     for ii, input in enumerate(self.inputs):
    #         self.metrics[input].reset()
    #     self.reset_probabilities()

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
            if self.mode == "voxel":
                # predictions = torch.cat(
                #     (softmax, data.category[:, ii].unsqueeze(1).to(self.device)),
                #     dim=1
                # ).to(self.device)
                # self.batch_predictions[input] = torch.cat(
                #     (self.batch_predictions[input], predictions),
                #     dim=0
                # )
                self.metrics[input].update(
                    softmax, data.category[:,ii].to(self.device)
                )
            elif self.mode == "cluster":
                # convert categories to probabilities.
                answer = torch.zeros(outputs[input].shape).to(self.device)
                for jj, batches in enumerate(torch.unique(data.batch)):
                    labels, counts = torch.unique(
                        data.category[(batch == batches), ii], return_counts=True, dim=0
                    )
                    for kk, label in enumerate(labels):
                        answer[jj][label] = counts[kk] / len(data.category[(batch == batches), ii])
                # predictions = torch.cat(
                #     (softmax, answer),
                #     dim=1
                # ).to(self.device)
                # self.batch_predictions[input] = torch.cat(
                #     (self.batch_predictions[input], predictions),
                #     dim=0
                # )
                self.metrics[input].update(
                    softmax, torch.argmax(answer, axis=1)
                )

    # def compute(self):
    #     outputs = {}
    #     for ii, input in enumerate(self.inputs):
    #         outputs[input] = self.metrics[input].compute()
    #     return outputs