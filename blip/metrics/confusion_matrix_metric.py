"""
Confusion matrix metric.
"""
import torch
import torch.nn as nn
from torchmetrics.classification import ConfusionMatrix
from torchmetrics.classification import MulticlassConfusionMatrix
from blip.metrics import GenericMetric

class ConfusionMatrixMetric(GenericMetric):
    
    def __init__(self,
        name:       str='confusion_matrix',
        shape:      tuple=(),
        input:      str='classifications',
        num_classes:    int=2,
    ):
        """
        """
        super(ConfusionMatrixMetric, self).__init__(
            name,
            shape,
            input
        )
        self.num_classes = num_classes
        if self.num_classes == 2:
            self.metric = ConfusionMatrix(task="binary", num_classes=2)
        else:
            self.metric = MulticlassConfusionMatrix(num_classes=self.num_classes)
        
        self.batch_probabilities = torch.empty(
            size=(0, self.num_classes + 1),
            dtype=torch.float, device=self.device
        )
        self.batch_summed_adc = torch.empty(
            size=(0, 1),
            dtype=torch.float, device=self.device
        )

    
    def reset_probabilities(self):
        self.batch_probabilities = torch.empty(
            size=(0, self.num_classes + 1),
            dtype=torch.float, device=self.device
        )
        self.batch_summed_adc = torch.empty(
            size=(0, 1),
            dtype=torch.float, device=self.device
        )
        
    def update(self,
        outputs,
        data,
    ):
        # get output probabilities for each class
        softmax = nn.functional.softmax(
            outputs["classifications"], 
            dim=1, dtype=torch.float
        )
        predictions = torch.cat(
            (softmax, data.category.unsqueeze(1)),
            dim=1
        ).to(self.device)

        self.batch_probabilities = torch.cat(
            (self.batch_probabilities, predictions),
            dim=0
        )
        
        # get summed adc from input
        self.batch_summed_adc = torch.cat(
            (self.batch_summed_adc, data.summed_adc.unsqueeze(1).to(self.device)),
            dim=0
        )
        self.metric.update(outputs["classifications"], data.category)

    def compute(self):
        return self.metric.compute()