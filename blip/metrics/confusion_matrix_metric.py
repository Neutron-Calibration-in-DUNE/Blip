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
        

    def update(self,
        outputs,
        data,
    ):
        self.metric.update(outputs["classifications"], data.category)

    def compute(self):
        return self.metric.compute()