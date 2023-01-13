"""
Confusion matrix metric.
"""
import torch
import torch.nn as nn
from torchmetrics.classification import Recall
from torchmetrics.classification import MulticlassRecall
from blip.metrics import GenericMetric

class RecallMetric(GenericMetric):
    
    def __init__(self,
        name:       str='recall',
        shape:      tuple=(),
        input:      str='classifications',
        num_classes:    int=2,
    ):
        """
        """
        super(RecallMetric, self).__init__(
            name,
            shape,
            input
        )
        self.num_classes = num_classes
        if self.num_classes == 2:
            self.metric = Recall(task="binary", num_classes=2)
        else:
            self.metric = MulticlassRecall(num_classes=self.num_classes)
        

    def update(self,
        outputs,
        data,
    ):
        self.metric.update(outputs["classifications"], data.category)

    def compute(self):
        return self.metric.compute()