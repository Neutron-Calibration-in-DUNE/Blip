"""
Confusion matrix metric.
"""
import torch
import torch.nn as nn
from torchmetrics.classification import JaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex
from blip.metrics import GenericMetric

class JaccardIndexMetric(GenericMetric):
    
    def __init__(self,
        name:       str='jaccard_index',
        shape:      tuple=(),
        input:      str='classifications',
        num_classes:    int=2,
        device: str='cpu'
    ):
        """
        """
        super(JaccardIndexMetric, self).__init__(
            name, shape, input, device
        )
        self.num_classes = num_classes
        if self.num_classes == 2:
            self.metric = JaccardIndex(task="binary", num_classes=2)
        else:
            self.metric = MulticlassJaccardIndex(num_classes=self.num_classes)
        

    def update(self,
        outputs,
        data,
    ):
        self.metric.update(outputs["classifications"], data.category)

    def compute(self):
        return self.metric.compute()