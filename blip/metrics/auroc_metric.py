"""
Confusion matrix metric.
"""
import torch
import torch.nn as nn
from torchmetrics.classification import AUROC
from torchmetrics.classification import MulticlassAUROC
from blip.metrics import GenericMetric

class AUROCMetric(GenericMetric):
    
    def __init__(self,
        name:       str='auroc_metric',
        inputs:     list=['classifications'],
        when_to_compute:    str='all',
        device:         str='cpu',
        num_classes:    int=[2],
    ):
        """
        """
        super(AUROCMetric, self).__init__(
            name, inputs, when_to_compute, device
        )

        self.num_classes = num_classes
        for input in self.inputs:
            if self.num_classes == 2:
                self.metric[input] = AUROC(task="binary", num_classes=2)
            else:
                self.metric[input] = MulticlassAUROC(num_classes=self.num_classes)
        
    def update(self,
        outputs,
        data,
    ):
        for input in self.inputs:
            self.metric.update(outputs[input], data.category)