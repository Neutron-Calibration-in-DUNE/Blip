"""
Confusion matrix metric.
"""
import torch
import torch.nn as nn
from torchmetrics.classification import Dice
from blip.metrics import GenericMetric

class DiceScoreMetric(GenericMetric):
    
    def __init__(self,
        name:       str='dice_score',
        shape:      tuple=(),
        input:      str='classifications',
        num_classes:    int=2,
    ):
        """
        """
        super(DiceScoreMetric, self).__init__(
            name,
            shape,
            input
        )
        self.num_classes = num_classes
        self.metric = Dice(num_classes=self.num_classes)
        

    def update(self,
        outputs,
        data,
    ):
        self.metric.update(outputs["classifications"], data.category)

    def compute(self):
        return self.metric.compute()