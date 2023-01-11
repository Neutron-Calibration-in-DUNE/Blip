"""
Confusion matrix metric.
"""
import torch
import torch.nn as nn

from blip.metrics import GenericMetric

class ConfusionMatrix(GenericMetric):
    
    def __init__(self,
        name:   str='confusion_matrix',
        shape:      tuple=(),
        output:     str='reductions',
        num_classes:    int=2,
    ):
        """
        """
        super(ConfusionMatrix, self).__init__(
            name,
            shape,
            output
        )
        self.num_classes = num_classes

    def update(self,
        outputs,
        data,
    ):
        # set predictions using cutoff
        predictions = (outputs[1][:, self.binary_variable] > self.cutoff).unsqueeze(1)
        accuracy = (predictions == data[1].to(self.device)).float().mean()
        self.batch_metric = torch.cat(
            (self.batch_metric, torch.tensor([[accuracy]], device=self.device)), 
            dim=0
        )

    def compute(self):
        return self.batch_metric.mean()