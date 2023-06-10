"""
Confusion matrix metric.
"""
import torch
import torch.nn as nn
from sklearn.metrics.cluster import adjusted_rand_score
from blip.metrics import GenericMetric

class AdjustedRandIndexMetric(GenericMetric):
    
    def __init__(self,
        name:       str='auroc_metric',
        inputs:     list=['classifications'],
        when_to_compute:    str='all',
        device:         str='cpu',
        num_classes:    int=[2],
    ):
        """
        """
        super(AdjustedRandIndexMetric, self).__init__(
            name, inputs, when_to_compute, device
        )
        self.batch_metric = {}
        for ii, input in enumerate(self.inputs):
            self.metrics[input] = adjusted_rand_score
            self.batch_metric[input] = torch.empty(size=(0,1), dtype=torch.float, device=self.device) 

        
    def update(self,
        outputs,
        data,
    ):
        for input in self.inputs:
            self.batch_metric[input] = torch.cat(
                (self.batch_metric[input], torch.tensor([[self.metrics[input](outputs[input], data.clusters.squeeze(1))]], device=self.device)),
                dim=0
            )