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
        classes:    list=[2],
        meta:   dict={}
    ):
        """
        """
        super(AdjustedRandIndexMetric, self).__init__(
            name, inputs, when_to_compute, meta
        )
        self.classes = classes
        self.batch_metric = {}
        self.batch_metric_individual = {}
        self.labels = {}
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
            # "rn222",
            "nuclear_recoil",
            "electron_recoil"
        ]
        self.reset_batch()

    def reset_batch(self):
        for ii, input in enumerate(self.inputs):
            self.metrics[input] = adjusted_rand_score
            self.batch_metric[input] = torch.empty(size=(0,1), dtype=torch.float, device=self.device)
            self.batch_metric_individual[input] = {}
            for jj in self.classes:
                self.batch_metric_individual[input][str(jj)] = torch.empty(size=(0,1), dtype=torch.float, device=self.device)

    def update(self,
        outputs,
        data,
    ):
        for ii, input in enumerate(self.inputs):
            self.batch_metric[input] = torch.cat(
                (self.batch_metric[input], torch.tensor([[self.metrics[input](outputs[input], data.clusters.squeeze(1))]], device=self.device)),
                dim=0
            )
            for jj in self.classes:
                self.batch_metric_individual[input][str(jj)] = torch.cat(
                (self.batch_metric_individual[input][str(jj)], torch.tensor([[self.metrics[input](outputs[input][(data.category[:,ii] == jj)], data.clusters.squeeze(1)[(data.category[:,ii] == jj)])]], device=self.device)),
                dim=0
            )
    
    def compute(self):
        return self.batch_metric, self.batch_metric_individual