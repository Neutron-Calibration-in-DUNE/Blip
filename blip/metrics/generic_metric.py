"""
Generic metric class for blip.
"""
import torch

class GenericMetric:
    """
    """
    def __init__(self,
        name:       str='generic',
        inputs:     list=['reductions'],
        when_to_compute:   str='all',
        meta:   dict={}
    ):
        self.name = name
        self.inputs = inputs
        self.when_to_compute = when_to_compute
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']

        self.metrics = {input: None for input in self.inputs}

    def reset(self):
        for name, metric in self.metrics.items():
            metric.reset()

    def set_device(self,
        device
    ):  
        self.device = device
        for name, metric in self.metrics.items():
            metric.to(self.device)
        self.reset()

    def update(self,
        outputs,
        data,
    ):
        pass

    def compute(self):
        return {
            input: self.metrics[input].compute() 
            for input in self.inputs
        }