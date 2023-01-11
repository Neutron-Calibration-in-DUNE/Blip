"""
Generic metric class for blip.
"""
import torch

class GenericMetric:
    """
    """
    def __init__(self,
        name:       str='generic',
        shape:      tuple=(),
        output:     str='reductions',
        when_compute:   str='all',
    ):
        self.name = name
        self.shape = shape
        self.output = output
        self.when_compute = when_compute
        # set device to none for now
        self.device = 'cpu'

        # create empty tensors for evaluation
        self.batch_metric = torch.empty(
            size=(0,1), 
            dtype=torch.float, device=self.device
        )

    def reset_batch(self):
        self.batch_metric = torch.empty(
            size=(0,1), 
            dtype=torch.float, device=self.device
        )

    def update(self,
        outputs,
        data,
    ):
        pass

    def set_device(self,
        device
    ):  
        self.device = device
        self.reset_batch()

    def compute(self):
        pass