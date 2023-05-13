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
        input:      str='reductions',
        when_to_compute:   str='all',
        device: str='cpu'
    ):
        self.name = name
        self.shape = shape
        self.input = input
        self.when_to_compute = when_to_compute
        # set device to none for now
        self.device = device

        # create empty tensors for evaluation
        self.batch_metric = torch.empty(
            size=(self.shape), 
            dtype=torch.float, device=self.device
        )
        self.metric = None

    def reset(self):
        return self.metric.reset()

    def update(self,
        outputs,
        data,
    ):
        pass

    def set_device(self,
        device
    ):  
        self.device = device
        self.reset()

    def compute(self):
        pass