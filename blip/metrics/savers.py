"""
Generic saver metric class for tpc_ml.
"""
import torch
import torch.nn as nn

from blip.metrics import GenericMetric

class OutputSaver(GenericMetric):
    
    def __init__(self,
        name:       str='output_saver',
        shape:      tuple=(),
        output:     str='reductions'
    ):
        """
        Output Saver
        """
        self.output = output
        super(OutputSaver, self).__init__(
            name,
            shape
        )
         # create empty tensors for epoch 
        self.batch_output = torch.empty(
            size=(self.shape), 
            dtype=torch.float, device=self.device
        )

        self.epoch_output = None
        
    def reset_batch(self):
        if self.batch_output == None:
            self.epoch_output = self.batch_output
        self.batch_output = torch.empty(
            size=(self.shape), 
            dtype=torch.float, device=self.device
        )

    def update(self,
        outputs,
        data,
    ):
        self.batch_output = torch.cat(
            (self.batch_output, outputs[self.output]),
            dim=0
        )

    def compute(self):
        pass

class AugmentedTargetSaver(GenericMetric):
    
    def __init__(self,
        name:   str='augmented_target_saver',
        shape:      tuple=(),
        output:     str='reductions'
    ):
        """
        Augmented Target Saver
        """
        super(AugmentedTargetSaver, self).__init__(
            name,
            shape,
            output
        )
         # create empty tensors for epoch 
        self.batch_target = torch.empty(
            size=(0,*self.target_shape), 
            dtype=torch.float, device=self.device
        )

        self.epoch_target = None
        
    def reset_batch(self):
        if len(self.batch_target) != 0:
            self.epoch_target = self.batch_target
        self.batch_target = torch.empty(
            size=(0,*self.target_shape), 
            dtype=torch.float, device=self.device
        )

    def update(self,
        outputs,
        data,
    ):
        augmented_labels = torch.cat([
            data.category.to(self.device) 
            for ii in range(int(len(outputs[2])/len(data.category)))
        ])
        self.batch_target = torch.cat((self.batch_target, augmented_labels), dim=0)

    def compute(self):
        pass

class TargetSaver(GenericMetric):
    
    def __init__(self,
        name:   str='target_saver',
        shape:      tuple=(),
        output:     str='reductions'
    ):
        """
        Target Saver
        """
        super(TargetSaver, self).__init__(
            name,
            shape,
            output
        )
         # create empty tensors for epoch 
        self.batch_target = torch.empty(
            size=(0,*self.target_shape), 
            dtype=torch.float, device=self.device
        )

        self.epoch_target = None
        
    def reset_batch(self):
        if len(self.batch_target) != 0:
            self.epoch_target = self.batch_target
        self.batch_target = torch.empty(
            size=(0,*self.target_shape), 
            dtype=torch.float, device=self.device
        )

    def update(self,
        outputs,
        data,
    ):
        self.batch_target = torch.cat(
            (self.batch_target, data.category.to(self.device)),
            dim=0
        )

    def compute(self):
        pass

class InputSaver(GenericMetric):
    
    def __init__(self,
        name:   str='input_saver',
        shape:      tuple=(),
        output:     str='reductions'
    ):
        """
        Input Saver
        """
        super(InputSaver, self).__init__(
            name,
            shape,
            output
        )
         # create empty tensors for epoch 
        self.batch_input = torch.empty(
            size=(0,*self.input_shape), 
            dtype=torch.float, device=self.device
        )
        self.epoch_input = None
        
    def reset_batch(self):
        if len(self.batch_input) != 0:
            self.epoch_input = self.batch_input
        self.batch_input = torch.empty(
            size=(0,*self.input_shape), 
            dtype=torch.float, device=self.device
        )

    def update(self,
        outputs,
        data,
    ):
        self.batch_input = torch.cat(
            (self.batch_input, data.x.to(self.device)),
            dim=0
        )

    def compute(self):
        pass