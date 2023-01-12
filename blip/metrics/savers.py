"""
Generic saver metric class for tpc_ml.
"""
import torch
import torch.nn as nn

from blip.metrics import GenericMetric

class DataSaver(GenericMetric):
    
    def __init__(self,
        metric:       str='data_saver',
        shape:        tuple=(),
        output:       str='position'
    ):
        """
        Data Saver
        """
        super(DataSaver, self).__init__(
            metric,
            shape
        )
        self.output = output

         # create empty tensors for epoch 
        self.batch_data = torch.empty(
            size=(0, *self.shape),
            dtype=torch.float,device='cpu'
        )

        self.epoch_data = None

        if self.output == "position":
            self.update = self._update_position
        elif self.output == "category":
            self.update = self._update_category
        elif self.output == "augmented_category":
            self.update = self._update_augmented_category
        else:
            self.update = self._update
        
    def reset_batch(self):
        if self.batch_data == None:
            self.epoch_data = self.batch_data
        self.batch_data = torch.empty(
            size=(0, *self.shape),
            dtype=torch.float,device='cpu'
        )

    def _update(self,
        outputs,
        data,
    ):
        self.batch_data = torch.cat(
            (self.batch_data, outputs[self.output].detach().cpu()),
            dim=0
        )

    def _update_position(self,
        outputs,
        data,
    ):
        self.batch_data = torch.cat(
            (self.batch_data, data.pos.detach().cpu()),
            dim=0
        )

    def _update_category(self,
        outputs,
        data,
    ):
        print(self.shape)
        print(self.batch_data)
        print(data.category)
        print(data.category.shape)
        self.batch_data = torch.cat(
            (self.batch_data, data.category.detach().cpu()),
            dim=0
        )
    
    def _update_augmented_category(self,
        outputs,
        data,
    ):
        augmented_labels = torch.cat([
            data.category.to('cpu') 
            for ii in range(int(len(outputs["reductions"])/len(data.category)))
        ])
        self.batch_data = torch.cat(
            (self.batch_data, augmented_labels.detach().cpu()), 
            dim=0
        )

    def compute(self):
        pass