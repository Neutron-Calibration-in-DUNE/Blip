"""
Wrapper for CrossEntropy loss
"""
import torch
import torch.nn as nn

from blip.losses import GenericLoss

class MultiClassCrossEntropyLoss(GenericLoss):
    """
    """
    def __init__(self,
        alpha: float=1.0,
        name:  str='cross_entropy_loss',
        classes:    list=[],
        reduction:  str='mean',
        class_weights:  dict={},
        device:     str='cpu'
    ):
        super(MultiClassCrossEntropyLoss, self).__init__(name, device)
        self.alpha = alpha
        self.reduction = reduction
        self.classes = classes
        self.class_weights = class_weights
        if len(class_weights.keys()) > 0:
            self.cross_entropy_loss = {
                key: nn.CrossEntropyLoss(
                    weight=self.class_weights[key].to(self.device), 
                    reduction=self.reduction
                )
                for key in self.class_weights.keys()
            }
        else:
            self.cross_entropy_loss = {
                key: nn.CrossEntropyLoss(reduction=self.reduction)
                for key in self.classes
            }

    def loss(self,
        outputs,
        data,
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, classes in enumerate(self.classes):
            loss += self.cross_entropy_loss[classes](outputs[classes], data.category[:,ii].to(self.device))
        self.batch_loss = torch.cat((self.batch_loss, torch.tensor([[loss]], device=self.device)), dim=0)
        return self.alpha * loss