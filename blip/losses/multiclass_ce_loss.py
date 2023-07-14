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
        meta:   dict={}
    ):
        super(MultiClassCrossEntropyLoss, self).__init__(name, meta)
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
            augmented_labels = torch.cat([
                data.category.to(self.device) 
                for ii in range(int(len(outputs['reductions'])/len(data.category)))
            ])
            loss += self.cross_entropy_loss[classes](outputs[classes], augmented_labels)
        return self.alpha * loss