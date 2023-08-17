"""
Wrapper for CrossEntropy loss
"""
import torch
import torch.nn as nn

from blip.losses import GenericLoss

class MultiClassProbabilityLoss(GenericLoss):
    """
    """
    def __init__(self,
        alpha: float=1.0,
        name:  str='probability_loss',
        classes:    list=[],
        reduction:  str='mean',
        class_weights:  dict={},
        meta:   dict={}
    ):
        super(MultiClassProbabilityLoss, self).__init__(name, meta)
        self.alpha = alpha
        self.reduction = reduction
        self.classes = classes
        self.class_weights = class_weights
        if len(class_weights.keys()) > 0:
            self.cross_entropy_loss = {
                key: nn.MSELoss(
                    weight=self.class_weights[key].to(self.device), 
                    reduction=self.reduction
                )
                for key in self.class_weights.keys()
            }
        else:
            self.cross_entropy_loss = {
                key: nn.MSELoss(reduction=self.reduction)
                for key in self.classes
            }

    def loss(self,
        outputs,
        data,
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        batch = data.batch
        for ii, classes in enumerate(self.classes):
            # convert categories to probabilities.
            answer = torch.zeros(outputs[classes].shape)
            for jj, batches in enumerate(torch.unique(data.batch)):
                labels, counts = torch.unique(
                    data.category[(batch == batches), ii], return_counts=True, dim=0
                )
                for kk, label in enumerate(labels):
                    answer[jj][label] = counts[kk] / len(data.category[(batch == batches), ii])
            loss += self.cross_entropy_loss[classes](outputs[classes], answer.to(self.device))
        return self.alpha * loss