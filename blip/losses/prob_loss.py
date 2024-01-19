"""
Wrapper for CrossEntropy loss
"""
import torch
import torch.nn as nn

from blip.losses import GenericLoss


class MultiClassProbabilityLoss(GenericLoss):
    """
    """
    def __init__(
        self,
        alpha:      float = 1.0,
        name:       str = 'probability_loss',
        classes:    list = [],
        reduction:  str = 'mean',
        meta:       dict = {}
    ):
        super(MultiClassProbabilityLoss, self).__init__(
            name, alpha, meta)
        self.reduction = reduction
        self.classes = classes
        self.cross_entropy_loss = {
            key: nn.MSELoss(reduction=self.reduction, weight=self.meta['class_weights'][key])
            for key in self.classes
        }

    def loss(
        self,
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
