"""
Wrapper for JaccardIndex loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from blip.losses import GenericLoss

class JaccardIndexLoss(GenericLoss):
    """
    """
    def __init__(self,
        name:           str='jaccard_index_loss',
        alpha:          float=0.0,
        target_type:    str='classes',
        targets:        list=[],
        outputs:        list=[],
        augmentations:  int=0,
        reduction:      str='mean',
        sigmoid:        bool=True,
        smooth:         float=1e-6,
        meta:           dict={}
    ):
        super(JaccardIndexLoss, self).__init__(
            name, alpha, target_type, targets, outputs, augmentations, meta
        )
        self.reduction = reduction
        self.sigmoid = sigmoid
        self.smooth = smooth
        if self.sigmoid:
            self.jaccard_index_loss = {
                key: self.sigmoid_jaccard_index
                for key in self.targets
            }
        else:
            self.jaccard_index_loss = {
                key: self.non_sigmoid_jaccard_index
                for key in self.targets
            }
        
    def sigmoid_jaccard_index(self,
        output,
        target
    ):
        output = F.sigmoid(output)
        output = output.view(-1)
        target = target.view(-1)
        intersection = (output * target).sum()
        total = (output + target).sum()
        union = total - intersection
        intersection_over_union = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - intersection_over_union

    def non_sigmoid_jaccard_index(self,
        output,
        target
    ):
        output = output.view(-1)
        target = target.view(-1)
        intersection = (output * target).sum()
        total = (output + target).sum()
        union = total - intersection
        intersection_over_union = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - intersection_over_union

    def _loss(self,
        target,
        outputs,
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, output in enumerate(self.outputs):
            loss += self.jaccard_index_loss[self.targets[ii]](
                outputs[output].to(self.device), 
                target[self.targets[ii]].to(self.device)
            )
        return self.alpha * loss