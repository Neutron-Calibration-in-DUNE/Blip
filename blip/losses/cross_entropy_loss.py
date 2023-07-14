"""
Wrapper for CrossEntropy loss
"""
import torch
import torch.nn as nn

from blip.losses import GenericLoss

class CrossEntropyLoss(GenericLoss):
    """
    """
    def __init__(self,
        alpha: float=1.0,
        name:   str='cross_entropy_loss',
        reduction:  str='mean',
        meta:   dict={}
    ):
        super(CrossEntropyLoss, self).__init__(name, meta)
        self.alpha = alpha
        self.reduction = reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=self.reduction)

    def loss(self,
        outputs,
        data,
    ):
        """Computes and returns/saves loss information"""
        embeddings = outputs['classifications']
        augmented_labels = torch.cat([
            data.category.to(self.device) 
            for ii in range(int(len(outputs['reductions'])/len(data.category)))
        ])
        loss = self.cross_entropy_loss(embeddings, augmented_labels)
        return self.alpha * loss