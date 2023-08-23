"""
Wrapper for Dice loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from blip.losses import GenericLoss

class DiceLoss(GenericLoss):
    """
    """
    def __init__(self,
        name:           str='dice_loss',
        alpha:          float=0.0,
        target_type:    str='classes',
        targets:        list=[],
        outputs:        list=[],
        augmentations:  int=0,
        reduction:      str='mean',
        sigmoid:        bool=False,
        smooth:         float=1e-6,
        meta:           dict={}
    ):
        super(DiceLoss, self).__init__(
            name, alpha, target_type, targets, outputs, augmentations, meta
        )
        self.reduction = reduction
        self.sigmoid = sigmoid
        self.smooth = smooth
        if sigmoid:
            self.dice_loss = {
                key: self.sigmoid_dice
                for key in self.targets
            }
        else:
            self.dice_loss = {
                key: self.non_sigmoid_dice
                for key in self.targets
            }
        
    def sigmoid_dice(self,
        output,
        target
    ):
        output = F.sigmoid(output)
        return self.non_sigmoid_dice(output, target)   

    def non_sigmoid_dice(self,
        output,
        target
    ):
        intersection = torch.sum(output * target)
        dice = (2.0 * intersection + self.smooth) / (torch.sum(output) + torch.sum(target) + self.smooth)
        return 1.0 - dice

    def _loss(self,
        target,
        outputs,
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, output in enumerate(self.outputs):
            temp_loss = self.dice_loss[self.targets[ii]](
                outputs[output].to(self.device), 
                F.one_hot(target[self.targets[ii]], num_classes=self.number_of_target_labels[ii]).to(self.device)
            )
            loss += temp_loss
            self.batch_loss[self.targets[ii]] = torch.cat(
                (self.batch_loss[self.targets[ii]], torch.tensor([[temp_loss]], device=self.device)), dim=0
            )
        return self.alpha * loss