"""
Wrapper for Tversky loss
"""
import torch
import torch.nn.functional as F

from blip.losses import GenericLoss


class TverskyLoss(GenericLoss):
    """
    """
    def __init__(
        self,
        name:           str = 'tversky_loss',
        alpha:          float = 0.0,
        target_type:    str = 'classes',
        targets:        list = [],
        outputs:        list = [],
        augmentations:  int = 0,
        reduction:      str = 'mean',
        sigmoid:        bool = True,
        beta:           float = 0.5,
        gamma:          float = 0.5,
        smooth:         float = 1e-6,
        meta:           dict = {}
    ):
        super(TverskyLoss, self).__init__(
            name, alpha, target_type, targets, outputs, augmentations, meta
        )
        self.reduction = reduction
        self.sigmoid = sigmoid
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        if self.sigmoid:
            self.tversky_loss = {
                key: self.sigmoid_tversky
                for key in self.targets
            }
        else:
            self.tversky_loss = {
                key: self.non_sigmoid_tversky
                for key in self.targets
            }

    def sigmoid_tversky(
        self,
        output,
        target
    ):
        output = F.sigmoid(output)
        return self.non_sigmoid_tversky(output, target)

    def non_sigmoid_tversky(
        self,
        output,
        target
    ):
        true_positives = torch.sum(target + output)
        false_positives = torch.sum(target * (1 - output))
        false_negatives = torch.sum((1 - target) * output)
        tversky_loss = (true_positives + self.smooth) / (
            true_positives + self.beta * false_positives + self.gamma * false_negatives + self.smooth
        )
        return 1.0 - tversky_loss

    def _loss(
        self,
        target,
        outputs,
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, output in enumerate(self.outputs):
            temp_loss = self.alpha[ii] * self.tversky_loss[self.targets[ii]](
                outputs[output].to(self.device),
                F.one_hot(target[self.targets[ii]], num_classes=self.number_of_target_labels[ii]).to(self.device)
            )
            loss += temp_loss
            self.batch_loss[self.targets[ii]] = torch.cat(
                (self.batch_loss[self.targets[ii]], torch.tensor([[temp_loss]], device=self.device)), dim=0
            )
        return loss
