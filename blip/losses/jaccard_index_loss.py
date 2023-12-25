"""
Wrapper for JaccardIndex loss
"""
import torch
import torch.nn.functional as F

from blip.losses import GenericLoss


class JaccardIndexLoss(GenericLoss):
    """
    """
    def __init__(
        self,
        name:           str = 'jaccard_index_loss',
        alpha:          float = 0.0,
        target_type:    str = 'classes',
        targets:        list = [],
        outputs:        list = [],
        augmentations:  int = 0,
        reduction:      str = 'mean',
        sigmoid:        bool = True,
        smooth:         float = 1e-6,
        meta:           dict = {}
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

    def sigmoid_jaccard_index(
        self,
        output,
        target
    ):
        output = F.sigmoid(output)
        return self.non_sigmoid_jaccard_index(output, target)

    def non_sigmoid_jaccard_index(
        self,
        output,
        target
    ):
        intersection = torch.sum(output * target)
        total = torch.sum(output + target)
        union = total - intersection
        intersection_over_union = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - intersection_over_union

    def _loss(
        self,
        target,
        outputs,
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, output in enumerate(self.outputs):
            temp_loss = self.alpha[ii] * self.jaccard_index_loss[self.targets[ii]](
                outputs[output].to(self.device),
                F.one_hot(target[self.targets[ii]], num_classes=self.number_of_target_labels[ii]).to(self.device)
            )
            loss += temp_loss
            self.batch_loss[self.targets[ii]] = torch.cat(
                (self.batch_loss[self.targets[ii]], torch.tensor([[temp_loss]], device=self.device)), dim=0
            )
        return loss
