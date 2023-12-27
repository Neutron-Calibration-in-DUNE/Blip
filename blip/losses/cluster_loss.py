"""
Wrapper for cluster loss
"""
import torch

from blip.losses import GenericLoss


class ClusterLoss(GenericLoss):
    """
    This loss attempts to push signal points towards the origin
    and background points away from the origin.
    """
    def __init__(
        self,
        name:           str = 'cluster_loss',
        alpha:          float = 0.0,
        target_type:    str = 'classes',
        targets:        list = [],
        outputs:        list = [],
        augmentations:  int = 0,
        reduction:      str = 'mean',
        cluster_type:   str = 'normalized',
        fixed_value:    float = 1.0,
        meta:           dict = {}
    ):
        super(ClusterLoss, self).__init__(
            name, alpha, target_type, targets, outputs, augmentations, meta
        )
        if cluster_type == 'normalized':
            self.cluster_loss = {
                key: self._cluster_loss_normalized
                for key in self.targets
            }
        elif cluster_type == 'inverse':
            self.cluster_loss = {
                key: self._cluster_loss_inverse
                for key in self.targets
            }
        else:
            self.cluster_loss = {
                key: self._cluster_loss_fixed
                for key in self.targets
            }
        self.fixed_value = fixed_value

    def _cluster_loss_normalized(
        self,
        target,
        output
    ):
        lengths = torch.norm(output, p=2, dim=1)
        max_length = torch.max(lengths)
        signal_loss = target * lengths
        background_loss = (1 - target) * (1. - lengths/max_length)
        loss = signal_loss + background_loss
        return loss.mean()

    def _cluster_loss_inverse(
        self,
        target,
        output
    ):
        """
        For signal points,
        we want |x|^2 = 0, so the loss is simply the distance.
        For background points, we want |x|^2 -> inf, so the loss is:
        1 - |x|^2/|x|^2_max, where |x|^2_max is the largest distance
        in the batch.
        """
        lengths = torch.norm(output, p=2, dim=1)
        signal_loss = target * lengths
        background_loss = (1 - target) * (1./(lengths + 1e-16))
        loss = signal_loss + background_loss
        return loss.mean()

    def _cluster_loss_fixed(
        self,
        target,
        output
    ):
        """
        For signal points,
        we want |x|^2 = 0, so the loss is simply the distance.
        For background points, we want |x|^2 -> inf, so the loss is:
        1 - |x|^2/|x|^2_max, where |x|^2_max is the largest distance
        in the batch.
        """
        lengths = torch.norm(output, p=2, dim=1)
        signal_loss = target * lengths
        background_loss = (1 - target) * torch.abs(self.fixed_value - lengths)
        loss = signal_loss + background_loss
        return loss.mean()

    def _loss(
        self,
        target,
        outputs,
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, output in enumerate(self.outputs):
            temp_loss = self.alpha[ii] * self.cluster_loss[self.targets[ii]](
                target[self.targets[ii]].to(self.device),
                outputs[output].to(self.device)
            )
            loss += temp_loss
            self.batch_loss[self.targets[ii]] = torch.cat(
                (self.batch_loss[self.targets[ii]], torch.tensor([[temp_loss]], device=self.device)), dim=0
            )
        return loss
