"""
Wasserstein loss
"""
import numpy as np
import torch

from blip.losses import GenericLoss
from blip.utils.distributions import *

class WassersteinLoss(GenericLoss):
    """
    """
    def __init__(self,
        name:           str='wasserstein_loss',
        alpha:          float=0.0,
        target_type:    str='classes',
        targets:        list=[],
        outputs:        list=[],
        augmentations:  int=0,
        reduction:      str='mean',
        distribution_type:      str='gaussian',
        distribution_params:    dict={
            'number_of_samples':    1e6,
            'dimension':            5,
            'mean':                 0.0,
            'sigma':                1.0,
        },
        num_projections:    int=100,
        meta:           dict={}
    ):
        super(WassersteinLoss, self).__init__(
            name, alpha, target_type, targets, outputs, augmentations, meta
        )
        self.distribution_type = distribution_type
        if self.distribution_type == 'gaussian':
            self.distribution = generate_gaussian(**distribution_params)
        elif self.distribution_type == 'concentric_spheres':
            self.distribution = generate_concentric_spheres(**distribution_params)
        elif self.distribution_type == 'sphere':
            self.distribution = generate_sphere(**distribution_params)
        elif self.distribution_type == 'uniform_annulus':
            self.distribution = generate_uniform_annulus(**distribution_params)
        else:
            self.logger.error(f'specified distribution type {self.distribution_type} not allowed!')
        self.num_projections = num_projections
        self.wasserstein_loss = {
            key: wasserstein_loss
            for key in self.targets
        }

    def wasserstein_loss(self,
        target,
        output
    ):
        """
            We project our distribution onto a sphere and compute the Wasserstein
            distance between the distribution (target) and our expected 
            distribution (distribution_samples).
        """
        distribution_samples = self.distribution[
            torch.randint(high = self.distribution.size(0), size =(target.size(0),))
        ].to(self.device)

        # first, generate a random sample on a sphere
        embedding_dimension = distribution_samples.size(1)
        normal_samples = np.random.normal(
            size=(self.num_projections, embedding_dimension)
        )
        normal_samples /= np.sqrt((normal_samples**2).sum())
        projections = torch.tensor(normal_samples).transpose(0, 1).to(self.device)

        # now project the embedded samples onto the sphere
        encoded_projections = target.matmul(projections.float()).transpose(0, 1).to(self.device)
        distribution_projections = distribution_samples.float().matmul(projections.float()).transpose(0, 1).to(self.device)

        # calculate the distance between the distributions
        wasserstein_distance = (
            torch.sort(encoded_projections, dim=1)[0] -
            torch.sort(distribution_projections, dim=1)[0]
        )
        wasserstein_mean = (torch.pow(wasserstein_distance, 2))

        return wasserstein_mean.mean()

    def _loss(self,
        target,
        outputs,
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, output in enumerate(self.outputs):
            temp_loss = self.alpha[ii] * self.angular_loss[self.targets[ii]](
                outputs[output].to(self.device), 
                target[self.targets[ii]].to(self.device)
            )
            loss += temp_loss
            self.batch_loss[self.targets[ii]] = torch.cat(
                (self.batch_loss[self.targets[ii]], torch.tensor([[temp_loss]], device=self.device)), dim=0
            )
        return loss