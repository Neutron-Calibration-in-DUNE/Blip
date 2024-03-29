"""
Generic losses for blip.
"""
import torch

from blip.utils.logger import Logger


class GenericLoss:
    """
    Abstract base class for Blip losses.  The inputs are
        1. name - a unique name for the loss function.
        2. alpha - a coefficient (should be from 0.0-1.0) for the strength of the influence of the loss.
        3. target_type - specifies whether the targets are features/classes/clusters/hits/etc.
        4. targets - list of names for the targets.
        5. outputs - list of names of the associated outputs for each target.
        6. augmentations - specified whether augmentations are created for the dataset
        7. meta - meta information from the module.
    """
    def __init__(
        self,
        name:           str = 'generic_loss',
        alpha:          float = 0.0,
        target_type:    str = 'classes',
        targets:        list = [],
        outputs:        list = [],
        augmentations:  int = 0,
        meta:           dict = {}
    ):
        self.name = name
        self.logger = Logger(self.name, output="both", file_mode="w")
        if not isinstance(alpha, list):
            self.alpha = [alpha for ii in range(len(targets))]
        else:
            if len(alpha) != len(targets):
                self.logger.error(f'specified alpha list {alpha} is not the same length as the number of targets ({targets})!')
            self.alpha = alpha
        self.target_type = target_type
        self.targets = targets
        self.outputs = outputs
        self.augmentations = augmentations
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']

        if len(self.targets) != len(self.outputs):
            self.logger.error(f'number of targets {self.targets} does not match number of outputs {self.outputs}!')

        if target_type == 'positions':
            self.loss = self.position_loss
            self.number_of_target_labels = [-1 for target in self.targets]
            self.target_indicies = [
                self.meta['dataset'].meta['blip_position_indices_by_name'][target] for target in self.targets
            ]
        elif target_type == 'features':
            self.loss = self.feature_loss
            self.number_of_target_labels = [-1 for target in self.targets]
            self.target_indicies = [
                self.meta['dataset'].meta['blip_features_indices_by_name'][target] for target in self.targets
            ]
        elif target_type == 'classes':
            self.loss = self.classes_loss
            self.number_of_target_labels = [
                len(self.meta['dataset'].meta['blip_labels_values'][target]) for target in self.targets
            ]
            self.target_indicies = [
                self.meta['dataset'].meta['blip_classes_indices_by_name'][target] for target in self.targets
            ]
        elif target_type == 'clusters':
            self.loss = self.cluster_loss
            self.number_of_target_labels = [-1 for target in self.targets]
            self.target_indicies = [
                self.meta['dataset'].meta['blip_clusters_indices_by_name'][target] for target in self.targets
            ]
        elif target_type == 'hit':
            self.loss = self.hit_loss
            self.number_of_target_labels = [-1 for target in self.targets]
            self.target_indicies = [
                self.meta['dataset'].meta['blip_hits_indices_by_name'][target] for target in self.targets
            ]
        elif target_type == 'augment_batch':
            self.loss = self.augment_batch_loss
        else:
            self.logger.error(f'specified target_type "{target_type}" not allowed!')

        # construct batch loss dictionaries
        self.batch_loss = {
            key: torch.empty(size=(0, 1), dtype=torch.float, device=self.device)
            for key in self.targets
        }

    def reset_batch(self):
        for key in self.batch_loss.keys():
            self.batch_loss[key] = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)

    def set_device(
        self,
        device
    ):
        self.device = device
        for key in self.batch_loss.keys():
            self.batch_loss[key] = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)

    def _loss(
        self,
        target,
        outputs
    ):
        self.logger.error('"_loss" not implemented in Loss!')

    def position_loss(
        self,
        outputs,
        data,
    ):
        target = {
            key: data.pos[:, self.target_indicies[ii]]
            for ii, key in enumerate(self.targets)
        }
        if self.augmentations > 0:
            target = {
                key: torch.cat([target[key] for ii in range(outputs['augmentations'])])
                for key in self.targets
            }
        return self._loss(target, outputs)

    def feature_loss(
        self,
        outputs,
        data,
    ):
        target = {
            key: data.x[:, self.target_indicies[ii]]
            for ii, key in enumerate(self.targets)
        }
        if self.augmentations > 0:
            target = {
                key: torch.cat([target[key] for ii in range(outputs['augmentations'])])
                for key in self.targets
            }
        return self._loss(target, outputs)

    def classes_loss(
        self,
        outputs,
        data,
    ):
        target = {
            key: data.category[:, self.target_indicies[ii]]
            for ii, key in enumerate(self.targets)
        }
        if self.augmentations > 0:
            target = {
                key: torch.cat([target[key] for ii in range(outputs['augmentations'])])
                for key in self.targets
            }
        return self._loss(target, outputs)

    def cluster_loss(
        self,
        outputs,
        data,
    ):
        target = {
            key: data.clusters[:, self.target_indicies[ii]]
            for ii, key in enumerate(self.targets)
        }
        if self.augmentations > 0:
            target = {
                key: torch.cat([target[key] for ii in range(outputs['augmentations'])])
                for key in self.targets
            }
        return self._loss(target, outputs)

    def hit_loss(
        self,
        outputs,
        data,
    ):
        target = {
            key: data.hits[:, self.target_indicies[ii]]
            for ii, key in enumerate(self.targets)
        }
        if self.augmentations > 0:
            target = {
                key: torch.cat([target[key] for ii in range(outputs['augmentations'])])
                for key in self.targets
            }
        return self._loss(target, outputs)

    def augment_batch_loss(
        self,
        outputs,
        data,
    ):
        indices = torch.arange(0, len(data.category), device=self.device)
        target = {
            key: torch.cat([
                indices
                for ii in range(int(len(outputs[key])/len(data.category)))
            ])
            for ii, key in enumerate(self.outputs)
        }
        return self._loss(target, outputs)
