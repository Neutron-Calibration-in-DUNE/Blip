"""
Generic metric class for blip.
"""
import torch

from blip.utils.logger import Logger


class GenericMetric:
    """
    Abstract base class for Blip metrics.  The inputs are
        1. name - a unique name for the metric function.
        2. target_type - specifies whether the targets are features/classes/clusters/hits/etc.
        3. when_to_compute - when to compute the metric, 'train', 'validation', 'test', 'train_all', 'inference', 'all'
        4. targets - list of names for the targets.
        5. outputs - list of names of the associated outputs for each target.
        6. augmentations - specified whether augmentations are created for the dataset
        7. meta - meta information from the module.
    """
    def __init__(
        self,
        name:           str = 'generic',
        target_type:        str = 'classes',
        when_to_compute:    str = 'all',
        targets:        list = [],
        outputs:        list = [],
        augmentations:  int = 0,
        meta:           dict = {}
    ):
        self.name = name
        self.logger = Logger(self.name, output="both", file_mode="w")
        self.target_type = target_type
        self.when_to_compute = when_to_compute
        self.targets = targets
        self.outputs = outputs
        self.augmentations = augmentations
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']

        if len(self.targets) != len(self.outputs):
            self.logger.error(f'number of targets {self.targets} does not match number of outputs {self.outputs}!')

        if target_type == 'positions':
            self.update = self.position_metric_update
            self.number_of_target_labels = [-1 for target in self.targets]
            self.target_indicies = [
                self.meta['dataset'].meta['blip_position_indices_by_name'][target] for target in self.targets
            ]
        elif target_type == 'features':
            self.update = self.feature_metric_update
            self.number_of_target_labels = [-1 for target in self.targets]
            self.target_indicies = [
                self.meta['dataset'].meta['blip_features_indices_by_name'][target] for target in self.targets
            ]
        elif target_type == 'classes':
            self.update = self.classes_metric_update
            self.number_of_target_labels = [
                len(self.meta['dataset'].meta['blip_labels_values'][target]) for target in self.targets
            ]
            self.target_indicies = [
                self.meta['dataset'].meta['blip_classes_indices_by_name'][target] for target in self.targets
            ]
        elif target_type == 'clusters':
            self.update = self.cluster_metric_update
            self.number_of_target_labels = [-1 for target in self.targets]
            self.target_indicies = [
                self.meta['dataset'].meta['blip_clusters_indices_by_name'][target] for target in self.targets
            ]
        elif target_type == 'hit':
            self.update = self.hit_metric_update
            self.number_of_target_labels = [-1 for target in self.targets]
            self.target_indicies = [
                self.meta['dataset'].meta['blip_hits_indices_by_name'][target] for target in self.targets
            ]
        elif target_type == 'augment_batch':
            self.update = self.augment_batch_metric_update
        else:
            self.logger.error(f'specified target_type "{target_type}" not allowed!')

    def set_device(
        self,
        device
    ):
        self.device = device

    def _reset_batch(self):
        pass

    def reset_batch(self):
        self._reset_batch()

    def _metric_update(
        self,
        target,
        outputs
    ):
        self.logger.error('"_metric_update" not implemented in Metric!')

    def position_metric_update(
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
        return self._metric_update(target, outputs)

    def feature_metric_update(
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
        return self._metric_update(target, outputs)

    def classes_metric_update(
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
        return self._metric_update(target, outputs)

    def cluster_metric_update(
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
        return self._metric_update(target, outputs)

    def hit_metric_update(
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
        return self._metric_update(target, outputs)

    def augment_batch_metric_update(
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
        return self._metric_update(target, outputs)

    def _metric_compute(self):
        pass

    def compute(self):
        return self._metric_compute()
