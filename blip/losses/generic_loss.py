"""
Generic losses for blip.
"""
import torch

class GenericLoss:
    """
    Abstract base class for Blip losses.  The inputs are
        1. name - a unique name for the loss function.
        2. alpha - a coefficient (should be from 0.0-1.0) for the strength of the influence of the loss.
        3. target_type - specifies whether the targets are features/classes/clusters/hits/etc.
        4. augmentations - specified whether augmentations are created for the dataset 
        5. meta - meta information from the module.
    """
    def __init__(self,
        name:       str='generic',
        alpha:      float=0.0,
        target_type:  str='classes',
        targets:        list=[],
        outputs:        list=[],
        augmentations:  int=0,
        meta:           dict={}
    ):
        self.name = name
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
            self.target_indicies = [
                self.meta['dataset'].meta['blip_position_indices_by_name'][target] for target in self.targets
            ]
        elif target_type == 'features':
            self.loss = self.feature_loss
            self.target_indicies = [
                self.meta['dataset'].meta['blip_features_indices_by_name'][target] for target in self.targets
            ]
        elif target_type == 'classes':
            self.loss = self.classes_loss
            self.target_indicies = [
                self.meta['dataset'].meta['blip_classes_indices_by_name'][target] for target in self.targets
            ]
        elif target_type == 'clusters':
            self.loss = self.cluster_loss
            self.target_indicies = [
                self.meta['dataset'].meta['blip_clusters_indices_by_name'][target] for target in self.targets
            ]
        elif target_type == 'hit':
            self.loss = self.hit_loss
            self.target_indicies = [
                self.meta['dataset'].meta['blip_hits_indices_by_name'][target] for target in self.targets
            ]
        elif target_type == 'augment_batch':
            self.loss = self.augment_batch_loss
        else:
            self.logger.error(f'specified target_type "{target_type}" not allowed!')

    def set_device(self,
        device
    ):  
        self.device = device

    def _loss(self, 
        target,
        outputs
    ):
        self.logger.error(f'"_loss" not implemented in Loss!')

    def position_loss(self,
        outputs,
        data,
    ):
        target = {
            key: data.pos[:, self.target_indicies[ii]]
            for ii, key in enumerate(self.targets)
        }
        if self.augmentations > 0:
            target = {
                key: torch.cat([target[key] for ii in range(self.augmentations)])
                for key in self.targets
            }
        return self._loss(target, outputs)
    
    def feature_loss(self,
        outputs,
        data,
    ):
        target = {
            key: data.x[:, self.target_indicies[ii]]
            for ii, key in enumerate(self.targets)
        }
        if self.augmentations > 0:
            target = {
                key: torch.cat([target[key] for ii in range(self.augmentations)])
                for key in self.targets
            }
        return self._loss(target, outputs)

    def classes_loss(self,
        outputs,
        data,
    ):
        target = {
            key: data.category[:, self.target_indicies[ii]]
            for ii, key in enumerate(self.targets)
        }
        if self.augmentations > 0:
            target = {
                key: torch.cat([target[key] for ii in range(self.augmentations)])
                for key in self.targets
            }
        return self._loss(target, outputs)

    def cluster_loss(self,
        outputs,
        data,
    ):
        target = {
            key: data.clusters[:, self.target_indicies[ii]]
            for ii, key in enumerate(self.targets)
        }
        if self.augmentations > 0:
            target = {
                key: torch.cat([target[key] for ii in range(self.augmentations)])
                for key in self.targets
            }
        return self._loss(target, outputs)

    def hit_loss(self,
        outputs,
        data,
    ):
        target = {
            key: data.hits[:, self.target_indicies[ii]]
            for ii, key in enumerate(self.targets)
        }
        if self.augmentations > 0:
            target = {
                key: torch.cat([target[key] for ii in range(self.augmentations)])
                for key in self.targets
            }
        return self._loss(target, outputs)

    def augment_batch_loss(self,
        outputs,
        data,
    ):
        indices = torch.arange(0, len(torch.unique(data.batch)), self.device)
        target = {
            key: torch.cat([
                indices
                for ii in range(int(len(data.batch)/len(torch.unique(data.batch))))
            ])
            for ii, key in enumerate(self.targets)
        }
        return self._loss(target, outputs)