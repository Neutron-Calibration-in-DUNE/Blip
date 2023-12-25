"""
Confusion matrix metric.
"""
import torch
from sklearn.metrics.cluster import adjusted_rand_score
from blip.metrics import GenericMetric


class AdjustedRandIndexMetric(GenericMetric):

    def __init__(
        self,
        name:           str = 'adjusted_rand_index',
        target_type:        str = 'classes',
        when_to_compute:    str = 'all',
        targets:        list = [],
        outputs:        list = [],
        augmentations:  int = 0,
        meta:           dict = {}
    ):
        """
        """
        super(AdjustedRandIndexMetric, self).__init__(
            name, target_type, when_to_compute, targets, outputs, augmentations, meta
        )
        self.adjusted_rand_index_function = adjusted_rand_score
        self.adjusted_rand_index_metric = {
            key: torch.empty(size=(0, 1), dtype=torch.float, device=self.device)
            for key in self.targets
        }
        self.adjusted_rand_index_metric_individual = {
            key: {
                label: torch.empty(size=(0, 1), dtype=torch.float, device=self.device)
                for label in self.meta['dataset'].meta['blip_labels_values'][key]
            }
            for key in self.meta['dataset'].meta['blip_labels_values']
        }

    def _reset_batch(self):
        for ii, output in enumerate(self.outputs):
            self.adjusted_rand_index_metric[output] = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)
            for jj, label in enumerate(self.adjusted_rand_index_metric_individual[output].keys()):
                self.adjusted_rand_index_metric_individual[output][label] = torch.empty(
                    size=(0, 1), dtype=torch.float, device=self.device
                )

    def _metric_update(
        self,
        target,
        outputs,
    ):
        for ii, output in enumerate(self.outputs):
            self.adjusted_rand_index_metric[output] = torch.cat((
                self.adjusted_rand_index_metric[output],
                torch.tensor([[
                    self.adjusted_rand_index_function(
                        target[self.targets[ii]].cpu(),
                        torch.argmax(outputs[output], dim=1).cpu()
                    )
                ]]).to(self.device)
            ), dim=0)
            for jj, label in enumerate(self.adjusted_rand_index_metric_individual[output].keys()):
                self.adjusted_rand_index_metric_individual[output][label] = torch.cat((
                    self.adjusted_rand_index_metric_individual[output][label],
                    torch.tensor([[
                        self.adjusted_rand_index_function(
                            target[self.targets[ii]][(target[self.targets[ii]] == label)].cpu(),
                            torch.argmax(outputs[output][(target[self.targets[ii]] == label)], dim=1).cpu()
                        )
                    ]]).to(self.device),
                ), dim=0)

    def _metric_compute(self):
        return self.adjusted_rand_index_metric, self.adjusted_rand_index_metric_individual
