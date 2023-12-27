"""
Generic metric callback
"""
import os
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from blip.losses.loss_handler import LossHandler
from blip.metrics.metric_handler import MetricHandler
from blip.utils.callbacks import GenericCallback
from blip.dataset.common import *


class AdjustedRandIndexCallback(GenericCallback):
    """
    """
    def __init__(
        self,
        name:   str = 'adjusted_rand_index_callback',
        criterion_handler: LossHandler = None,
        metrics_handler: MetricHandler = None,
        meta:   dict = {}
    ):
        super(AdjustedRandIndexCallback, self).__init__(
            name, criterion_handler, metrics_handler, meta
        )

        if "AdjustedRandIndexMetric" in self.metrics_handler.metrics.keys():
            self.metric = self.metrics_handler.metrics["AdjustedRandIndexMetric"]

        if not os.path.isdir("plots/adjust_rand_index/"):
            os.makedirs("plots/adjust_rand_index/")

        # self.adjusted_rand_index = self.metric.batch_metric.copy()
        # self.adjusted_rand_index_individual = copy.deepcopy(self.metric.batch_metric_individual)
        # self.parameter_values = []

    def reset_batch(self):
        pass

    def evaluate_epoch(
        self,
        train_type='training'
    ):
        pass
        # batch_metric, batch_metric_individual = self.metric.compute()
        # for ii, input in enumerate(self.metric.inputs):
        #     if len(self.adjusted_rand_index[input]) == 0:
        #         self.adjusted_rand_index[input] = batch_metric[input]
        #     else:
        #         self.adjusted_rand_index[input] = torch.cat(
        #             (self.adjusted_rand_index[input], batch_metric[input]),
        #             dim=1
        #         )
        #     for jj in self.metric.classes:
        #         if len(self.adjusted_rand_index_individual[input][str(jj)]) == 0:
        #             self.adjusted_rand_index_individual[input][str(jj)] = batch_metric_individual[input][str(jj)]
        #         else:
        #             self.adjusted_rand_index_individual[input][str(jj)] = torch.cat(
        #                 (self.adjusted_rand_index_individual[input][str(jj)], batch_metric_individual[input][str(jj)]),
        #                 dim=1
        #             )
        # self.metric.reset_batch()

    def evaluate_clustering(self):
        pass
        # for ii, input in enumerate(self.metric.inputs):
        #     self.adjusted_rand_index[input] = self.adjusted_rand_index[input].cpu()
        #     for jj in self.metric.classes:
        #         self.adjusted_rand_index_individual[input][str(jj)] = self.adjusted_rand_index_individual[input][str(jj)].cpu()
        # for ii, input in enumerate(self.metric.inputs):
        #     fig, axs = plt.subplots(figsize=(10,6))
        #     means = self.adjusted_rand_index[input].mean(dim=0)
        #     stds = self.adjusted_rand_index[input].std(dim=0)
        #     hi_error = (means + stds)
        #     hi_error[(hi_error > 1.0)] = 1.0
        #     lo_error = (means - stds)
        #     lo_error[(lo_error < 0.0)] = 0.0

        #     axs.errorbar(
        #         self.parameter_values,
        #         means,
        #         yerr=[means.numpy() - lo_error.numpy(), hi_error.numpy() - means.numpy()],
        #         capsize=2,
        #         color='k',
        #         label=input
        #     )
        #     axs.set_xlabel('eps')
        #     axs.set_ylabel('Adjusted Rand Index')
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.savefig(f'plots/adjust_rand_index/{input}.png')

        #     fig_total, axs_total = plt.subplots(figsize=(10,6))
        #     for jj in self.metric.classes:
        #         means = self.adjusted_rand_index_individual[input][str(jj)].mean(dim=0)
        #         stds = self.adjusted_rand_index_individual[input][str(jj)].std(dim=0)
        #         hi_error = (means + stds)
        #         hi_error[(hi_error > 1.0)] = 1.0
        #         lo_error = (means - stds)
        #         lo_error[(lo_error < 0.0)] = 0.0
        #         axs_total.errorbar(
        #             self.parameter_values,
        #             means,
        #             yerr=[means.numpy() - lo_error.numpy(), hi_error.numpy() - means.numpy()],
        #             capsize=2,
        #             label=f'{self.metric.labels["particle"][jj]}'
        #         )
        #     axs_total.set_xlabel('eps')
        #     axs_total.set_ylabel('Adjusted Rand Index')
        #     plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        #     plt.tight_layout()
        #     plt.savefig(f'plots/adjust_rand_index/{input}_classes.png')

        #     for jj in self.metric.classes:
        #         fig_total, axs_total = plt.subplots(figsize=(10,6))
        #         means = self.adjusted_rand_index_individual[input][str(jj)].mean(dim=0)
        #         stds = self.adjusted_rand_index_individual[input][str(jj)].std(dim=0)
        #         hi_error = (means + stds)
        #         hi_error[(hi_error > 1.0)] = 1.0
        #         lo_error = (means - stds)
        #         lo_error[(lo_error < 0.0)] = 0.0
        #         axs_total.errorbar(
        #             self.parameter_values,
        #             means,
        #             yerr=[means.numpy() - lo_error.numpy(), hi_error.numpy() - means.numpy()],
        #             capsize=2,
        #             label=f'{self.metric.labels["particle"][jj]}'
        #         )
        #         axs_total.set_xlabel('eps')
        #         axs_total.set_ylabel('Adjusted Rand Index')
        #         plt.legend()
        #         plt.tight_layout()
        #         plt.savefig(f'plots/adjust_rand_index/{input}_{self.metric.labels["particle"][jj]}.png')
