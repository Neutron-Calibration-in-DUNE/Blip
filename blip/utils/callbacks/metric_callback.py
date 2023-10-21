"""
Generic metric callback
"""
import numpy as np
import torch
from matplotlib import pyplot as plt

from blip.metrics import *
from blip.utils.callbacks import GenericCallback
from blip.losses.loss_handler import LossHandler
from blip.metrics.metric_handler import MetricHandler
from blip.metrics import ConfusionMatrixMetric, AdjustedRandIndexMetric

class MetricCallback(GenericCallback):
    """
    """
    def __init__(self,
        name:   str='metric_callback',
        criterion_handler:  LossHandler=None,
        metrics_handler:    MetricHandler=None,
        skip_metrics:       bool=False,
        meta:               dict={}
    ):  
        super(MetricCallback, self).__init__(
            name, criterion_handler, metrics_handler, meta
        )
        self.skip_metrics = skip_metrics
        if metrics_handler != None:
            self.metric_names = []
            for name, metric in self.metrics_handler.metrics.items():
                if sum([
                    name == "AdjustedRandIndexMetric",
                    name == "ConfusionMatrixMetric",
                ]):
                    continue
                else:
                    self.metric_names.append(name)
            # containers for training metric
            self.training_metrics = torch.empty(
                size=(0,len(self.metric_names)), 
                dtype=torch.float, device=self.device
            )
            self.validation_metrics = torch.empty(
                size=(0,len(self.metric_names)), 
                dtype=torch.float, device=self.device
            )
            self.test_metrics = torch.empty(
                size=(0,len(self.metric_names)), 
                dtype=torch.float, device=self.device
            )
            self.training_target_metrics = {
                name: torch.empty(
                    size=(0,len(metric.targets)),
                    dtype=torch.float, device=self.device
                )
                for name, metric in self.metrics_handler.metrics.items()
                if sum([
                    name != "AdjustedRandIndexMetric",
                    name != "ConfusionMatrixMetric",
                ])
            }
            self.validation_target_metrics = {
                name: torch.empty(
                    size=(0,len(metric.targets)),
                    dtype=torch.float, device=self.device
                )
                for name, metric in self.metrics_handler.metrics.items()
                if sum([
                    name != "AdjustedRandIndexMetric",
                    name != "ConfusionMatrixMetric",
                ])
            }
            self.test_target_metrics = {
                name: torch.empty(
                    size=(0,len(metric.targets)),
                    dtype=torch.float, device=self.device
                )
                for name, metric in self.metrics_handler.metrics.items()
                if sum([
                    name != "AdjustedRandIndexMetric",
                    name != "ConfusionMatrixMetric",
                ])
            }

    def save_metrics(self):
        if not self.skip_metrics:
            for name in self.training_target_metrics.keys():
                self.training_target_metrics[name].cpu().numpy()
            for name in self.validation_target_metrics.keys():
                self.validation_target_metrics[name].cpu().numpy()
            for name in self.test_target_metrics.keys():
                self.test_target_metrics[name].cpu().numpy()
            np.savez(
                f"{self.meta['local_scratch']}/metrics.npz",
                metric_names=self.metric_names,
                training_metric=self.training_metrics.cpu().numpy(),
                validation_metric=self.validation_metrics.cpu().numpy(),
                test_metric=self.test_metrics.cpu().numpy(),
                training_target_metric=self.training_target_metrics,
                validation_target_metric=self.validation_target_metrics,
                test_target_metric=self.test_target_metrics
            )
        else:
            for name in self.test_target_metrics.keys():
                self.test_target_metrics[name].cpu().numpy()
            np.savez(
                f"{self.meta['local_scratch']}/metrics.npz",
                metric_names=self.metric_names,
                test_metric=self.test_metrics.cpu().numpy(),
                test_target_metric=self.test_target_metrics
            )

    def reset_batch(self):
        self.training_metrics = torch.empty(
            size=(0,len(self.metric_names)), 
            dtype=torch.float, device=self.device
        )
        self.validation_metrics = torch.empty(
            size=(0,len(self.metric_names)), 
            dtype=torch.float, device=self.device
        )
        self.test_metrics = torch.empty(
            size=(0,len(self.metric_names)), 
            dtype=torch.float, device=self.device
        )
        self.training_target_metrics = {
            name: torch.empty(
                size=(0,len(metric.targets)),
                dtype=torch.float, device=self.device
            )
            for name, metric in self.metrics_handler.metrics.items()
            if sum([
                name != "AdjustedRandIndexMetric",
                name != "ConfusionMatrixMetric",
            ])
        }
        self.validation_target_metrics = {
            name: torch.empty(
                size=(0,len(metric.targets)),
                dtype=torch.float, device=self.device
            )
            for name, metric in self.metrics_handler.metrics.items()
            if sum([
                name != "AdjustedRandIndexMetric",
                name != "ConfusionMatrixMetric",
            ])
        }
        self.test_target_metrics = {
            name: torch.empty(
                size=(0,len(metric.targets)),
                dtype=torch.float, device=self.device
            )
            for name, metric in self.metrics_handler.metrics.items()
            if sum([
                name != "AdjustedRandIndexMetric",
                name != "ConfusionMatrixMetric",
            ])
        }

    def evaluate_epoch(self,
        train_type='train'
    ):  
        temp_metrics = torch.empty(
            size=(1,0), 
            dtype=torch.float, device=self.device
        )             
        metrics = self.metrics_handler.compute(train_type)
        # run through metrics
        if train_type == 'train':
            if self.skip_metrics:
                return
            for name, metric in metrics.items():
                if sum([
                    name == "AdjustedRandIndexMetric",
                    name == "ConfusionMatrixMetric",
                ]):  
                    continue
                temp_metric = 0
                temp_target_metrics = torch.empty(
                    size=(1,0), 
                    dtype=torch.float, device=self.device
                )
                for target in metric.keys():
                    temp_metric += metric[target] / len(metric.keys())
                    temp_target_metrics = torch.cat(
                        (temp_target_metrics, torch.tensor([[metric[target]]], device=self.device)),
                        dim=1
                    )
                self.training_target_metrics[name] = torch.cat(
                    (self.training_target_metrics[name], temp_target_metrics),
                    dim=0
                )
                temp_metrics = torch.cat(
                    (temp_metrics, torch.tensor([[temp_metric]], device=self.device)),
                    dim=1
                )
            self.training_metrics = torch.cat(
                (self.training_metrics, temp_metrics),
                dim=0
            )
        elif train_type == 'validation':
            if self.skip_metrics:
                return
            for name, metric in metrics.items():
                if sum([
                    name == "AdjustedRandIndexMetric",
                    name == "ConfusionMatrixMetric",
                ]):    
                    continue
                temp_metric = 0
                temp_target_metrics = torch.empty(
                    size=(1,0), 
                    dtype=torch.float, device=self.device
                )
                for target in metric.keys():
                    temp_metric += metric[target] / len(metric.keys())
                    temp_target_metrics = torch.cat(
                        (temp_target_metrics, torch.tensor([[metric[target]]], device=self.device)),
                        dim=1
                    )
                self.validation_target_metrics[name] = torch.cat(
                    (self.validation_target_metrics[name], temp_target_metrics),
                    dim=0
                )
                temp_metrics = torch.cat(
                    (temp_metrics, torch.tensor([[temp_metric]], device=self.device)),
                    dim=1
                )
            self.validation_metrics = torch.cat(
                (self.validation_metrics, temp_metrics),
                dim=0
            )
        else:
            for name, metric in metrics.items():
                if sum([
                    name == "AdjustedRandIndexMetric",
                    name == "ConfusionMatrixMetric",
                ]):    
                    continue
                temp_metric = 0
                temp_target_metrics = torch.empty(
                    size=(1,0), 
                    dtype=torch.float, device=self.device
                )
                for target in metric.keys():
                    temp_metric += metric[target] / len(metric.keys())
                    temp_target_metrics = torch.cat(
                        (temp_target_metrics, torch.tensor([[metric[target]]], device=self.device)),
                        dim=1
                    )
                self.test_target_metrics[name] = torch.cat(
                    (self.test_target_metrics[name], temp_target_metrics),
                    dim=0
                )
                temp_metrics = torch.cat(
                    (temp_metrics, torch.tensor([[temp_metric]], device=self.device)),
                    dim=1
                )
            self.test_metrics = torch.cat(
                (self.test_metrics, temp_metrics),
                dim=0
            )

    def evaluate_training(self):
        pass

    def evaluate_testing(self):  
        if self.skip_metrics:
            self.save_metrics()
            return
        # evaluate metrics from training and validation
        if self.metrics_handler == None:
            return
        epoch_ticks = np.arange(1,self.epochs+1)
        # training plot
        fig, axs = plt.subplots(figsize=(15, 10))
        if len(self.training_metrics) != 0:
            for ii, metric in enumerate(self.metric_names):
                temp_metric = self.training_metrics[:,ii]
                final_metric_value = f"(final={temp_metric[-1]:.2e})"
                axs.plot(
                    epoch_ticks,
                    temp_metric.cpu().numpy(),
                    c=self.plot_colors[-(ii+1)],
                    label=rf"{metric}"
                )
                axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"{final_metric_value}"
                )
            axs.set_xlabel("epoch")
            axs.set_ylabel("metric")
            axs.set_yscale('log')
            plt.title("metric vs. epoch (training)")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{self.meta['local_scratch']}/plots/epoch_training_metrics.png")
        
        if len(self.validation_metrics) != 0:
            fig, axs = plt.subplots(figsize=(15, 10))
            for ii, metric in enumerate(self.metric_names):
                temp_metric = self.validation_metrics[:,ii]
                final_metric_value = f"(final={temp_metric[-1]:.2e})"
                axs.plot(
                    epoch_ticks,
                    temp_metric.cpu().numpy(),
                    c=self.plot_colors[-(ii+1)],
                    label=rf"{metric}"
                )
                axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"{final_metric_value}"
                )
            axs.set_xlabel("epoch")
            axs.set_ylabel("metric")
            axs.set_yscale('log')
            plt.title("metric vs. epoch (validation)")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{self.meta['local_scratch']}/plots/epoch_validation_metrics.png")

        if len(self.training_metrics) != 0 and len(self.validation_metrics) != 0:
            fig, axs = plt.subplots(figsize=(15, 10))
            for ii, metric in enumerate(self.metric_names):
                temp_training_metric = self.training_metrics[:,ii]
                temp_validation_metric = self.validation_metrics[:,ii]
                final_training_metric_value = f"(final={temp_training_metric[-1]:.2e})"
                final_validation_metric_value = f"(final={temp_validation_metric[-1]:.2e})"
                axs.plot(
                    epoch_ticks,
                    temp_training_metric.cpu().numpy(),
                    c=self.plot_colors[-(ii+1)],
                    linestyle='-',
                    label=rf"{metric}"
                )
                axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"{final_training_metric_value}"
                )
                axs.plot(
                    epoch_ticks,
                    temp_validation_metric.cpu().numpy(),
                    c=self.plot_colors[-(ii+1)],
                    linestyle='--',
                    label=rf"{metric}"
                )
                axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"{final_validation_metric_value}"
                )
            if len(self.test_metrics) != 0:
                for ii, metric in enumerate(self.metric_names):
                    temp_metric = self.test_metrics[:,ii]
                    final_metric_value = f"(final={temp_metric[-1]:.2e})"
                    axs.plot([],[],
                        marker='x',
                        linestyle='',
                        c=self.plot_colors[-(ii+1)],
                        label=rf"(test) {metric}"
                    )
                    axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"{final_metric_value}"
                )
            axs.set_xlabel("epoch")
            axs.set_ylabel("metric")
            axs.set_yscale('log')
            plt.title("metric vs. epoch")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{self.meta['local_scratch']}/plots/epoch_metrics.png")

        ########### Plots for each metric with target contributions ##########
        for name, metric in self.metrics_handler.metrics.items():
            if sum([
                name == "AdjustedRandIndexMetric",
                name == "ConfusionMatrixMetric",
            ]):    
                continue
            fig, axs = plt.subplots(figsize=(15, 10))
            for ii, target in enumerate(metric.targets):
                temp_training_metrics = self.training_target_metrics[name][:, ii]
                final_training_value = f"(final={temp_training_metrics[-1]:.2e})"
                axs.plot(
                    epoch_ticks,
                    temp_training_metrics.cpu().numpy(),
                    c=self.plot_colors[ii],
                    linestyle='-',
                    label=rf"{target:<12} {final_training_value:>16}"
                )
                axs.set_xlabel("epoch")
                axs.set_ylabel("metric")
                axs.set_yscale('log')
                plt.title(f"{name} - metric vs. epoch")
                plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
                plt.tight_layout()
                plt.savefig(f"{self.meta['local_scratch']}/plots/epoch_metric_{name}.png")
        # save metrics to npz
        self.save_metrics()

    def evaluate_inference(self):
        pass