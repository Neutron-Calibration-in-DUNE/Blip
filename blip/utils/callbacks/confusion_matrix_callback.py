"""
Generic metric callback
"""
import numpy as np
import torch
from matplotlib import pyplot as plt
import os
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from blip.metrics.savers import *
from blip.utils.callbacks import GenericCallback
from blip.utils import utils

class ConfusionMatrixCallback(GenericCallback):
    """
    """
    def __init__(self,
        metrics_list,
    ):  
        super(ConfusionMatrixCallback, self).__init__()
        self.metrics_list = metrics_list
        self.output_name = None
        self.target_name = None
        self.augmented_target_name = None
        self.input_name = None

        if metrics_list != None:
            for name, metric in self.metrics_list.metrics.items():
                if isinstance(metric, DataSaver):
                    if(metric.output == "classifications"):
                        self.output_name = name
                    elif(metric.output == "position"):
                        self.input_name = name
                    elif(metric.output == "category"):
                        self.target_name = name
                    elif(metric.output == "augmented_category"):
                        self.augmented_target_name = name

        if not os.path.isdir("plots/confusion_matrix/"):
            os.makedirs("plots/confusion_matrix/")

        # containers for training metrics
        if self.output_name != None:
            self.training_output = None
            self.validation_output = None
        if self.augmented_target_name != None:
            self.training_target = None
        if self.target_name != None:
            self.validation_target = None
            
    def reset_batch(self):
        pass

    def evaluate_epoch(self,
        train_type='training'
    ):  
        if train_type == 'training':
            if self.output_name != None:
                self.training_output = self.metrics_list.metrics[self.output_name].batch_data
            if self.augmented_target_name != None:
                self.training_target = self.metrics_list.metrics[self.augmented_target_name].batch_data 
        else:
            if self.output_name != None:
                self.validation_output = self.metrics_list.metrics[self.output_name].batch_data
            if self.target_name != None:
                self.validation_target = self.metrics_list.metrics[self.target_name].batch_data

    def evaluate_training(self):
        # plot the latent distributions
        if self.output_name != None:
            print(self.training_target)
            print(self.training_output)
            # compute confusion matrix for training data
            targets = self.training_target.cpu().numpy().flatten()
            outputs = np.argmax(self.training_output.cpu().numpy(),axis=1).flatten()
            print(targets)
            print(outputs)
            confusion = confusion_matrix(
                targets, outputs
            )
            print(confusion)

            fig, axs = plt.subplots(figsize=(10,6))
            

            axs.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.suptitle("Training Confusion Matrix")
            plt.tight_layout()
            plt.savefig(f"plots/confusion_matrix/training_confusion_matrix.png")
            
    def evaluate_testing(self):  
        pass

        # # evaluate metrics from training and validation
        # if self.metrics_list == None:
        #     return
        # epoch_ticks = np.arange(1,self.epochs+1)
        # # training plot
        # fig, axs = plt.subplots(figsize=(10,5))
        # if len(self.training_metrics) != 0:
        #     for ii, metric in enumerate(self.metric_names):
        #         temp_metric = self.training_metrics[:,ii]
        #         final_metric_value = f"(final={temp_metric[-1]:.2e})"
        #         axs.plot(
        #             epoch_ticks,
        #             temp_metric.cpu().numpy(),
        #             c=self.plot_colors[-(ii+1)],
        #             label=rf"{metric}"
        #         )
        #         axs.plot([],[],
        #             marker='',
        #             linestyle='',
        #             label=rf"{final_metric_value}"
        #         )
        #     axs.set_xlabel("epoch")
        #     axs.set_ylabel("metric")
        #     axs.set_yscale('log')
        #     plt.title("metric vs. epoch (training)")
        #     plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        #     plt.tight_layout()
        #     plt.savefig("plots/epoch_training_metrics.png")
        
        # if len(self.validation_metrics) != 0:
        #     fig, axs = plt.subplots(figsize=(10,5))
        #     for ii, metric in enumerate(self.metric_names):
        #         temp_metric = self.validation_metrics[:,ii]
        #         final_metric_value = f"(final={temp_metric[-1]:.2e})"
        #         axs.plot(
        #             epoch_ticks,
        #             temp_metric.cpu().numpy(),
        #             c=self.plot_colors[-(ii+1)],
        #             label=rf"{metric}"
        #         )
        #         axs.plot([],[],
        #             marker='',
        #             linestyle='',
        #             label=rf"{final_metric_value}"
        #         )
        #     axs.set_xlabel("epoch")
        #     axs.set_ylabel("metric")
        #     axs.set_yscale('log')
        #     plt.title("metric vs. epoch (validation)")
        #     plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        #     plt.tight_layout()
        #     plt.savefig("plots/epoch_validation_metrics.png")

        # if len(self.training_metrics) != 0 and len(self.validation_metrics) != 0:
        #     fig, axs = plt.subplots(figsize=(10,5))
        #     for ii, metric in enumerate(self.metric_names):
        #         temp_training_metric = self.training_metrics[:,ii]
        #         temp_validation_metric = self.validation_metrics[:,ii]
        #         final_training_metric_value = f"(final={temp_training_metric[-1]:.2e})"
        #         final_validation_metric_value = f"(final={temp_validation_metric[-1]:.2e})"
        #         axs.plot(
        #             epoch_ticks,
        #             temp_training_metric.cpu().numpy(),
        #             c=self.plot_colors[-(ii+1)],
        #             linestyle='-',
        #             label=rf"{metric}"
        #         )
        #         axs.plot([],[],
        #             marker='',
        #             linestyle='',
        #             label=rf"{final_training_metric_value}"
        #         )
        #         axs.plot(
        #             epoch_ticks,
        #             temp_validation_metric.cpu().numpy(),
        #             c=self.plot_colors[-(ii+1)],
        #             linestyle='--',
        #             label=rf"{metric}"
        #         )
        #         axs.plot([],[],
        #             marker='',
        #             linestyle='',
        #             label=rf"{final_validation_metric_value}"
        #         )
        #     if len(self.test_metrics) != 0:
        #         for ii, metric in enumerate(self.metric_names):
        #             temp_metric = self.test_metrics[:,ii]
        #             final_metric_value = f"(final={temp_metric[-1]:.2e})"
        #             axs.plot([],[],
        #                 marker='x',
        #                 linestyle='',
        #                 c=self.plot_colors[-(ii+1)],
        #                 label=rf"(test) {metric}"
        #             )
        #             axs.plot([],[],
        #             marker='',
        #             linestyle='',
        #             label=rf"{final_metric_value}"
        #         )
        #     axs.set_xlabel("epoch")
        #     axs.set_ylabel("metric")
        #     axs.set_yscale('log')
        #     plt.title("metric vs. epoch")
        #     plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        #     plt.tight_layout()
        #     plt.savefig("plots/epoch_metrics.png")

    def evaluate_inference(self):
        pass