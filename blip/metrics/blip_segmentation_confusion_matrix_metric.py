"""
Generic metrics for blip.
"""
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix
import seaborn as sns

from blip.utils.utils import fig_to_array
from blip.metrics.generic_metric import GenericMetric


class BlipSegmentationConfusionMatrix(GenericMetric):
    """
    Abstract base class for Blip metrics.  The inputs are
        1. name - a unique name for the metric function.
        2. meta - meta information from the module.
    """
    def __init__(
        self,
        name:           str = 'blip_segmentation_confusion_matrix_metric',
        meta:           dict = {}
    ):
        self.name = name
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        self.num_classes = {
            'topology': 3,
            'physics': 9
        }
        self.labels = {
            'topology': ['Track', 'Shower', 'Blip'],
            'physics': [
                'MIP',
                'HIP',
                'Electron Ionization',
                'Delta Electron',
                'Michel Electron',
                'Gamma Compton',
                'Gamma Conversion',
                'Nuclear Recoil',
                'Electron Recoil',
            ]
        }

        # construct batch metric dictionaries
        self.confusion_matrix = {
            key: MulticlassConfusionMatrix(
                num_classes=self.num_classes[key]
            ).to(self.device)
            for key in self.labels
        }

    def reset_batch(self):
        for key in self.confusion_matrix.keys():
            self.confusion_matrix[key] = MulticlassConfusionMatrix(
                num_classes=self.num_classes[key]
            ).to(self.device)

    def set_device(
        self,
        device
    ):
        self.device = device
        for key in self.confusion_matrix.keys():
            self.confusion_matrix[key].to(self.device)

    def report_tensorboard(
        self,
        iterations,
        train_type
    ):
        for ii, output in enumerate(self.confusion_matrix.keys()):
            fig, axs = plt.subplots(figsize=(10, 10))
            axs.set_title(f"{output.capitalize()} ({train_type.capitalize()})")
            self.confusion_matrix[output].plot(ax=axs, labels=self.labels[output])
            axs.set_xticklabels(axs.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            fig_array = fig_to_array(fig)
            self.meta['tensorboard'].add_image(
                f'{self.name}: {output} ({train_type})',
                fig_array,
                iterations,
                dataformats='HWC'
            )
            plt.close()

            """Same plot with percentages"""
            conf_matrix = self.confusion_matrix[output].compute()
            conf_matrix_percentages = conf_matrix / conf_matrix.sum(axis=1, keepdims=True) * 100
            fig, axs = plt.subplots()
            sns.heatmap(
                conf_matrix_percentages.cpu(),
                annot=True,
                fmt='.2f',
                cmap='Blues',
                ax=axs,
                cbar_kws={'format': '%.0f%%'}
            )

            axs.set_xlabel('Predicted class')
            axs.set_ylabel('True class')
            axs.set_title(f"{output.capitalize()} ({train_type.capitalize()})")
            axs.set_xticklabels(self.labels[output], rotation=45, ha='right', rotation_mode='anchor')
            axs.set_yticklabels(self.labels[output], rotation=45, ha='right', rotation_mode='anchor')
            fig_array = fig_to_array(fig)
            self.meta['tensorboard'].add_image(
                f'{self.name}: {output} ({train_type}) (percentages)',
                fig_array,
                iterations,
                dataformats='HWC'
            )
            plt.close()

    def update(
        self,
        data
    ):
        for ii, output in enumerate(self.confusion_matrix.keys()):
            self.confusion_matrix[output].update(
                nn.functional.softmax(data['outputs'][output].to(self.device), dim=1, dtype=torch.float),
                data['labels'].squeeze(0)[:, ii].long().to(self.device)
            )

    def compute(
        self,
    ):
        return {
            output: self.confusion_matrix[output].compute()
            for output in self.confusion_matrix.keys()
        }
