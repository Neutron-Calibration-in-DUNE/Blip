"""
Generic metrics for blip.
"""
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import seaborn as sns

from blip.utils.utils import fig_to_array
from blip.metrics.generic_metric import GenericMetric


class BlipSegmentationKeyPoint(GenericMetric):
    """
    """
    def __init__(
        self,
        name:           str = 'blip_segmentation_key_point_metric',
        meta:           dict = {}
    ):
        self.name = name
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']

        self.labels = ['vertex' ]

        # construct batch metric dictionaries
        self.key_point = {
            key: None
            for key in self.labels
        }

    def reset_batch(self):
        for key in self.key_point.keys():
            self.key_point[key] = None

    def set_device(
        self,
        device
    ):
        self.device = device
        for key in self.key_point.keys():
            self.key_point[key].to(self.device)

    def report_tensorboard(
        self,
        iterations,
        train_type
    ):
        for ii, output in enumerate(self.key_point.keys()):
            pass
            # fig, axs = plt.subplots(figsize=(10, 10))
            # axs.set_title(f"{output.capitalize()} ({train_type.capitalize()})")
            # self.key_point[output].plot(ax=axs, labels=self.labels[output])
            # axs.set_xticklabels(axs.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            # fig_array = fig_to_array(fig)
            # self.meta['tensorboard'].add_image(
            #     f'{self.name}: {output} ({train_type})',
            #     fig_array,
            #     iterations,
            #     dataformats='HWC'
            # )
            # plt.close()

            # """Same plot with percentages"""
            # conf_matrix = self.key_point[output].compute()
            # conf_matrix_percentages = conf_matrix / conf_matrix.sum(axis=1, keepdims=True) * 100
            # fig, axs = plt.subplots()
            # sns.heatmap(
            #     conf_matrix_percentages.cpu(),
            #     annot=True,
            #     fmt='.2f',
            #     cmap='Blues',
            #     ax=axs,
            #     cbar_kws={'format': '%.0f%%'}
            # )

            # axs.set_xlabel('Predicted class')
            # axs.set_ylabel('True class')
            # axs.set_title(f"{output.capitalize()} ({train_type.capitalize()})")
            # axs.set_xticklabels(self.labels[output], rotation=45, ha='right', rotation_mode='anchor')
            # axs.set_yticklabels(self.labels[output], rotation=45, ha='right', rotation_mode='anchor')
            # fig_array = fig_to_array(fig)
            # self.meta['tensorboard'].add_image(
            #     f'{self.name}: {output} ({train_type}) (percentages)',
            #     fig_array,
            #     iterations,
            #     dataformats='HWC'
            # )
            # plt.close()
            
    def find_contours(
        self,
        data
    ):
        pass
    
    def non_max_suppression(
        self,
        data
    ):
        pass

    def update(
        self,
        data
    ):
        for ii, output in enumerate(self.key_point.keys()):
            pass

    def compute(
        self,
    ):
        return {
            output: None
            for output in self.key_point.keys()
        }
