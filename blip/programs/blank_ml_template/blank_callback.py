
"""
Generic model code.
"""
import torch
from torch import nn

from blip.losses import LossHandler
from blip.metrics import MetricHandler
from blip.utils.callbacks import GenericCallback

place_holder_callback_config = {
    "no_params":    "no_values"
}


class PlaceHolderCallback(GenericCallback):
    """
    """
    def __init__(
        self,
        name:               str = 'place_holder_callback',
        criterion_handler:  LossHandler = None,
        metrics_handler:    MetricHandler = None,
        meta:               dict = {}
    ):
        super(PlaceHolderCallback, self).__init__(
            name, criterion_handler, metrics_handler, meta
        )

    def reset_batch(self):
        pass

    def evaluate_epoch(
        self,
        train_type='train'
    ):
        pass

    def evaluate_training(self):
        pass

    def evaluate_testing(self):
        pass

    def evaluate_inference(self):
        pass
