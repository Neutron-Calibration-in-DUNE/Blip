
"""
Generic model code.
"""
import torch
from torch import nn
from collections import OrderedDict

from blip.models import GenericModel

place_holder_model_config = {
    "no_params":    "no_values"
}


class PlaceHolderModel(GenericModel):
    """
    """
    def __init__(
        self,
        name:   str = 'place_holder_model',
        config: dict = place_holder_model_config,
        meta:   dict = {}
    ):
        super(PlaceHolderModel, self).__init__(
            name, config, meta
        )

        # construct the model
        self.construct_model()
        # register hooks
        self.register_forward_hooks()

    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.

        """
        self.logger.info(f"Attempting to build PlaceHolder architecture using config: {self.config}")

        _model_dict = OrderedDict()

        self.model_dict = nn.ModuleDict(_model_dict)

        # record the info
        self.logger.info(
            f"Constructed PlaceHolder with dictionaries: {self.model_dict}"
        )

    def forward(
        self,
        data
    ):
        for ii, layer in enumerate(self.model_dict.keys()):
            data = self.model_dict[layer](data)
        return data
