
"""
Generic model code.
"""
from torch import nn
from collections import OrderedDict

from blip.models import GenericModel

generic_config = {
    "no_params":    "no_values"
}


class FeaturePropagation(GenericModel):
    """
    Wrapper of torch nn.Module that generates a FeaturePropagation
    """
    def __init__(
        self,
        name:   str,
        config: dict = generic_config,
        meta:   dict = {}
    ):
        super(FeaturePropagation, self).__init__(name, config, meta)

    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.

        """
        self.logger.info(f"Attempting to build FeaturePropagation architecture using config: {self.config}")

        _model_dict = OrderedDict()
        self.model_dict = nn.ModuleDict(_model_dict)

    def forward(self, x):
        pass
