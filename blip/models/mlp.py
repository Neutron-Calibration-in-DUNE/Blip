"""
Custom MLP using torch
"""
from collections import OrderedDict
import torch.nn as nn

from blip.models.common import activations
from blip.models import GenericModel

mlp_config = {
    'layers':   [5, 10, 25, 10],
    'dropout':  0.,
    'activation':   'leaky_relu',
    'activation_params':    {
        'negative_slope': 0.01
    }
}


class MLP(GenericModel):
    def __init__(
        self,
        name:   str = 'mlp',
        config: dict = mlp_config,
        meta:   dict = {}
    ):
        super(MLP, self).__init__(
            name, config, meta
        )
        self.config = config

        # construct the model
        self.construct_model()
        # register hooks
        self.register_forward_hooks()

    def construct_model(self):
        self.logger.info(f"Attempting to build MLP architecture using config: {self.config}")
        _mlp_dict = OrderedDict()

        if "layers" not in self.config.keys():
            self.logger.error('layers not specified for MLP config!')
        if not isinstance(self.config['layers'], list):
            self.logger.error(f'layers not specified as a list in config! got {self.config["layers"]}!')
        if "dropout" not in self.config.keys():
            self.logger.warn('dropout not specified in config! setting to 0.')
            self.config['dropout'] = None
        else:
            if isinstance(self.config['dropout'], float):
                dropout = self.config['dropout']
                self.config['dropout'] = [dropout for ii in range(len(self.config['layers']))]
            elif isinstance(self.config['dropout'], list):
                if (
                    len(self.config['dropout']) != len(self.config['layers']) or
                    len(self.config['dropout']) != len(self.config['layers'] - 1)
                ):
                    self.logger.error(
                        f'specified dropout values {self.config["dropout"]} not equal to the number of layers, ' +
                        'or the number of layers - 1!'
                    )
            else:
                self.logger.error(f'dropout should be a float, or a list of floats! got {self.config["dropout"]}')
        if "activation" not in self.config.keys():
            self.logger.warn('activation not specified in config! setting to "leaky_relu".')
            self.config['activation'] = activations['leaky_relu']
        else:
            if self.config['activation'] in activations.keys():
                if 'activation_params' in self.config.keys():
                    try:
                        self.activation = activations[self.config['activation']](**self.config['activation_params'])
                    except Exception as exception:
                        self.logger.error(
                            f'setting activation in MLP failed with inputs {self.config["activation"]}, ' +
                            f'{self.config["activation_params"]}! exception: {exception}'
                        )
                else:
                    try:
                        self.activation = activations[self.config['activation']]()
                    except Exception as exception:
                        self.logger.error(
                            f'setting activation in MLP failed with inputs {self.config["activation"]}!' +
                            f' exception: {exception}'
                        )

        layers = self.config['layers']
        dropout = self.config['dropout']
        input_channels = layers[0]

        for ii in range(len(layers - 1)):
            _mlp_dict[f'layer_{ii}'] = nn.Linear(input_channels, layers[ii+1])
            if 'activation_params' in self.config.keys():
                _mlp_dict[f'layer_{ii}_activation'] = self.activation(**self.config['activation_params'])
            else:
                _mlp_dict[f'layer_{ii}_activation'] = self.activation()
            if self.config['dropout'] is not None:
                _mlp_dict[f'layer_{ii}_dropout'] = nn.Dropout(p=self.config['dropout'][ii])
            input_channels = layers[ii+1]

        self.mlp_dict = nn.ModuleDict(_mlp_dict)

    def forward(
        self,
        data
    ):
        x = data.x
        for ii, layer in enumerate(self.mlp_dict.keys()):
            x = self.mlp_dict[layer](x)
        return x
