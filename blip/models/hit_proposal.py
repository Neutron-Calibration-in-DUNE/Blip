"""
HitProposal implementation using MinkowskiEngine
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import MinkowskiEngine as ME
from collections import OrderedDict

from blip.models import GenericModel
from blip.models.common import sparse_activations
from blip.models import PointProposalNetwork
from Blip.blip.models.arxiv.sparse_uresnet import DoubleConv


def get_activation(
    activation: str,
):
    if activation in sparse_activations.keys():
        return sparse_activations[activation]


"""
Here are a set of standard UNet parameters, which must be
adjusted by the user for each application
"""
hit_proposal_params = {
    'uresnet_in_channels':  1,
    'uresnet_out_channels': 1,  # this is the number of classes for the semantic segmentation
    'uresnet_filtrations':  [64, 128, 256, 512],    # the number of filters in each downsample
    'uresnet_double_conv_params': {
        'kernel':       3,
        'stride':       1,
        'dilation':     1,
        'activation':   'relu',
        'dimension':    3,
        'batch_norm':   True,
    },
    'uresnet_conv_transpose_params': {
        'kernel':    2,
        'stride':    2,
        'dilation':  1,
        'dimension': 3,
    },
    'uresnet_max_pooling_params': {
        'kernel':   2,
        'stride':   2,
        'dilation': 1,
        'dimension': 3,
    },
    # point proposal part
    'shared_convolutions':  [256, 512],
    'shared_kernel_size':   3,
    'shared_stride':        1,
    'shared_dilation':      1,
    'shared_activation':     'relu',
    'shared_batch_norm':    True,
    'classification_anchors':       10,
    'classification_size':          1,
    'classification_kernel_size':   1,
    'regression_anchors':       10,
    'regression_size':          7,
    'regression_kernel_size':   1,
}


class HitProposalNetwork(GenericModel):
    """
    """
    def __init__(
        self,
        name:   str = 'my_unet',
        config: dict = hit_proposal_params,
        meta:   dict = {}
    ):
        super(HitProposalNetwork, self).__init__(name, config, meta)
        # check config
        self.logger.info(f"checking HitProposalNetwork architecture using config: {self.config}")
        for item in hit_proposal_params.keys():
            if item not in self.config:
                self.logger.error(f"parameter {item} was not specified in config file {self.config}")
                raise AttributeError
        if ((self.config["uresnet_double_conv_params"]["dimension"] !=
             self.config["uresnet_conv_transpose_params"]["dimension"]) or
            (self.config["uresnet_double_conv_params"]["dimension"] !=
             self.config["uresnet_max_pooling_params"]["dimension"])):
            self.logger.error(
                "dimensions for 'uresnet_double_conv_params', 'uresnet_conv_transpose_params' and" +
                f"'uresnet_max_pooling_params' (with values {self.config['uresnet_double_conv_params']['dimension']}" +
                f", {self.config['uresnet_conv_transpose_params']['dimension']} and " +
                f"{self.config['uresnet_max_pooling_params']['dimension']}) do not match!"
            )
            raise AttributeError

        # construct the model
        self.construct_model()
        self.register_forward_hooks()
        self.save_model(flag='init')

    def construct_model(self):
        """
        """
        self.logger.info(f"Attempting to build UNet architecture using config: {self.config}")

        _down_dict = OrderedDict()
        _up_dict = OrderedDict()
        _classification_dict = OrderedDict()

        # iterate over the down part
        in_channels = self.config['uresnet_in_channels']
        for filter in self.config['uresnet_filtrations']:
            _down_dict[f'down_filter_double_conv{filter}'] = DoubleConv(
                name=f'down_{filter}',
                in_channels=in_channels,
                out_channels=filter,
                kernel_size=self.config['uresnet_double_conv_params']['kernel_size'],
                stride=self.config['uresnet_double_conv_params']['stride'],
                dilation=self.config['uresnet_double_conv_params']['dilation'],
                dimension=self.config['uresnet_double_conv_params']['dimension'],
                activation=self.config['uresnet_double_conv_params']['activation'],
                batch_norm=self.config['uresnet_double_conv_params']['batch_norm'],
            )
            # set new in channel to current filter size
            in_channels = filter

        # iterate over the up part
        for filter in reversed(self.config['uresnet_filtrations']):
            _up_dict[f'up_filter_transpose{filter}'] = ME.MinkowskiConvolutionTranspose(
                in_channels=2*filter,   # adding the skip connection, so the input doubles
                out_channels=filter,
                kernel_size=self.config['uresnet_conv_transpose_params']['kernel_size'],
                stride=self.config['uresnet_conv_transpose_params']['stride'],
                dilation=self.config['uresnet_conv_transpose_params']['dilation'],
                dimension=self.config['uresnet_conv_transpose_params']['dimension']
            )
            _up_dict[f'up_filter_double_conv{filter}'] = DoubleConv(
                name=f'up_{filter}',
                in_channels=2*filter,
                out_channels=filter,
                kernel_size=self.config['uresnet_double_conv_params']['kernel_size'],
                stride=self.config['uresnet_double_conv_params']['stride'],
                dilation=self.config['uresnet_double_conv_params']['dilation'],
                dimension=self.config['uresnet_double_conv_params']['dimension'],
                activation=self.config['uresnet_double_conv_params']['activation'],
                batch_norm=self.config['uresnet_double_conv_params']['batch_norm'],
            )

        # create bottleneck layer
        self.bottleneck = DoubleConv(
            name=f"bottleneck_{self.config['uresnet_filtrations'][-1]}",
            in_channels=self.config['uresnet_filtrations'][-1],
            out_channels=2*self.config['uresnet_filtrations'][-1],
            kernel_size=self.config['uresnet_double_conv_params']['kernel_size'],
            stride=self.config['uresnet_double_conv_params']['stride'],
            dilation=self.config['uresnet_double_conv_params']['dilation'],
            dimension=self.config['uresnet_double_conv_params']['dimension'],
            activation=self.config['uresnet_double_conv_params']['activation'],
            batch_norm=self.config['uresnet_double_conv_params']['batch_norm'],
        )

        # create output layer
        for ii, classification in enumerate(self.config['classifications']):
            _classification_dict[f"{classification}"] = ME.MinkowskiConvolution(
                in_channels=self.config['uresnet_filtrations'][0],      # to match first filtration
                out_channels=self.config['uresnet_out_channels'][ii],   # to the number of classes
                kernel_size=1,                                  # a one-one convolution
                dimension=self.config['uresnet_double_conv_params']['dimension'],
            )

        # create the max pooling layer
        self.max_pooling = ME.MinkowskiMaxPooling(
            kernel_size=self.config['uresnet_max_pooling_params']['kernel_size'],
            stride=self.config['uresnet_max_pooling_params']['stride'],
            dilation=self.config['uresnet_max_pooling_params']['dilation'],
            dimension=self.config['uresnet_max_pooling_params']['dimension']
        )

        # create the dictionaries
        self.module_down_dict = nn.ModuleDict(_down_dict)
        self.module_up_dict = nn.ModuleDict(_up_dict)
        self.classification_dict = nn.ModuleDict(_classification_dict)
        self.config['ppn_in_channels'] = self.config['uresnet_filtrations'][0]

        # create the point proposal network
        self.point_proposal = PointProposalNetwork(f"{self.name}_ppn", self.config, self.meta)

    def forward(
        self,
        data
    ):
        """
        Convert input, which should be a tuple a Data
        object to a ME.SparseTensor(feats, coords).
        Iterate over the module dictionary.
        """
        x = ME.SparseTensor(
            features=data.x,
            coordinates=torch.cat(
                (data.batch.unsqueeze(1), data.pos),
                dim=1
            ).int(),
            device=self.device
        )
        # record the skip connections
        skip_connections = {}
        # iterate over the down part
        for filter in self.config['uresnet_filtrations']:
            x = self.module_down_dict[f'down_filter_double_conv{filter}'](x)
            skip_connections[f'{filter}'] = x
            x = self.max_pooling(x)
        # through the bottleneck layer
        x = self.bottleneck(x)
        for filter in reversed(self.config['uresnet_filtrations']):
            x = self.module_up_dict[f'up_filter_transpose{filter}'](x)
            # concatenate the skip connections
            skip_connection = skip_connections[f'{filter}']
            # check for compatibility
            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])

            concat_skip = ME.cat(skip_connection, x)
            x = self.module_up_dict[f'up_filter_double_conv{filter}'](concat_skip)
        outputs = {
            classifications: self.classification_dict[classifications](x).features
            for classifications in self.classification_dict.keys()
        }
        point_proposal = self.point_proposal(x)
        for key, val in point_proposal.items():
            outputs[key] = val
        return outputs
