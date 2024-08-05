"""
BlipSegmentation implementation using MinkowskiEngine
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import MinkowskiEngine as ME
from collections import OrderedDict

from blip.utils.logger import BlipError
from blip.models.generic_model import GenericModel
from blip.models.common import Identity, activations, sparse_activations


def get_activation(
    activation: str,
):
    if activation in sparse_activations.keys():
        return sparse_activations[activation]


class SegmentationBlock(ME.MinkowskiNetwork):
    """
    The main block structure of the BlipSegmentation model.
    A block consists of a series of convolutions structured according
    to a ResNext architecture which has the following form:
    
    """
    def __init__(
        self,
        name,
        in_channels,
        out_channels,
        cardinality:    int = 1,
        kernel_size:    int = 3,
        stride:         int = 1,
        dilation:       int = 1,
        activation:     str = 'relu',
        batch_norm:     bool = True,
        dimension:      int = 3,
        num_of_convs:   int = 2,
        dropout:        float = 0.0,
        residual:       bool = True,
    ):
        """
        """
        super(SegmentationBlock, self).__init__(dimension)
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cardinality = cardinality
        self.dimension = dimension
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.num_of_convs = num_of_convs
        self.residual = residual

        """Check input parameters"""
        if dimension <= 0:
            raise BlipError(
                f'dimension given to SegmentationBlock is {dimension} but should be > 0'
            )
        if cardinality <= 0:
            raise BlipError(
                f'cardinality given to SegmentationBlock is {cardinality} but should be > 0'
            )
        if in_channels % cardinality != 0:
            raise BlipError(
                f'in_channels ({in_channels}) must be divisible by cardinality ({cardinality})'
            )
        if out_channels % cardinality != 0:
            raise BlipError(
                f'out_channels ({out_channels}) must be divisible by cardinality ({cardinality})'
            )

        self.card_in_channels = self.in_channels // cardinality
        self.card_out_channels = self.out_channels // cardinality

        """Set up lists of kernels, dilations and strides for different cardinalities"""
        self.dilation = []
        if dilation is None:
            self.dilation = [3**i for i in range(cardinality)]
        elif isinstance(dilation, int):
            self.dilation = [dilation for _ in range(cardinality)]
        elif isinstance(dilation, list):
            if len(dilation) != cardinality:
                raise BlipError(
                    f'length of dilation ({len(dilation)}) and cardinality ({cardinality}) must be equal'
                )
            self.dilation = dilation
        else:
            raise BlipError(
                f'type for dilation was {type(dilation)} but must be int or list'
            )

        self.kernel_size = []
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size for _ in range(cardinality)]
        elif isinstance(kernel_size, list):
            if len(kernel_size) != cardinality:
                raise BlipError(
                    f'length of kernel_size ({len(kernel_size)}) and cardinality ({cardinality}) must be equal'
                )
            self.kernel_size = kernel_size
        else:
            raise BlipError(
                f'type for stride was {type(kernel_size)} but must be int or list'
            )

        self.stride = []
        if isinstance(stride, int):
            self.stride = [stride for _ in range(cardinality)]
        elif isinstance(stride, list):
            if len(stride) != cardinality:
                raise BlipError(
                    f'length of stride ({len(stride)}) and cardinality ({cardinality}) must be equal'
                )
            self.stride = stride
        else:
            raise BlipError(
                f'type for stride was {type(stride)} but must be int or list'
            )

        if self.batch_norm:
            self.bias = False
        else:
            self.bias = True
        self.activation = activation
        self.activation_fn = get_activation(self.activation)
        self.construct_model()

    def construct_model(self):
        """
        Create model dictionary
        """
        _conv_dict = OrderedDict()
        _dropout_dict = OrderedDict()
        _residual_dict = OrderedDict()
        _output_dict = OrderedDict()

        """The residual connection at the output of the block"""
        if self.in_channels != self.out_channels:
            _residual_dict['residual'] = ME.MinkowskiLinear(
                self.in_channels, self.out_channels, bias=self.bias
            )
        else:
            _residual_dict['residual'] = Identity()

        """Create the ResNext layers"""
        for ii in range(self.cardinality):
            """
            Each layer consists of a linear input layer, followed
            by a number of convolutional layers with a specific kernel,
            and then another linear layer at the output.
            """
            in_channels = self.in_channels
            """Start with a linear layer"""
            conv_layers = [
                ME.MinkowskiLinear(in_channels, self.card_in_channels)
            ]
            for jj in range(self.num_of_convs):
                in_C = (self.card_in_channels if jj == 0 else self.card_out_channels)
                conv_layers.append(ME.MinkowskiConvolution(
                    in_channels=in_C,
                    out_channels=self.card_out_channels,
                    kernel_size=self.kernel_size[ii],
                    stride=self.stride[ii],
                    dilation=self.dilation[ii],
                    bias=self.bias,
                    dimension=self.dimension
                ))
                if self.batch_norm:
                    conv_layers.append(ME.MinkowskiBatchNorm(self.card_out_channels))
                conv_layers.append(self.activation_fn)
            if self.dropout > 0.0:
                conv_layers.append(ME.MinkowskiDropout(p=self.dropout))
            _conv_dict[f'{self.name}_path_{ii}'] = nn.Sequential(*conv_layers)

        """Output layer"""
        _output_dict[f'{self.name}_output_linear'] = ME.MinkowskiLinear(self.out_channels, self.out_channels)
        if self.batch_norm:
            _output_dict[f'{self.name}_output_batch_norm'] = ME.MinkowskiBatchNorm(self.out_channels)

        _output_dict[f'{self.name}_output_activation'] = self.activation_fn

        self.conv_dict = nn.ModuleDict(_conv_dict)
        self.dropout_dict = nn.ModuleDict(_dropout_dict)
        self.residual_dict = nn.ModuleDict(_residual_dict)
        self.output_dict = nn.ModuleDict(_output_dict)

    def forward(
        self,
        x
    ):
        """
        Iterate over the module dictionary.
        """
        identity = self.residual_dict['residual'](x)
        output = tuple([self.conv_dict[layer](x) for layer in self.conv_dict.keys()])
        output = ME.cat(output)
        for layer in self.output_dict.keys():
            output = self.output_dict[layer](output)
        output += identity
        return output
        for layer in self.conv_dict.keys():
            x = self.conv_dict[layer](x)
        if self.residual:
            x = x + identity
        x = self.activation_fn(x)
        if self.dropout:
            x = self.dropout_dict['dropout'](x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            bias=False,
            dimension=3
        )
        self.sigmoid = ME.MinkowskiSigmoid()

    def forward(self, x):
        # Extract features
        features = x.F

        # Compute average and max pooling along the channel dimension
        avg_out = torch.mean(features, dim=1, keepdim=True)
        max_out, _ = torch.max(features, dim=1, keepdim=True)

        # Concatenate along the channel dimension
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # Create a new sparse tensor for the concatenated features
        x_cat_sparse = ME.SparseTensor(
            features=x_cat,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )

        # Apply convolution and sigmoid
        x_cat_sparse = self.conv(x_cat_sparse)
        attention_map = self.sigmoid(x_cat_sparse)

        # Apply the attention map to the original features
        x_out = ME.SparseTensor(
            features=features * attention_map.F,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )

        return x_out


"""
Here are a set of standard UNet parameters, which must be
adjusted by the user for each application
"""
blip_segmentation_params = {
    'in_channels':  1,
    'filtrations':  [64, 128, 256, 512],    # the number of filters in each downsample
    'residual':     True,
    'sparse_conv_params': {
        'kernel_size':       3,
        'stride':       1,
        'dilation':     1,
        'activation':   'relu',
        'dimension':    3,
        'batch_norm':   True,
        'num_of_convs': 2,
        'dropout':      0.1,
        'residual':     True,
    },
    'conv_transpose_params': {
        'kernel_size':    2,
        'stride':    2,
        'dilation':  1,
        'dimension': 3,
    },
    'max_pooling_params': {
        'kernel_size':    2,
        'stride':    2,
        'dilation':  1,
        'dimension': 3,
    }
}


class BlipSegmentation(GenericModel):
    """
    """
    def __init__(
        self,
        name:   str = 'blip_segmentation',      # name of the model
        config: dict = {},
        meta:   dict = {}
    ):
        super(BlipSegmentation, self).__init__(name, config, meta)
        self.config = config
        for item in blip_segmentation_params.keys():
            if item not in self.config:
                raise BlipError(f"parameter {item} was not specified in config file {self.config}")
        if (
            (self.config["sparse_conv_params"]["dimension"] != self.config["conv_transpose_params"]["dimension"]) or
            (self.config["sparse_conv_params"]["dimension"] != self.config["max_pooling_params"]["dimension"])
        ):
            raise BlipError(
                "dimensions for 'sparse_conv_params', 'conv_transpose_params' and" +
                f"'max_pooling_params' (with values {self.config['sparse_conv_params']['dimension']}" +
                f", {self.config['conv_transpose_params']['dimension']} and " +
                f"{self.config['max_pooling_params']['dimension']}) do not match!"
            )

        # construct the model
        self.construct_model()
        self.save_model(flag='init')

    def construct_model(self):
        """
        """
        _input_dict = OrderedDict()
        _down_dict = OrderedDict()
        _pooling_dict = OrderedDict()
        _up_dict = OrderedDict()
        _bottleneck_dict = OrderedDict()
        _classification_dict = OrderedDict()
        _heat_map_dict = OrderedDict()

        """Create spatial attention module"""
        # self.spatial_attention = SpatialAttention(in_channels=2*self.config['filtrations'][-1])

        """Create input layer"""
        _input_dict['input_layer'] = ME.MinkowskiConvolution(
            self.config['in_channels'],
            self.config['filtrations'][0],
            kernel_size=self.config['input_kernel'],
            stride=1,
            dimension=self.config['sparse_conv_params']['dimension']
        )

        """Iterate over the down part"""
        in_channels = self.config['filtrations'][0]
        for filter in self.config['filtrations']:
            _down_dict[f'down_filter_double_conv{filter}'] = SegmentationBlock(
                name=f'down_{filter}',
                in_channels=in_channels,
                out_channels=filter,
                cardinality=self.config['sparse_conv_params']['cardinality'],
                kernel_size=self.config['sparse_conv_params']['kernel_size'],
                stride=self.config['sparse_conv_params']['stride'],
                dilation=self.config['sparse_conv_params']['dilation'],
                dimension=self.config['sparse_conv_params']['dimension'],
                activation=self.config['sparse_conv_params']['activation'],
                batch_norm=self.config['sparse_conv_params']['batch_norm'],
                num_of_convs=self.config['sparse_conv_params']['num_of_convs'],
                dropout=self.config['sparse_conv_params']['dropout'],
                residual=self.config['residual']
            )
            in_channels = filter
            _pooling_dict[f'down_filter_pooling{filter}'] = ME.MinkowskiConvolution(
                in_channels=filter,
                out_channels=filter,
                kernel_size=2,
                stride=2,
                dimension=self.config['sparse_conv_params']['dimension']
            )

        """Iterate over the up part"""
        for filter in reversed(self.config['filtrations']):
            _up_dict[f'up_filter_transpose{filter}'] = ME.MinkowskiConvolutionTranspose(
                in_channels=2*filter,   # adding the skip connection, so the input doubles
                out_channels=filter,
                kernel_size=self.config['conv_transpose_params']['kernel_size'],
                stride=self.config['conv_transpose_params']['stride'],
                dilation=self.config['conv_transpose_params']['dilation'],
                dimension=self.config['conv_transpose_params']['dimension']
            )
            _up_dict[f'up_filter_double_conv{filter}'] = SegmentationBlock(
                name=f'up_{filter}',
                in_channels=2*filter,
                out_channels=filter,
                cardinality=self.config['sparse_conv_params']['cardinality'],
                kernel_size=self.config['sparse_conv_params']['kernel_size'],
                stride=self.config['sparse_conv_params']['stride'],
                dilation=self.config['sparse_conv_params']['dilation'],
                dimension=self.config['sparse_conv_params']['dimension'],
                activation=self.config['sparse_conv_params']['activation'],
                batch_norm=self.config['sparse_conv_params']['batch_norm'],
                num_of_convs=self.config['sparse_conv_params']['num_of_convs'],
                dropout=self.config['sparse_conv_params']['dropout'],
                residual=self.config['residual']
            )

        """Create bottleneck layer"""
        _bottleneck_dict['bottleneck'] = SegmentationBlock(
            name=f"bottleneck_{self.config['filtrations'][-1]}",
            in_channels=self.config['filtrations'][-1],
            out_channels=2*self.config['filtrations'][-1],
            cardinality=self.config['sparse_conv_params']['cardinality'],
            kernel_size=self.config['sparse_conv_params']['kernel_size'],
            stride=self.config['sparse_conv_params']['stride'],
            dilation=self.config['sparse_conv_params']['dilation'],
            dimension=self.config['sparse_conv_params']['dimension'],
            activation=self.config['sparse_conv_params']['activation'],
            batch_norm=self.config['sparse_conv_params']['batch_norm'],
            num_of_convs=self.config['sparse_conv_params']['num_of_convs'],
            dropout=self.config['sparse_conv_params']['dropout'],
            residual=self.config['residual']
        )

        """Create output layer for classifications"""
        for ii, classification in enumerate(self.config['classifications']):
            _classification_dict[f"{classification}"] = ME.MinkowskiConvolution(
                in_channels=self.config['filtrations'][0],      # to match first filtration
                out_channels=self.config['out_channels'][ii],   # to the number of classes
                kernel_size=1,                                  # a one-one convolution
                dimension=self.config['sparse_conv_params']['dimension'],
            )

        """Create output layer for heat maps"""
        for ii, heat_map in enumerate(self.config["heat_maps"]):
            _heat_map_dict[f"{heat_map}"] = ME.MinkowskiConvolution(
                in_channels=self.config['filtrations'][0],
                out_channels=1,
                kernel_size=1,
                dimension=self.config['sparse_conv_params']['dimension'],
            )

        """Create the max pooling layer"""
        self.max_pooling = ME.MinkowskiMaxPooling(
            kernel_size=self.config['max_pooling_params']['kernel_size'],
            stride=self.config['max_pooling_params']['stride'],
            dilation=self.config['max_pooling_params']['dilation'],
            dimension=self.config['max_pooling_params']['dimension']
        )

        """Create the dictionaries"""
        self.input_dict = nn.ModuleDict(_input_dict)
        self.module_down_dict = nn.ModuleDict(_down_dict)
        self.pooling_dict = nn.ModuleDict(_pooling_dict)
        self.module_up_dict = nn.ModuleDict(_up_dict)
        self.bottleneck_dict = nn.ModuleDict(_bottleneck_dict)
        self.classification_dict = nn.ModuleDict(_classification_dict)
        self.heat_map_dict = nn.ModuleDict(_heat_map_dict)

    def forward(
        self,
        data
    ):
        """
        Convert input, which should be a tuple a Data
        object to a ME.SparseTensor(feats, coords).
        Iterate over the module dictionary.
        """
        features = data['features'].squeeze(0).to(self.device)
        batch_ids = data['batch_id'].squeeze(0).to(self.device)
        positions = data['positions'].squeeze(0).to(self.device)
        coordinates = torch.cat(
            (batch_ids, positions),
            dim=1
        ).to(self.device)

        """Step 1: Find unique rows"""
        unique_coordinates, inverse_indices = torch.unique(
            coordinates,
            dim=0,
            return_inverse=True,
            sorted=True
        )

        """Step 2: Generate sparse tensor"""
        x = ME.SparseTensor(
            features=features.float(),
            coordinates=coordinates,
            quantization_mode=self.meta['quantization_mode'],
            minkowski_algorithm=self.meta['minkowski_algorithm'],
        )

        """Record the skip connections"""
        skip_connections = {}

        """Step 3: Iterate over initial layer"""
        x = self.input_dict['input_layer'](x)

        """Step 3: Iterate over the downward part"""
        for filter in self.config['filtrations']:
            x = self.module_down_dict[f'down_filter_double_conv{filter}'](x)
            skip_connections[f'{filter}'] = x
            x = self.pooling_dict[f'down_filter_pooling{filter}'](x)

        """Step 4: Step through the bottleneck layer"""
        x = self.bottleneck_dict['bottleneck'](x)

        """Step 5: Pass through spatial attention layer"""
        # x = self.spatial_attention(x)

        """Step 5: Iterate over the upward part"""
        for filter in reversed(self.config['filtrations']):
            x = self.module_up_dict[f'up_filter_transpose{filter}'](x)
            skip_connection = skip_connections[f'{filter}']
            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])
            concat_skip = ME.cat(skip_connection, x)
            x = self.module_up_dict[f'up_filter_double_conv{filter}'](concat_skip)

        """Step 6: Evaluate the classification and heat_map output"""
        outputs = {
            classifications: self.classification_dict[classifications](x).features[inverse_indices]
            for classifications in self.classification_dict.keys()
        }
        for heat_map in self.heat_map_dict.keys():
            outputs[heat_map] = self.heat_map_dict[heat_map](x).features[inverse_indices]
        return outputs
