"""
BlipSegmentation implementation using MinkowskiEngine
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import MinkowskiEngine as ME
from collections import OrderedDict

from blip.models import GenericModel
#import common dict, lists, and functions from common.py
from blip.models.common import Identity, activations, sparse_activations  


#sparse tensor: tensor with a lot of zeros storaged in a good way?
#ME is a library with the functions for sparse tensors 
#activation are the functions to be added to the NN: relu, softmax, etc
# get_activation takes as input one of those functions. If it's available
#on sparse_activations dictionary returns the value
def get_activation(
    activation: str,
):
    if activation in sparse_activations.keys():
        return sparse_activations[activation]


class SparseConv(ME.MinkowskiNetwork):
    """
    """
    #initialize the model parameteres to be used.
    def __init__(
        self,
        name,
        in_channels,
        out_channels,
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
        super(SparseConv, self).__init__(dimension)
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.dimension = dimension
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.num_of_convs = num_of_convs
        self.residual = residual
        
        #bias is the additive parameter in the conv, batch_norm requires bias false
        #otherwise, both true would slow down the training by having to cancel the bias first       
        if self.batch_norm:
            self.bias = False
        else:
            self.bias = True
        self.activation = activation
        #get common activation functions, ex get_activation(Relu)
        self.activation_fn = get_activation(self.activation)  
        #calls the model function
        self.construct_model()

    def construct_model(self):
        """
        Create model dictionary
        """
        _conv_dict = OrderedDict()
        _dropout_dict = OrderedDict()
        _residual_dict = OrderedDict()
        
        #linear layer to resize the output? 
        #if input != than output then there is residual? **ask nick 
        if self.in_channels != self.out_channels:
            _residual_dict['residual'] = ME.MinkowskiLinear(
                self.in_channels, self.out_channels, bias=self.bias
            )
        #identity mapping if same size    
        else:
            _residual_dict['residual'] = Identity()

        # create conv layer
          #make a variable in_channels initially equal to the inchannels,
        # update it later so the next conv gets as input the output size
        in_channels = self.in_channels
        #num_of_conv is 2 by default, iterate over it to fill the conv dictionary
        for i in range(1, self.num_of_convs):
            _conv_dict[f'{self.name}_conv[i]'] = ME.MinkowskiConvolution(
                in_channels=in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                bias=self.bias,
                dimension=self.dimension
            )
            #update size of input to the prev output
            in_channels = self.out_channels
            
        #batch norm and dropout are regularization methods, help to improve the trainning
        #and avoid overfit 
        
            #if batch_norm is set True, add the normalization layer with num_features = outchannels? 
            #normalize each parameter of the conv kernel, channels, stride, dilatation, etc. 
            # ** ask nick
            if self.batch_norm:
                _conv_dict[f'{self.name}_batch_norm_{i}'] = ME.MinkowskiBatchNorm(self.out_channels)
 
        #Dropout simulate having different network architectures by randomly dropping out nodes
        #some output layers are randomly dropped out
        #dropout rate 1.0 no dropout, 0.0 means no outputs.
        #if there are outputs (>0.0), then do the MinkowskiDropout
        # ask nick: what is it taking as output here
        if self.dropout > 0.0:
            _dropout_dict['dropout'] = ME.MinkowskiDropout(p=self.dropout)

        
        #make the regular orderedict a moduleDict
        #moduleDict properly register the modules   
        self.conv_dict = nn.ModuleDict(_conv_dict)
        self.dropout_dict = nn.ModuleDict(_dropout_dict)
        self.residual_dict = nn.ModuleDict(_residual_dict)
    
   
   #subclasses of ME.Minkowski need to go to the forward, residual, conv and drop
    def forward(
        self,
        x
    ):
        """
        Iterate over the module dictionary.
        """
        if self.residual:
            identity = self.residual_dict['residual'](x)
        #pass all    the layers of the conv to the forward 
        for layer in self.conv_dict.keys():
            x = self.conv_dict[layer](x)
        if self.residual:
            x = x + identity
        x = self.activation_fn(x)
        if self.dropout:
            x = self.dropout_dict['dropout'](x)
        return x


"""
Here are a set of standard UNet parameters, which must be
adjusted by the user for each application
"""
blip_segmentation_params = {
    'in_channels':  1,
    'out_channels': 1,  # this is the number of classes for the semantic segmentation
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
    #initialize parameters of the class
    def __init__(
        self,
        name:   str = 'blip_segmentation',      # name of the model
        config: dict = blip_segmentation_params,
        meta:   dict = {}
    ):
        super(BlipSegmentation, self).__init__(name, config, meta)
        self.name = name
        self.config = config
        # check config
        #check that all the parameters were specified *ask nick  how can it be different?
        self.logger.info(f"checking BlipSegmentation architecture using config: {self.config}")
        for item in blip_segmentation_params.keys():
            if item not in self.config:
                self.logger.error(f"parameter {item} was not specified in config file {self.config}")
                raise AttributeError
        #check the dim of conv transp and max pooling match, if not, raise an error    
        if (
            (self.config["sparse_conv_params"]["dimension"] != self.config["conv_transpose_params"]["dimension"]) or
            (self.config["sparse_conv_params"]["dimension"] != self.config["max_pooling_params"]["dimension"])
        ):
            self.logger.error(
                "dimensions for 'sparse_conv_params', 'conv_transpose_params' and" +
                f"'max_pooling_params' (with values {self.config['sparse_conv_params']['dimension']}" +
                f", {self.config['conv_transpose_params']['dimension']} and " +
                f"{self.config['max_pooling_params']['dimension']}) do not match!"
            )
            raise AttributeError

        # construct the model
        self.construct_model()
        self.save_model(flag='init')

    def construct_model(self):
        """
        """
        _down_dict = OrderedDict()
        _up_dict = OrderedDict()
        _bottleneck_dict = OrderedDict()
        _classification_dict = OrderedDict()

        # iterate over the down part
        in_channels = self.config['in_channels'] 
        for filter in self.config['filtrations']: #iterate over each layer, 
            #each downsample calls the variable conv function
            #fills a dictionary for the downsamplings
            _down_dict[f'down_filter_double_conv{filter}'] = SparseConv( 
                name=f'down_{filter}',   #set the parameters for each filter
                in_channels=in_channels,
                out_channels=filter,
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
            # set new in channel to current filter size
            in_channels = filter

        # iterate over the up part
        for filter in reversed(self.config['filtrations']): #covers the filtrations reversed order
            #upconvolution uses minkowskiconvtranspose
            _up_dict[f'up_filter_transpose{filter}'] = ME.MinkowskiConvolutionTranspose(
                in_channels=2*filter,   # adding the skip connection, so the input doubles
                out_channels=filter,
                kernel_size=self.config['conv_transpose_params']['kernel_size'],
                stride=self.config['conv_transpose_params']['stride'],
                dilation=self.config['conv_transpose_params']['dilation'],
                dimension=self.config['conv_transpose_params']['dimension']
            )
            #apply the variable convolution per layer
            _up_dict[f'up_filter_double_conv{filter}'] = SparseConv(
                name=f'up_{filter}',
                in_channels=2*filter,
                out_channels=filter,
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

        # create bottleneck layer
        _bottleneck_dict['bottleneck'] = SparseConv(
            name=f"bottleneck_{self.config['filtrations'][-1]}",
            in_channels=self.config['filtrations'][-1],
            out_channels=2*self.config['filtrations'][-1],
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

        # create output layer
        for ii, classification in enumerate(self.config['classifications']):
            _classification_dict[f"{classification}"] = ME.MinkowskiConvolution(
                in_channels=self.config['filtrations'][0],      # to match first filtration
                out_channels=self.config['out_channels'][ii],   # to the number of classes
                kernel_size=1,                                  # a one-one convolution
                dimension=self.config['sparse_conv_params']['dimension'],
            )

        # create the max pooling layer
        self.max_pooling = ME.MinkowskiMaxPooling(
            kernel_size=self.config['max_pooling_params']['kernel_size'],
            stride=self.config['max_pooling_params']['stride'],
            dilation=self.config['max_pooling_params']['dilation'],
            dimension=self.config['max_pooling_params']['dimension']
        )

        # create the dictionaries
        self.module_down_dict = nn.ModuleDict(_down_dict)
        self.module_up_dict = nn.ModuleDict(_up_dict)
        self.bottleneck_dict = nn.ModuleDict(_bottleneck_dict)
        self.classification_dict = nn.ModuleDict(_classification_dict)
        # record the info

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
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_SUM,
            device=self.device
        )
        # record the skip connections
        skip_connections = {}
        # iterate over the down part
        for filter in self.config['filtrations']:
            x = self.module_down_dict[f'down_filter_double_conv{filter}'](x)
            skip_connections[f'{filter}'] = x
            x = self.max_pooling(x)
        # through the bottleneck layer
        x = self.bottleneck_dict['bottleneck'](x)
        for filter in reversed(self.config['filtrations']):
            x = self.module_up_dict[f'up_filter_transpose{filter}'](x)
            # concatenate the skip connections
            skip_connection = skip_connections[f'{filter}']
            # check for compatibility
            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])

            concat_skip = ME.cat(skip_connection, x)
            x = self.module_up_dict[f'up_filter_double_conv{filter}'](concat_skip)

        return {
            classifications: self.classification_dict[classifications](x).features
            for classifications in self.classification_dict.keys()
        }

