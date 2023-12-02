"""
Implementation of the BlipGraph model using pytorch.

BlipGraph consists of the following architecture:


"""
import torch
import torch.nn as nn
from collections import OrderedDict
import torch_geometric.transforms as T
from torch.nn import Linear
from torch_geometric.nn import (
    MLP, 
    DynamicEdgeConv, 
    PointNetConv, 
    PointTransformerConv
)
from torch_geometric.nn import (
    global_add_pool, 
    global_mean_pool, 
    global_max_pool
)
from torch_geometric.nn import fps, radius

from blip.models import GenericModel

blip_graph_config = {
    "input_dimension":      3,
    "classifications":      ["source", "shape", "particle"],
    'augmentations':    {
        'jitter':   0.03,
        'flip':     1.0,
        'shear':    0.2,
        'number_of_augmentations':  2
    },
    'embedding': {
        'embedding_1': {
            "embedding_type":       "dynamic_edge_conv",
            "number_of_neighbors":  5,
            "aggregation":          "max",
            "embedding_mlp_layers": [5, 10, 25, 10],
            "dropout":              0.01,
            "activation":           "LeakyReLU"
        },
        'embedding_2': {
            "embedding_type":       "point_net_conv",
            "local_mlp_layers":     [5, 10, 25, 10],
            "dropout":              0.01,
            "activation":           "LeakyReLU",
            "fps_ratio":            0.5,
            "cluster_radius":       0.25,
            "max_number_neighbors": 5,
            "add_self_loops":       True,
        },
        'embedding_3': {
            "embedding_type":       'point_transformer_conv',
            "pos_nn_layers":        [10, 25, 10],
            "pos_nn_dropout":       0.01,
            "pos_nn_activation":    "LeakyReLU",
            "attn_nn_layers":       [10, 25, 10],
            "attn_nn_dropout":      0.01,
            "attn_nn_activation":   "LeakyReLU",
            "fps_ratio":            0.25,
            "cluster_radius":       0.25,
            "max_number_neighbors": 5,
            "add_self_loops":       True,
        }
    },
    'reduction': {
        'linear_output':        128,
        'reduction_type':       'max_pool',
        'projection_head_layers':   [128, 256, 128],
        'projection_head_dropout':  0.1,
        'projection_head_activation':   'LeakyReLU',
        'projection_head_activation_params': {
            'negative_slope':   0.1
        }
    },
    'classification': {
        'mlp_output_layers':    [128, 256, 32],
        'out_channels':         [8, 7, 32],
    },
}


# PointNetConv Modules
class PointNetModule(torch.nn.Module):
    """
    Methodology for using PointNetConv is to not reduce
    positions into cluster centers, but rather return the
    full set of positions after each layer.
    """
    def __init__(
        self,
        local_mlp:      MLP,
        fps_ratio:      float = 0.5,
        cluster_radius: float = 0.25,
        max_number_neighbors:   int = 10,
        add_self_loops:     bool = False
    ):
        super().__init__()
        self.local_mlp = local_mlp
        self.fps_ratio = fps_ratio
        self.cluster_radius = cluster_radius
        self.max_number_neighbors = max_number_neighbors
        self.add_self_loops = add_self_loops
        self.point_net_conv = PointNetConv(
            local_nn=self.local_mlp,
            add_self_loops=self.add_self_loops
        )

    def forward(self, pos, batch):
        # find indices with furthest point sampling
        idx = fps(
            pos, batch,
            ratio=self.fps_ratio
        )
        # gather clusters
        row, col = radius(
            pos, pos[idx],
            self.cluster_radius,
            batch, batch[idx],
            max_num_neighbors=self.max_number_neighbors
        )
        # determine edges for convolution
        edge_index = torch.stack([col, row], dim=0)
        output = self.point_net_conv(
            None,
            pos,
            edge_index
        )
        return output


# PointTransformerConv Modules
class PointTransformerModule(torch.nn.Module):
    """
    Methodology for using PointTransformerConv is to not reduce
    positions into cluster centers, but rather return the
    full set of positions after each layer.
    """
    def __init__(
        self,
        in_channels:    int,
        out_channels:   int,
        pos_nn:         MLP,
        attn_nn:        MLP,
        fps_ratio:      float = 0.5,
        cluster_radius: float = 0.25,
        max_number_neighbors:   int = 10,
        add_self_loops:     bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pos_nn = pos_nn
        self.attn_nn = attn_nn
        self.fps_ratio = fps_ratio
        self.cluster_radius = cluster_radius
        self.max_number_neighbors = max_number_neighbors
        self.add_self_loops = add_self_loops
        self.point_transformer_conv = PointTransformerConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            pos_nn=self.pos_nn,
            attn_nn=self.attn_nn,
            add_self_loops=self.add_self_loops
        )

    def forward(self, pos, batch):
        # find indices with furthest point sampling
        idx = fps(
            pos, batch,
            ratio=self.fps_ratio
        )
        # gather clusters
        row, col = radius(
            pos, pos[idx],
            self.cluster_radius,
            batch, batch[idx],
            max_num_neighbors=self.max_number_neighbors
        )
        # determine edges for convolution
        edge_index = torch.stack([col, row], dim=0)
        output = self.point_transformer_conv(
            pos,
            pos,
            edge_index
        )
        return output


class BlipGraph(GenericModel):
    """
    """
    def __init__(
        self,
        name:   str = 'blip_graph',
        config: dict = blip_graph_config,
        meta:   dict = {}
    ):
        super(BlipGraph, self).__init__(
            name, config, meta
        )
        self.config = config
        # linear evaluation protocol
        self.linear = False

        # construct augmentations
        self.construct_augmentations()
        # construct the model
        self.construct_model()
        # register hooks
        self.register_forward_hooks()

    def linear_evaluation(self):
        self.linear = True
        for embedding in self.embedding_dict.keys():
            self.embedding_dict[embedding].requires_grad = False
        for reduction in self.reduction_dict.keys():
            self.reduction_dict[reduction].requires_grad = False
        for classification in self.classification_dict.keys():
            self.classification_dict[classification].requires_grad = False
        for linear_classification in self.linear_classification_dict.keys():
            self.linear_classification_dict[linear_classification].requires_grad = True

    def contrastive_learning(self):
        self.linear = False
        for embedding in self.embedding_dict.keys():
            self.embedding_dict[embedding].requires_grad = True
        for reduction in self.reduction_dict.keys():
            self.reduction_dict[reduction].requires_grad = True
        for classification in self.classification_dict.keys():
            self.classification_dict[classification].requires_grad = True
        for linear_classification in self.linear_classification_dict.keys():
            self.linear_classification_dict[linear_classification].requires_grad = False

    def construct_augmentations(self):
        """
        Currently there are four augmentations that can be implemented
        as part of the contrastive learning in BlipGraph,
        'jitter', 'flip', 'shear' and 'rotate'.

        """
        if 'augmentations' not in self.config.keys():
            self.logger.error(
                'augmentations section not in BlipGraph config!'
            )

        augmentations = []
        for augmentation, params in self.config["augmentations"].items():
            if augmentation == 'jitter':
                augmentations.append(T.RandomJitter(
                    params)
                )
            elif augmentation == 'flip':
                for ii in range(len(
                    params['positions']
                )):
                    augmentations.append(T.RandomFlip(
                        params['positions'][ii],
                        params['probabilities'][ii]
                    ))
            elif augmentation == 'shear':
                augmentations.append(T.RandomShear(params))
            elif augmentation == "rotate":
                for ii in range(len(params['degrees'])):
                    augmentations.append(T.RandomRotate(
                        params['degrees'][ii],
                        params['axis'][ii]
                    ))
            elif augmentation == 'number_of_augmentations':
                self.number_of_augmentations = params

        self.augmentations = T.Compose(augmentations)

    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.

        This network consists of three sections:
            (1) embedding layers - graph layers for embedding the point clouds.
            (2) reduction layers - layer for reducing the embedding to a
                                   fixed dimension.
            (3) classification layers - layers for classifying the
                                        reduced embeddings.

        each section should have its own config subsection like the following:

        BlipGraph:
            augmentations:
                ...
            embedding:
                ...
            reduction:
                ...
            classification:
                ...
        """
        self.logger.info(
            f"Attempting to build BlipGraph architecture using config: \
            {self.config}"
        )

        self.embedding_dicts = []
        _embedding_dict = OrderedDict()
        _reduction_dict = OrderedDict()
        _classification_dict = OrderedDict()
        _linear_classification_dict = OrderedDict()

        if 'input_dimension' not in self.config.keys():
            self.logger.error(
                'input_dimension not specified in BlipGraph config!'
            )
        if 'classifications' not in self.config.keys():
            self.logger.error(
                'classifications not specified in BlipGraph config! \
                should be a list of names of classes!'
            )
        if 'add_summed_adc' not in self.config.keys():
            self.logger.warn(
                'add_summed_adc not specified in BlipGraph config! \
                setting to False'
            )
            self.config['add_summed_adc'] = False

        _input_dimension = self.config['input_dimension']
        _num_embedding_outputs = 0

        # check embedding parameters
        if "embedding" not in self.config.keys():
            self.logger.error('embedding section not in BlipGraph config!')
        if "reduction" not in self.config.keys():
            self.logger.error('reduction section not in BlipGraph config!')
        if "classification" not in self.config.keys():
            self.logger.error(
                'classification section not in BlipGraph config!'
            )

        for embedding, params in self.config['embedding'].items():
            if self.config['embedding'][embedding]['embedding_type'] == 'dynamic_edge_conv':
                if 'number_of_neighbors' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'number_of_neighbors not specified for layer {embedding}! setting to 5')
                    self.config['embedding'][embedding]['number_of_neighbors'] = 5
                if 'aggregation' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'aggregation not specified for layer {embedding}! setting to "max"')
                    self.config['embedding'][embedding]['aggregation'] = 'max'
                if 'embedding_mlp_layers' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'embedding_mlp_layers not specified for layer {embedding}! setting to [5,10,25,10]')
                    self.config['embedding'][embedding]['embedding_mlp_layers'] = [5, 10, 25, 10]
                if 'dropout' not in self.config['embedding'][embedding].keys():
                    self.logger.warn(f'dropout not specified for layer {embedding}! setting to 0.')
                    self.config['embedding'][embedding]['dropout'] = 0.
                if 'activation' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'activation not specified for layer {embedding}! setting to "LeakyReLU"')
                    self.config['embedding'][embedding]['activation'] = 'LeakyReLU'
                if 'activation_params' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'activation_params not specified for layer {embedding}! setting to empty dictionary')
                    self.config['embedding'][embedding]['activation_params'] = {}
            if self.config['embedding'][embedding]['embedding_type'] == 'point_net_conv':
                if 'fps_ratio' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'fps_ratio not specified for layer {embedding}! setting to 0.5')
                    self.config['embedding'][embedding]['fps_ratio'] = '0.5'
                if 'cluster_radius' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'cluster_radius not specified for layer {embedding}! setting to 0.25')
                    self.config['embedding'][embedding]['cluster_radius'] = 0.25
                if 'max_number_of_neighbors' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'max_number_of_neighbors not specified for layer {embedding}! setting to 5')
                    self.config['embedding'][embedding]['max_number_of_neighbors'] = 5
                if 'add_self_loops' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'add_self_loops not specified for layer {embedding}! setting to True')
                    self.config['embedding'][embedding]['add_self_loops'] = True
                if 'local_mlp_layers' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'local_mlp_layers not specified for layer {embedding}! setting to [5,10,25,10]')
                    self.config['embedding'][embedding]['local_mlp_layers'] = [5, 10, 25, 10]
                if 'dropout' not in self.config['embedding'][embedding].keys():
                    self.logger.warn(f'dropout not specified for layer {embedding}! setting to 0.')
                    self.config['embedding'][embedding]['dropout'] = 0.
                if 'activation' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'activation not specified for layer {embedding}! setting to "LeakyReLU"')
                    self.config['embedding'][embedding]['activation'] = 'LeakyReLU'
                if 'activation_params' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'activation_params not specified for layer {embedding}! setting to empty dictionary')
                    self.config['embedding'][embedding]['activation_params'] = {}
            if self.config['embedding'][embedding]['embedding_type'] == 'point_transformer_conv':
                if 'fps_ratio' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'fps_ratio not specified for layer {embedding}! setting to 0.5')
                    self.config['embedding'][embedding]['fps_ratio'] = '0.5'
                if 'cluster_radius' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'cluster_radius not specified for layer {embedding}! setting to 0.25')
                    self.config['embedding'][embedding]['cluster_radius'] = 0.25
                if 'max_number_of_neighbors' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'max_number_of_neighbors not specified for layer {embedding}! setting to 5')
                    self.config['embedding'][embedding]['max_number_of_neighbors'] = 5
                if 'add_self_loops' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'add_self_loops not specified for layer {embedding}! setting to True')
                    self.config['embedding'][embedding]['add_self_loops'] = True
                if 'pos_nn_layers' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'pos_nn_layers not specified for layer {embedding}! setting to [5,10,25,10]')
                    self.config['embedding'][embedding]['pos_nn_layers'] = [5, 10, 25, 10]
                if 'pos_nn_dropout' not in self.config['embedding'][embedding].keys():
                    self.logger.warn(f'pos_nn_dropout not specified for layer {embedding}! setting to 0.')
                    self.config['embedding'][embedding]['pos_nn_dropout'] = 0.
                if 'pos_nn_activation' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'pos_nn_activation not specified for layer {embedding}! setting to "LeakyReLU"')
                    self.config['embedding'][embedding]['pos_nn_activation'] = 'LeakyReLU'
                if 'pos_nn_activation_params' not in self.config['embedding'][embedding]:
                    self.logger.warn(
                        f'pos_nn_activation_params not specified for layer {embedding}! setting to empty dictionary'
                    )
                    self.config['embedding'][embedding]['pos_nn_activation_params'] = {}
                if 'attn_nn_layers' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'attn_nn_layers not specified for layer {embedding}! setting to [5,10,25,10]')
                    self.config['embedding'][embedding]['attn_nn_layers'] = [5, 10, 25, 10]
                if 'attn_nn_dropout' not in self.config['embedding'][embedding].keys():
                    self.logger.warn(f'attn_nn_dropout not specified for layer {embedding}! setting to 0.')
                    self.config['embedding'][embedding]['attn_nn_dropout'] = 0.
                if 'attn_nn_activation' not in self.config['embedding'][embedding]:
                    self.logger.warn(f'attn_nn_activation not specified for layer {embedding}! setting to "LeakyReLU"')
                    self.config['embedding'][embedding]['attn_nn_activation'] = 'LeakyReLU'
                if 'attn_nn_activation_params' not in self.config['embedding'][embedding]:
                    self.logger.warn(
                        f'attn_nn_activation_params not specified for layer {embedding}! setting to empty dictionary'
                    )
                    self.config['embedding'][embedding]['attn_nn_activation_params'] = {}

        # iterate over embeddings
        embedding_config = self.config['embedding']
        for ii, embedding in enumerate(embedding_config.keys()):

            # DynamicEdgeConv
            if embedding_config[embedding]['embedding_type'] == 'dynamic_edge_conv':
                _embedding_dict[embedding] = DynamicEdgeConv(
                    nn=MLP(
                        [2 * _input_dimension] + embedding_config[embedding]['embedding_mlp_layers'],
                        dropout=embedding_config[embedding]['dropout'],
                        act=embedding_config[embedding]['activation'],
                        act_kwargs=embedding_config[embedding]['activation_params']
                    ),
                    k=embedding_config[embedding]['number_of_neighbors'],
                    aggr=embedding_config[embedding]['aggregation']
                )
                _input_dimension = embedding_config[embedding]['embedding_mlp_layers'][-1]

            # PointNetConv
            elif embedding_config[embedding]['embedding_type'] == 'point_net_conv':
                _embedding_dict[embedding] = PointNetModule(
                    local_mlp=MLP(
                        [_input_dimension] + embedding_config[embedding]['local_mlp_layers'],
                        dropout=embedding_config[embedding]['dropout'],
                        act=embedding_config[embedding]['activation'],
                        act_kwargs=embedding_config[embedding]['activation_params']
                    ),
                    fps_ratio=embedding_config[embedding]['fps_ratio'],
                    cluster_radius=embedding_config[embedding]['cluster_radius'],
                    max_number_neighbors=embedding_config[embedding]['max_number_neighbors'],
                    add_self_loops=embedding_config[embedding]['add_self_loops']
                )
                _input_dimension = embedding_config[embedding]['local_mlp_layers'][-1]

            # PointTransformerConv
            elif embedding_config[embedding]['embedding_type'] == 'point_transformer_conv':
                _embedding_dict[embedding] = PointTransformerModule(
                    in_channels=_input_dimension,
                    out_channels=embedding_config[embedding]['pos_nn_layers'][-1],
                    pos_nn=MLP(
                        [_input_dimension] + embedding_config[embedding]['pos_nn_layers'],
                        dropout=embedding_config[embedding]['pos_nn_dropout'],
                        act=embedding_config[embedding]['pos_nn_activation'],
                        act_kwargs=embedding_config[embedding]['pos_nn_activation_params']
                    ),
                    attn_nn=MLP(
                        [embedding_config[embedding]['pos_nn_layers'][-1]] + embedding_config[embedding]['attn_nn_layers'],
                        dropout=embedding_config[embedding]['attn_nn_dropout'],
                        act=embedding_config[embedding]['attn_nn_activation'],
                        act_kwargs=embedding_config[embedding]['attn_nn_activation_params']
                    ),
                    fps_ratio=embedding_config[embedding]['fps_ratio'],
                    cluster_radius=embedding_config[embedding]['cluster_radius'],
                    max_number_neighbors=embedding_config[embedding]['max_number_neighbors'],
                    add_self_loops=embedding_config[embedding]['add_self_loops']
                )
                _input_dimension = embedding_config[embedding]['pos_nn_layers'][-1]
            else:
                self.logger.error(f'specified embedding type {embedding_config[embedding]["embedding_type"]} not allowed!')
            _num_embedding_outputs += _input_dimension

        # reduction layer
        # projection head
        if "projection_head_layers" in self.config['reduction'].keys():
            self.projection_head = True
            if 'projection_head_dropout' not in self.config['reduction'].keys():
                self.logger.warn('projection_head_dropout not specified for reduction layer! setting to 0.')
                self.config['reduction']['projection_head_dropout'] = 0.
            if 'projection_head_activation' not in self.config['reduction']:
                self.logger.warn('projection_head_activation not specified for reduction layer! setting to "LeakyReLU"')
                self.config['reduction']['projection_head_activation'] = 'LeakyReLU'
            if 'projection_head_activation_params' not in self.config['reduction']:
                self.logger.warn(
                    'projection_head_activation_params not specified for reduction layer! setting to empty dictionary'
                )
                self.config['reduction']['projection_head_activation_params'] = {}
        else:
            self.logger.info('no projection_head specified.')
            self.projection_head = False

        reduction_config = self.config['reduction']
        # add linear layer Encoder head
        _reduction_dict['linear_layer'] = Linear(
            _num_embedding_outputs,
            reduction_config['linear_output']
        )
        if reduction_config['reduction_type'] == 'add_pool':
            self.pooling_layer = global_add_pool
        elif reduction_config['reduction_type'] == 'mean_pool':
            self.pooling_layer = global_mean_pool
        else:
            self.pooling_layer = global_max_pool

        if self.config["add_summed_adc"]:
            reduction_config['linear_output'] += 1

        # add output mlp Projection head (See explanation in SimCLRv2)
        if self.projection_head:
            _reduction_dict['projection_head'] = MLP(
                [reduction_config['linear_output']] + reduction_config['projection_head_layers'],
                dropout=reduction_config['projection_head_dropout'],
                act=reduction_config['projection_head_activation'],
                act_kwargs=reduction_config['projection_head_activation_params']
            )
            reduction_output = reduction_config['projection_head_layers'][-1]
        else:
            reduction_output = reduction_config['linear_output']

        # classification layer
        if "mlp_output_layers" not in self.config['classification']:
            self.logger.error('mlp_output_layers not specified in classification section of config!')
        if 'dropout' not in self.config['classification'].keys():
            self.logger.warn('dropout not specified for classification layer! setting to 0.')
            self.config['classification']['dropout'] = 0.
        if 'activation' not in self.config['classification']:
            self.logger.warn('activation not specified for classification layer! setting to "LeakyReLU"')
            self.config['classification']['activation'] = 'LeakyReLU'
        if 'activation_params' not in self.config['classification']:
            self.logger.warn('activation_params not specified for classification layer! setting to empty dictionary')
            self.config['classification']['activation_params'] = {}

        classification_config = self.config['classification']

        for ii, classification in enumerate(self.config["classifications"]):
            _classification_dict[f'{classification}'] = MLP(
                [reduction_output] + classification_config['mlp_output_layers'] + [classification_config['out_channels'][ii]],
                dropout=classification_config['dropout'],
                act=classification_config['activation'],
                act_kwargs=classification_config['activation_params']
            )
            _linear_classification_dict[f'{classification}'] = MLP(
                [reduction_output] + [classification_config['out_channels'][ii]],
                act=classification_config['activation'],
                act_kwargs=classification_config['activation_params']
            )

        self.embedding_dict = nn.ModuleDict(_embedding_dict)
        self.reduction_dict = nn.ModuleDict(_reduction_dict)
        self.classification_dict = nn.ModuleDict(_classification_dict)
        self.linear_classification_dict = nn.ModuleDict(_linear_classification_dict)
        self.softmax = nn.Softmax(dim=1)

        # record the info
        self.logger.info(
            f"Constructed BlipGraph with dictionaries {self.embedding_dict}, {self.reduction_dict}, {self.classification_dict}"
        )

    def forward(
        self,
        data
    ):
        """
        Iterate over the model dictionary
        """
        if self.training:
            reductions, projection_head, classifications = [], [], [
                [] for kk in range(self.number_of_augmentations)
            ]
            for kk in range(self.number_of_augmentations):
                # Create augmentations
                augmentations = self.augmentations(data).to(self.device)
                pos, batch = augmentations.pos, augmentations.batch
                if self.config["add_summed_adc"]:
                    summed_adc = augmentations.summed_adc

                # Pass through embedding dictionary
                for ii, embedding in enumerate(self.embedding_dict.keys()):
                    pos = self.embedding_dict[embedding](pos, batch)
                    if ii == 0:
                        linear_input = pos
                    else:
                        linear_input = torch.cat([linear_input, pos], dim=1)

                # Pass through reduction dictionary
                linear_output = self.reduction_dict['linear_layer'](linear_input)

                # Apply Pooling
                linear_pool = self.pooling_layer(linear_output, batch)

                if self.config["add_summed_adc"]:
                    linear_pool = torch.cat([linear_pool, summed_adc.unsqueeze(1)], dim=1)

                reductions.append(linear_pool)

                if self.projection_head:
                    linear_pool = self.reduction_dict['projection_head'](linear_pool)
                projection_head.append(linear_pool)

                if not self.linear:
                    # Pass through classification dictionary
                    for jj, classification in enumerate(
                        self.classification_dict.keys()
                    ):
                        classifications[jj].append(
                            self.classification_dict[classification](linear_pool)
                        )
                else:
                    # Pass through linear classification dictionary
                    for jj, classification in enumerate(
                        self.classification_dict.keys()
                    ):
                        classifications[jj].append(
                            self.linear_classification_dict[classification](linear_pool)
                        )

            outputs = {
                classification: torch.cat(classifications[jj])
                for jj, classification in enumerate(
                    self.classification_dict.keys()
                )
            }
            outputs['reductions'] = torch.cat(reductions)
            outputs['projection_head'] = torch.cat(projection_head)
            outputs['augmentations'] = self.number_of_augmentations
        else:
            pos = data.pos.to(self.device)
            batch = data.batch.to(self.device)
            if self.config["add_summed_adc"]:
                summed_adc = data.summed_adc.to(self.device)
            for ii, embedding in enumerate(self.embedding_dict.keys()):
                pos = self.embedding_dict[embedding](pos, batch)
                if ii == 0:
                    linear_input = pos
                else:
                    linear_input = torch.cat([linear_input, pos], dim=1)

            # Pass through reduction dictionary
            linear_output = self.reduction_dict['linear_layer'](linear_input)
            linear_pool = self.pooling_layer(linear_output, batch)

            if self.config["add_summed_adc"]:
                linear_pool = torch.cat(
                    [linear_pool, summed_adc.unsqueeze(1)], dim=1
                )

            reductions = linear_pool

            if self.projection_head:
                linear_pool = self.reduction_dict['projection_head'](linear_pool)

            if not self.linear:
                outputs = {
                    classifications: self.classification_dict[classifications](linear_pool)
                    for classifications in self.classification_dict.keys()
                }
            else:
                outputs = {
                    classifications: self.linear_classification_dict[classifications](linear_pool)
                    for classifications in self.classification_dict.keys()
                }

            outputs['reductions'] = reductions
            outputs['projection_head'] = linear_pool
            outputs['augmentations'] = 1
        return outputs
