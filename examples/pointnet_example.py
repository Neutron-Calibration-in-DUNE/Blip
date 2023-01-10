"""
Training loop for a BLIP model   
"""
import torch_geometric.transforms as T
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# blip imports
from blip.dataset.arrakis import Arrakis
from blip.dataset.blip import BlipDataset
from blip.utils.loader import Loader
from blip.losses import LossHandler
from blip.optimizers import Optimizer
from blip.metrics import MetricHandler
from blip.trainer import Trainer
from blip.utils.callbacks import CallbackHandler
from blip.utils.utils import get_files, save_model
from blip.models import BLIP
import numpy as np
import torch
import os
import shutil
from datetime import datetime


if __name__ == "__main__":

    # clean up directories first
    save_model()

    prepare_data = True
    if prepare_data:
        arrakis_dataset = Arrakis(
            "../../ArrakisEventDisplay/data/multiple_neutron_arrakis.root"
        )
        arrakis_dataset.generate_training_data()
    """
    Now we load our dataset as a torch dataset (blipDataset),
    and then feed that into a dataloader.
    """
    blip_dataset = BlipDataset(
        name = "blip_example",
        input_file='../data/point_cloud_view0.npz',
        root="."
    )
    blip_loader = Loader(
        blip_dataset, 
        batch_size=64,
        test_split=0.1,
        test_seed=100,
        validation_split=0.1,
        validation_seed=100,
        num_workers=4
    )
    """
    Construct the blip Model, specify the loss and the 
    optimizer and metrics.
    """
    blip_config = {
        # input dimension
        'input_dimension':  3,
        # number of dynamic edge convs
        'num_dynamic_edge_conv':    2,
        # edge conv layer values
        'edge_conv_mlp_layers': [
            [64, 64],
            [64, 64]
        ],
        'number_of_neighbors':  20,
        'aggregation_operators': [
            'sum', 'sum'
        ],
        # linear layer
        'linear_output':    128,
        'mlp_output_layers': [128, 256, 32],
        'augmentations':    [
            T.RandomJitter(0.03), 
            T.RandomFlip(1), 
            T.RandomShear(0.2)
        ],
        # number of augmentations per batch
        'number_of_augmentations': 2
    }
    blip_model = BLIP(
        name = 'blip_test',
        cfg  = blip_config
    ) 

    # create loss, optimizer and metrics
    blip_optimizer = Optimizer(
        model=blip_model,
        optimizer='Adam'
    )

    # create criterions
    blip_loss_config = {
        'NTXEntropyLoss':   {
            'alpha':    1.0,
            'temperature': 0.10,
        }        
    }
    blip_loss = LossHandler(
        name="blip_loss",
        cfg=blip_loss_config,
    )
    
    # create metrics
    blip_metric_config = {
        # 'LatentSaver':  {},
        # 'TargetSaver':  {},
        # 'InputSaver':   {},
        # 'OutputSaver':  {},
    }
    blip_metrics = MetricHandler(
        "blip_metric",
        cfg=blip_metric_config,
    )

    # create callbacks
    callback_config = {
        'loss':   {'criterion_list': blip_loss},
        'metric': {'metrics_list':   blip_metrics},
    }
    blip_callbacks = CallbackHandler(
        "blip_callbacks",
        callback_config
    )

    # create trainer
    blip_trainer = Trainer(
        model=blip_model,
        criterion=blip_loss,
        optimizer=blip_optimizer,
        metrics=blip_metrics,
        callbacks=blip_callbacks,
        metric_type='test',
        gpu=True,
        gpu_device=0
    )
    
    blip_trainer.train(
        blip_loader,
        epochs=25,
        checkpoint=25,
        save_predictions=False
    )

    # Get sample batch
    sample = next(iter(blip_loader.train_loader))

    # Get representations
    pools, compacts, labels = blip_model.forward_eval(sample)
    compacts = compacts.cpu().detach()

    labels = sample.category.cpu().detach().numpy()

    # Get low-dimensional t-SNE Embeddings
    h_embedded = TSNE(n_components=2, learning_rate='auto',
                    init='random').fit_transform(compacts.numpy())

    # Plot
    ax = sns.scatterplot(x=h_embedded[:,0], y=h_embedded[:,1], hue=labels, 
                        alpha=0.5, palette="tab10")

    # Add labels to be able to identify the data points
    annotations = list(range(len(h_embedded[:,0])))

    def label_points(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x']+.02, point['y'], str(int(point['val'])))

    label_points(pd.Series(h_embedded[:,0]), 
                pd.Series(h_embedded[:,1]), 
                pd.Series(annotations), 
                plt.gca())
    plt.show()