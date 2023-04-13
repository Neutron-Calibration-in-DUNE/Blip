"""
Training loop for a BLIP model   
"""
# blip imports
from blip.dataset.blip import BlipDataset
from blip.utils.loader import Loader
from blip.losses import LossHandler
from blip.optimizers import Optimizer
from blip.metrics import MetricHandler
from blip.trainer import Trainer
from blip.utils.callbacks import CallbackHandler
from blip.utils.utils import get_files, save_model
from blip.models import PointNetClassification 

import torch_geometric.transforms as T

if __name__ == "__main__":

    # clean up directories first
    save_model()

    """
    Now we load our dataset as a torch dataset (blipDataset),
    and then feed that into a dataloader.
    """
    blip_dataset = BlipDataset(
        name = "gain_test1_dataset",
        input_file='data/point_cloud_view0.npz',
        root="."
    )
    blip_loader = Loader(
        blip_dataset, 
        batch_size=50,
        test_split=0.3,
        test_seed=100,
        validation_split=0.3,
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
        'num_embedding':    2,
        # edge conv layer values
        'embedding_mlp_layers': [
            [64, 64],
            [64, 64]
        ],
        'number_of_neighbors':  20,
        'aggregation_operators': [
            'max', 'max'
        ],
        # linear layer
        'linear_output':    128,
        'mlp_output_layers': [128, 256, 32],
        'classification_layers': [
            32, 64, 32, blip_dataset.number_classes
        ],
        'augmentations':    [
            T.RandomJitter(0.03), 
            T.RandomFlip(1), 
            T.RandomShear(0.2),
            T.RandomRotate(15, axis=2)
        ],
        # number of augmentations per batch
        'number_of_augmentations': 2
    }
    blip_model = PointNetClassification(
        name = 'gain_test1',
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
        },
        'SparseCrossEntropyLoss': {
            'alpha':    1.0
        }        
    }
    blip_loss = LossHandler(
        name="blip_loss",
        cfg=blip_loss_config,
    )
    
    # create metrics
    blip_metric_config = {
        'auroc': {
            'num_classes': blip_dataset.number_classes
        },
        'confusion_matrix': {
            'num_classes': blip_dataset.number_classes
        },
        'dice_score': {
            'num_classes': blip_dataset.number_classes
        },
        'jaccard_index': {
            'num_classes': blip_dataset.number_classes
        },
        'precision': {
            'num_classes': blip_dataset.number_classes
        },
        'recall': {
            'num_classes': blip_dataset.number_classes
        }
    }
    blip_metrics = MetricHandler(
        "blip_metric",
        cfg=blip_metric_config,
        labels=blip_dataset.labels
    )

    # create callbacks
    callback_config = {
        'loss':   {'criterion_list': blip_loss},
        'metric': {'metrics_list':   blip_metrics},
        # 'embedding': {
        #     'metrics_list':     blip_metrics
        # },
        'confusion_matrix': {
            'metrics_list':     blip_metrics
        },
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
        save_predictions=True
    )