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
from blip.models import BLIP
import numpy as np
import torch
import os
import shutil
from datetime import datetime


if __name__ == "__main__":

    # clean up directories first
    save_model()

    """
    Now we load our dataset as a torch dataset (blipDataset),
    and then feed that into a dataloader.
    """
    features = [
            'gut_m0', 
            'gut_m12', 
            'gut_A0', 
            'gut_tanb', 
            'sign_mu'
    ]
    blip_dataset = BlipDataset(
        name="blip_dataset",
        input_file='datasets/higgs_dm_lsp_symmetric.npz',
        features = features,
        classes = ['valid']
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
        # dimension of the input variables
        'input_dimension':      3,
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
        'L2OutputLoss':   {
            'alpha':    1.0,
            'reduction':'mean',
        }        
    }
    blip_loss = LossHandler(
        name="blip_loss",
        cfg=blip_loss_config,
    )
    
    # create metrics
    blip_metric_config = {
        'LatentSaver':  {},
        'TargetSaver':  {},
        'InputSaver':   {},
        'OutputSaver':  {},
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
        epochs=10,
        checkpoint=25
    )