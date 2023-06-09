"""
Class for a generic clusterer.
"""
import torch
import numpy as np
import os
from tqdm import tqdm
from blip.dataset.blip import BlipDataset
from blip.utils.logger import Logger
from blip.utils.timing import Timers
from blip.utils.memory import MemoryTrackers
from blip.utils.callbacks import CallbackHandler
from blip.utils.callbacks import TimingCallback, MemoryTrackerCallback
import blip.utils.utils as utils

class Clusterer:
    """
    This class is ... 
    """
    def __init__(self,
        name:   str,
        callbacks:  CallbackHandler=None,
        device:     str='cpu',
        gpu:        bool=False,
        seed:       int=0,
    ):
        """
        """
        self.name = name + "_clusterer"
        self.logger = Logger(self.name, output='both', file_mode='w')
        self.logger.info(f"constructing clusterer.")
        # Check for compatability with parameters

        # define directories
        self.predictions_dir = f'predictions/{self.name}/'
        self.timing_dir    = f'plots/{self.name}/timing/'
        self.memory_dir    = f'plots/{self.name}/memory/'

        # create directories
        if not os.path.isdir(self.predictions_dir):
            self.logger.info(f"creating predictions directory '{self.predictions_dir}'")
            os.makedirs(self.predictions_dir)
        if not os.path.isdir(self.timing_dir):
            self.logger.info(f"creating timing directory '{self.timing_dir}'")
            os.makedirs(self.timing_dir)
        if not os.path.isdir(self.memory_dir):
            self.logger.info(f"creating memory directory '{self.memory_dir}'")
            os.makedirs(self.memory_dir)
        
        # check for devices
        self.device = device
        self.gpu = gpu
        self.seed = seed
        
        if callbacks == None:
            # add generic callbacks
            self.callbacks = CallbackHandler(
                name="default"
            )
        else:
            self.callbacks = callbacks
        # add timing info
        self.timers = Timers(gpu=self.gpu)
        self.timer_callback = TimingCallback(
            self.timing_dir,
            self.timers
        )
        self.callbacks.add_callback(self.timer_callback)

        # add memory info
        self.memory_trackers = MemoryTrackers(gpu=self.gpu)
        self.memory_callback = MemoryTrackerCallback(
            self.memory_dir,
            self.memory_trackers
        )
        self.callbacks.add_callback(self.memory_callback)
    
    def cluster(self,
        dataset_loader,             # dataset_loader to pass in
        progress_bar:   str='all',  # progress bar from tqdm
        rewrite_bar:    bool=False, # wether to leave the bars after each epoch
        save_predictions:bool=True, # wether to save network outputs for all events to original file
        no_timing:  bool=False,     # wether to keep the bare minimum timing info as a callback          
    ):
        """
        Main clustering loop.  First, we see if the user wants to omit timing information.
        """
        # run consistency check
        # self.logger.info(f"running consistency check...")

        # setting values in callbacks
        self.callbacks.set_device(self.device)
        self.callbacks.set_training_info(
            0,
            dataset_loader.num_train_batches,
            dataset_loader.num_validation_batches,
            dataset_loader.num_test_batches
        )
        # Clustering
        self.logger.info(f"running clustering on '{dataset_loader.dataset.name}' for {num_parameters} parameters.")
        if no_timing:
            # TODO: Need to fix this so that memory and timing callbacks aren't called.
            self.__train_no_timing(
                dataset_loader,
                epochs,
                checkpoint,
                progress_bar,
                rewrite_bar,
                save_predictions
            )
        else:
            self.__train_with_timing(
                dataset_loader,
                epochs,
                checkpoint,
                progress_bar,
                rewrite_bar,
                save_predictions
            )
    