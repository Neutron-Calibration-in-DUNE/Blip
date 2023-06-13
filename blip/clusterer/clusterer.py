"""
Class for a generic clusterer.
"""
import torch
import numpy as np
import os
from tqdm import tqdm
from blip.dataset.blip import BlipDataset
from blip.clustering_algorithms import ClusteringAlgorithmHandler
from blip.utils.logger import Logger
from blip.utils.timing import Timers
from blip.utils.memory import MemoryTrackers
from blip.metrics import MetricHandler
from blip.utils.callbacks import CallbackHandler
from blip.utils.callbacks import TimingCallback, MemoryTrackerCallback
import blip.utils.utils as utils

class Clusterer:
    """
    This class is ... 
    """
    def __init__(self,
        name:   str,
        clustering_algorithms:  ClusteringAlgorithmHandler=None,
        clustering_metrics:     MetricHandler=None,
        clustering_callbacks:   CallbackHandler=None,
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
        
        self.clustering_algorithms = clustering_algorithms
        self.clustering_metrics = clustering_metrics

        # check for devices
        self.device = device
        self.gpu = gpu
        self.seed = seed
        
        if clustering_callbacks == None:
            # add generic clustering_callbacks
            self.clustering_callbacks = CallbackHandler(
                name="default"
            )
        else:
            self.clustering_callbacks = clustering_callbacks
        # add timing info
        self.timers = Timers(gpu=self.gpu)
        self.timer_callback = TimingCallback(
            self.timing_dir,
            self.timers
        )
        self.clustering_callbacks.add_callback(self.timer_callback)

        # add memory info
        self.memory_trackers = MemoryTrackers(gpu=self.gpu)
        self.memory_callback = MemoryTrackerCallback(
            self.memory_dir,
            self.memory_trackers
        )
        self.clustering_callbacks.add_callback(self.memory_callback)
    
    def cluster(self,
        dataset_loader,             # dataset_loader to pass in
        num_parameters: int=10,
        eps_range:      list=[1.0, 100.0],
        progress_bar:   bool=True,  # progress bar from tqdm
        rewrite_bar:    bool=False, # wether to leave the bars after each epoch
        save_predictions:bool=True, # wether to save network outputs for all events to original file
        no_timing:      bool=False,     # wether to keep the bare minimum timing info as a callback          
    ):
        """
        Main clustering loop.  First, we see if the user wants to omit timing information.
        """
        # run consistency check
        # self.logger.info(f"running consistency check...")

        # setting values in callbacks
        self.clustering_callbacks.set_device(self.device)
        self.clustering_callbacks.set_training_info(
            num_parameters,
            dataset_loader.num_train_batches,
            dataset_loader.num_validation_batches,
            dataset_loader.num_test_batches
        )
        # Clustering
        self.logger.info(
            f"running clustering on '{dataset_loader.dataset.name}' " + 
            f"for {num_parameters} parameters in range {eps_range}."
        )
        if no_timing:
            self.__cluster_no_timing(
                dataset_loader,
                num_parameters,
                eps_range,
                progress_bar,
                rewrite_bar,
                save_predictions
            )
        else:
            self.__cluster_with_timing(
                dataset_loader,
                num_parameters,
                eps_range,
                progress_bar,
                rewrite_bar,
                save_predictions
            )

    def __cluster_with_timing(self,
        dataset_loader,             # dataset_loader to pass in
        num_parameters: int=10,     # number of parameter values to cluster with
        eps_range:      list=[1.0, 100.0],  # range of parameter values
        progress_bar:   bool=True,  # progress bar from tqdm
        rewrite_bar:    bool=False, # wether to leave the bars after each epoch
        save_predictions:bool=True, # wether to save network outputs for all events to original file
    ):
        """
        """
        eps_steps = np.linspace(eps_range[0], eps_range[1], num_parameters)
        parameter_values = []
        for kk in range(num_parameters):
            
            if (progress_bar):
                event_loop = tqdm(
                    enumerate(dataset_loader.all_loader, 0), 
                    total=len(dataset_loader.all_loader), 
                    leave=rewrite_bar,
                    colour='green'
                )
            else:
                event_loop = enumerate(dataset_loader.all_loader, 0)
            
            """            
            Setup timing/memory information for epoch.
            """
            self.timers.timers['parameter_clustering'].start()
            self.memory_trackers.memory_trackers['parameter_clustering'].start()

            self.timers.timers['parameter_change'].start()
            self.memory_trackers.memory_trackers['parameter_change'].start()
            # replace this with a call to parameter set that will
            # generate a set of parameters to use for this scan.
            parameters = {'eps': eps_steps[kk]}
            parameter_values.append(parameters['eps'])
            self.memory_trackers.memory_trackers['parameter_change'].end()
            self.timers.timers['parameter_change'].end()

            self.timers.timers['cluster_data'].start()
            self.memory_trackers.memory_trackers['cluster_data'].start()
            for ii, data in event_loop:
                self.memory_trackers.memory_trackers['cluster_data'].end()
                self.timers.timers['cluster_data'].end()
                
                """
                Send the event to the clustering algorithms to produce
                clustering results.  
                """
                self.timers.timers['cluster_algorithm'].start()
                self.memory_trackers.memory_trackers['cluster_algorithm'].start()
                clustering = self.clustering_algorithms.cluster(parameters, data)
                self.memory_trackers.memory_trackers['cluster_algorithm'].end()
                self.timers.timers['cluster_algorithm'].end()

                # update metrics
                self.timers.timers['cluster_metrics'].start()
                self.memory_trackers.memory_trackers['cluster_metrics'].start()
                self.clustering_metrics.update(clustering, data)
                self.memory_trackers.memory_trackers['cluster_metrics'].end()
                self.timers.timers['cluster_metrics'].end()

                # update progress bar
                self.timers.timers['cluster_progress'].start()
                self.memory_trackers.memory_trackers['cluster_progress'].start()
                if (progress_bar):
                    event_loop.set_description(f"Parameter: [{kk+1}/{num_parameters}]")
                    event_loop.set_postfix_str(f"loss")
                self.memory_trackers.memory_trackers['cluster_progress'].end()
                self.timers.timers['cluster_progress'].end()

                self.timers.timers['cluster_data'].start()
                self.memory_trackers.memory_trackers['cluster_data'].start()

            self.memory_trackers.memory_trackers['parameter_clustering'].end()
            self.timers.timers['parameter_clustering'].end()

            # evaluate callbacks
            self.timers.timers['cluster_callbacks'].start()
            self.memory_trackers.memory_trackers['cluster_callbacks'].start()
            self.clustering_callbacks.evaluate_epoch(train_type='cluster')
            self.memory_trackers.memory_trackers['cluster_callbacks'].end()
            self.timers.timers['cluster_callbacks'].end()

        # for now, just save the arrays to a numpy file,
        # later we will do something more intelligent with them.
        self.clustering_callbacks.callbacks['AdjustedRandIndexCallback'].parameter_values = eps_steps
        self.clustering_callbacks.evaluate_clustering()
        if save_predictions:
            np.savez(
                'data/cluster_results.npz',
                parameter_values=parameter_values,
                **self.clustering_callbacks.callbacks['AdjustedRandIndexCallback'].adjusted_rand_index,
                **self.clustering_callbacks.callbacks['AdjustedRandIndexCallback'].adjusted_rand_index_individual['DBSCAN']
            )


    