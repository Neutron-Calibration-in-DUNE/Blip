
"""
Generic model code.
"""
import torch, os, csv, getpass
from torch                import nn
from time                 import time
from datetime             import datetime
from tqdm                 import tqdm
from torch_geometric.data import Data

from blip.models                import ModelHandler
from blip.module.common         import *
from blip.module.generic_module import GenericModule

class BlipNetModule(GenericModule):
    """
    The BlipNet module consists of three different steps which can be run together by
    chaining them in the 'module_mode' parameter.  The three steps are, (1) optimize_blip_graph
     - which runs a series of blip_graph steps such as a hyperparameter scan over singles
    data, a linear evaluation, and then analysis - (2) the evaluate_merge_tree - which creates
    the merge tree for a set of full simulation/data events and evaluates each node in the tree
    using the optimized blip_graph - (3) the monte_carlo_tree_search step, which generates a
    MCTS for each event and attempts to purify blip objects within the events based on the
    merge tree evaluation scores.
    """
    def __init__(self,
        name:   str,
        config: dict={},
        mode:   str='',
        meta:   dict={}
    ):
        self.name = name + "_ml_module"
        super(BlipNetModule, self).__init__(
            self.name, config, mode, meta
        )

    def parse_config(self):
        """
        """
        self.check_config()

        self.blip_graph = None
        self.merge_tree = None
        self.mcts = None

        self.parse_blip_graph()
        self.parse_merge_tree()
        self.parse_mcts()
    
    def check_config(self):
        if self.mode == 'evaluate_merge_tree':
            if "merge_tree" not in self.config.keys():
                self.logger.error(f"'merge_tree' section not specified in config!")
            if 'progress_bar' not in self.config['merge_tree'].keys():
                self.config['merge_tree']['progress_bar'] = True
            if 'rewrite_bar' not in self.config['merge_tree'].keys():
                self.config['merge_tree']['rewrite_bar'] = False

        if self.mode == 'mcts':
            if "mcts" not in self.config.keys():
                self.logger.error(f"'mcts' section not specified in config!")
    
    def parse_blip_graph(self):
        if self.mode != 'evaluate_merge_tree':
            return
        self.logger.info("configuring blip_graph.")
        merge_tree_config = self.config['merge_tree']
        if "blip_graph" not in merge_tree_config.keys():
            self.logger.error(f"blip_graph not specified in merge_tree config!")
        blip_graph_ckpt = merge_tree_config['blip_graph']
        if not os.path.isfile(blip_graph_ckpt):
            self.logger.error(f"specified blip_graph ckpt {blip_graph_ckpt} does not exist!")
        blip_graph_config = {
            'BlipGraph': torch.load(blip_graph_ckpt)['model_config']
        }
        self.blip_graph = ModelHandler(
            self.name + '_trained_blip_net',
            blip_graph_config,
            meta=self.meta
        ).model
        self.blip_graph.load_model(blip_graph_ckpt)
        self.blip_graph.set_device(self.device)

    def parse_merge_tree(self):
        pass

    def parse_mcts(self):
        pass

    def run_module(self):
        if   self.mode == 'optimize_blip_graph':  self.run_optimize_blip_graph()
        elif self.mode == 'construct_merge_tree': self.run_construct_merge_tree()
        elif self.mode == 'evaluate_merge_tree':  self.run_evaluate_merge_tree()
        elif self.mode == 'mcts':                 self.run_mcts()
        else: self.logger.warning(f"current mode {self.mode} not an available type!")

    def run_optimize_blip_graph(self):
        pass

    def run_construct_merge_tree(self):
        pass

    def run_evaluate_merge_tree(self):
        self.logger.info('running merge_tree evaluation.')
        self.blip_graph.eval()
        predictions = {
            classification: []
            for classification in self.blip_graph.config['classifications']
        }

        # loop over events
        if self.config['merge_tree']['progress_bar']:
            inference_loop = tqdm(
                enumerate(self.meta['loader'].inference_loader, 0), 
                total=len(self.meta['loader'].inference_loader), 
                leave=self.config['merge_tree']['rewrite_bar'],
                position=0,
                colour='green'
            )
        else:
            inference_loop = enumerate(self.meta['loader'].inference_loader, 0)
        
        with torch.no_grad():
            for ii, data in inference_loop:
                # grab event data
                event_positions = data.pos
                event_features = data.x
                merge_tree = data.merge_tree
                
                # loop over nodes in the tree
                event_loop = enumerate(merge_tree['clusters'][0], 0)

                for classification in predictions.keys():
                    predictions[classification].append([])

                for jj, cluster in event_loop:
                    # grab positions of cluster
                    cluster_indices = cluster.pre_order()
                    cluster_positions = event_positions[cluster_indices]
                    cluster_features = event_features[cluster_indices]

                    min_positions = torch.min(cluster_positions, dim=0)[0]
                    max_positions = torch.max(cluster_positions, dim=0)[0]
                    scale = max_positions - min_positions
                    scale[(scale == 0)] = max_positions[(scale == 0)]
                    cluster_positions = 2 * (cluster_positions - min_positions) / scale - 1

                    node_cluster = Data(
                        pos   = cluster_positions,
                        x     = cluster_features,
                        batch = torch.ones(len(cluster_positions), dtype=torch.long)
                    )
                    model_output = self.blip_graph(node_cluster)
                    print(model_output, ii, jj)
                    for kk, key in enumerate(model_output.keys()):
                        if key in predictions.keys():
                            predictions[key][ii].append([model_output[key].cpu().numpy()])

                # update progress bar
                if (self.config['merge_tree']['progress_bar'] == True):
                    inference_loop.set_description(f"Merge Tree Inference: Event [{ii+1}/{len(self.meta['loader'].inference_loader)}]")
                    inference_loop.set_postfix_str(f"nodes={len(merge_tree['clusters'][0])}")
        
        # save all predictions
        np.savez(
            f'blip_graph_outputs.npz',
            **predictions
        )

    def run_mcts(self):
        pass
        