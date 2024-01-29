"""
Generic module
"""
import os
import copy
import random
import numpy as np
from tqdm import tqdm

from blip.models import ModelHandler
from blip.module.generic_module import GenericModule
from blip.dataset.common import (
    ParticleLabel, 
    TopologyLabel,
    PhysicsMicroLabel,
    PhysicsMesoLabel,
    PhysicsMacroLabel
)


class EfficiencyPurity(GenericModule):
    """

    """
    def __init__(
        self,
        name:   str,
        config: dict = {},
        mode:   str = '',
        meta:   dict = {}
    ):
        self.name = name
        super(EfficiencyPurity, self).__init__(
            self.name, config, mode, meta
        )
        self.consumes = ['dataset', 'loader']
        self.produces = ['predictions']

    def parse_config(self):
        """
        """
        self.check_config()

        self.meta['model'] = None

        self.parse_model()

    def check_config(self):
        if "model" not in self.config.keys():
            self.logger.warning('"model" section not specified in config!')

    def parse_model(
        self,
        name:   str = ''
    ):
        """
        """
        if "model" not in self.config.keys():
            self.logger.warn("no model in config file.")
            return
        self.logger.info("configuring model.")
        model_config = self.config["model"]
        self.meta['model'] = ModelHandler(
            self.name + name,
            model_config,
            meta=self.meta
        )

    def run_module(self):
        if self.mode == "arrakis":
            self.run_arrakis()
        else:
            self.logger.warning(f"current mode {self.mode} not an available type!")

    def run_arrakis(self):
        inference_loop = tqdm(
            enumerate(self.meta['loader'].inference_loader, 0),
            total=len(self.meta['loader'].inference_loader),
            leave=False,
            position=0,
            colour='green'
        )
        for ii, data in inference_loop:
            predictions = self.meta['model'](data)
            track_ids = data.clusters["unique_particle_label"]
            unique_track_ids = np.unique(track_ids)

            # initialize efficiency and purity empty arrays
            efficiency = np.zeros(len(predictions))
            purity = np.zeros(len(predictions))
            # compute efficiency and purity per track
            for track_id in unique_track_ids:
                track_mask = (track_ids == track_id)
                track_predictions = predictions[track_mask] # need to check what is in here
                track_labels = data.clusters["particle_label"][track_mask] # need to check what is in here
                # compute efficiency
                track_efficiency = np.sum(track_predictions == track_labels) / len(track_predictions)
                # compute purity
                track_purity = np.sum(track_predictions == track_labels) / len(track_labels)
                # store precision and purity
                efficiency[track_mask] = track_efficiency
                purity[track_mask] = track_purity

            # i guess we need to store the output somewhere


