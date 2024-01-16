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


class nArInelasticModule(GenericModule):
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
        super(nArInelasticModule, self).__init__(
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
            if PhysicsMacroLabel.CCNue.value in data.category[:, 4]:
                print(f"cc_nu_e: {ii}")
            elif PhysicsMacroLabel.CCNuMu.value in data.category[:, 4]:
                print(f"cc_nu_mu: {ii}")
            elif PhysicsMacroLabel.NC.value in data.category[:, 4]:
                print(f"cc_nc: {ii}")
