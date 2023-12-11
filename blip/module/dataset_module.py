
"""
Dataset module code.
"""
from tqdm import tqdm

from blip.module import GenericModule
from blip.dataset.common import blip_datasets, mssm_datasets
from blip.dataset.mssm_dataset import MSSMDataset
from blip.dataset.blip_dataset import BlipDataset
from blip.dataset.vanilla_dataset import VanillaDataset
from blip.utils.loader import Loader

dataset_config = {
}


class DatasetModule(GenericModule):
    """
    The module class helps to organize meta data and objects related to different tasks
    and execute those tasks based on a configuration file.  The spirit of the 'Module' class
    is to mimic some of the functionality of LArSoft, e.g. where you can specify a chain
    of tasks to be completed, the ability to have nested config files where default parameters
    can be overwritten.

    The Dataset specific module runs in several different modes,
    """
    def __init__(
        self,
        name:   str,
        config: dict = {},
        mode:   str = '',
        meta:   dict = {}
    ):
        self.name = name
        super(DatasetModule, self).__init__(
            self.name, config, mode, meta
        )
        self.consumes = [None]
        self.produces = ['dataset', 'loader']

        self.all_datasets = blip_datasets + mssm_datasets

    def parse_config(self):
        """
        """
        self.check_config()

    def check_config(self):
        if 'dataset_type' not in self.config['dataset'].keys():
            self.logger.error('dataset_type not specified in config!')
        if self.config['dataset']['dataset_type'] not in self.all_datasets:
            self.logger.error(
                f"specified dataset_type {self.config['dataset']['dataset_type']} not an allowed type!"
            )
        if 'dataset_params' in self.config['dataset'].keys():
            if self.config['dataset']['dataset_params'] == "":
                self.logger.info('setting dataset_params location to /local_data/dataset.params')
                self.config['dataset']['dataset_params'] = '/local_data/dataset.params'

    def run_module(self):
        if self.mode == "dataset_prep":
            self.run_dataset_prep()
        elif self.mode == "dataset_load":
            self.run_dataset_load()
        else:
            self.logger.error(f'specified module_mode {self.mode} not an allowed type!')

    def run_dataset_prep(self):
        self.logger.info('prepping dataset')
        if self.config['dataset']['dataset_type'] in blip_datasets:
            self.dataset = BlipDataset(
                self.name,
                self.config['dataset'],
                self.meta
            )
            self.meta['dataset'] = self.dataset
        elif self.config['dataset']['dataset_type'] in mssm_datasets:
            self.dataset = MSSMDataset(
                self.name,
                self.config['dataset'],
                self.meta
            )
        else:
            self.logger.error(
                f"specified dataset_type {self.config['dataset']['dataset_type']} not an allowed type!"
            )

    def run_dataset_load(self):
        self.logger.info("configuring dataset.")
        if self.config['dataset']['dataset_type'] in blip_datasets:
            self.dataset = BlipDataset(
                self.name,
                self.config['dataset'],
                self.meta
            )
            self.meta['dataset'] = self.dataset
        elif self.config['dataset']['dataset_type'] in mssm_datasets:
            self.dataset = MSSMDataset(
                self.name,
                self.config['dataset'],
                self.meta
            )
        else:
            self.logger.error(
                f"specified dataset_type {self.config['dataset']['dataset_type']} not an allowed type!"
            )
        # Configure the loader
        self.logger.info("configuring loader.")
        self.loader = Loader(
            self.name,
            self.config['loader'],
            self.meta
        )
        self.meta['loader'] = self.loader
