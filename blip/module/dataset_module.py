
"""
Dataset module code.
"""
from tqdm import tqdm

from blip.module import GenericModule
from blip.dataset.mssm import MSSM
from blip.dataset.blip import BlipDataset
from blip.dataset.vanilla import VanillaDataset
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

    def parse_config(self):
        """
        """
        self.check_config()

    def check_config(self):
        pass

    def run_module(self):
        self.logger.info("configuring dataset.")
        self.dataset = BlipDataset(
            self.name,
            self.config['dataset'],
            self.meta
        )
        self.meta['dataset'] = self.dataset

        # Configure the loader
        self.logger.info("configuring loader.")
        self.loader = Loader(
            self.name,
            self.config['loader'],
            self.meta
        )
        self.meta['loader'] = self.loader
