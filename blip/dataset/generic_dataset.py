import os

from torch_geometric.data import Data, InMemoryDataset

from blip.utils.logger import Logger
from blip.dataset.common import *

generic_config = {}


class GenericDataset(InMemoryDataset):
    """
    """
    def __init__(
        self,
        name:   str = 'generic',
        config: dict = generic_config,
        meta:   dict = {}
    ):
        self.name = name + '_dataset'
        self.config = config
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(self.name, output="both",   file_mode="w")
        else:
            self.logger = Logger(self.name, level='warning', file_mode="w")

        self.number_of_events = 0

        self.process_generic_config()
        self.process_config()
        
        InMemoryDataset.__init__(
            self,
            self.root,
            self.transform,
            self.pre_transform,
            self.pre_filter,
            log=False
        )
    
    def process_generic_config(self):
        # set up "root" directory.  this is mainly for processed data.
        if "root" in self.config.keys():
            if os.path.isdir(self.config['root']):
                self.root = self.config['root']
            else:
                self.logger.warn(
                    f'specified root directory {self.config["root"]} doesnt exist. attempting to create directory'
                )
                try:
                    os.makdirs(self.config['root'])
                except:
                    self.logger.warn(
                        f'attempt at making directory {self.config["root"]} failed.  setting root to /local_scratch/'
                    )
                    self.root = self.meta['local_scratch']
        else:
            self.root = self.meta['local_scratch']
        self.logger.info('set "root" directory to {self.root}')

        # set skip_processing
        if "skip_processing" in self.config.keys():
            if not isinstance(self.config["skip_processing"], bool):
                self.logger.error(f'skip_processing set to {self.config["skip_processing"]}, but should be a bool!')
            else:
                self.skip_processing = self.config["skip_processing"]

        if self.skip_processing:
            if os.path.isdir(self.root + '/processed/'):
                for path in os.listdir(self.root + '/processed/'):
                    if os.path.isfile(os.path.join('processed/', path)):
                        self.number_of_events += 1
            self.logger.info(f'found {self.number_of_events} processed files.')
        
        # set transform
        if "transform" in self.config.keys():
            self.transform = self.config["transform"]
        else:
            self.transform = None

        # set pre_transform
        if "pre_transform" in self.config.keys():
            self.pre_transform = self.config["pre_transform"]
        else:
            self.pre_transform = None

        # set pre_filter
        if "pre_filter" in self.config.keys():
            self.pre_filter = self.config["pre_filter"]
        else:
            self.pre_filter = None
        

    def process_config(self):
        self.logger.error('"process_config" function not implemented in Dataset!')

    def len(self):
        self.logger.error('"len" function not implemented in Dataset!')

    def get(self, idx):
        self.logger.error('"get" function not implemented in Dataset!')

    def process(self):
        self.logger.error('"process" function not implement in Dataset!')