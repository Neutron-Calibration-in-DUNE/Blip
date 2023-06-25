"""
Tools for parsing config files
"""
import numpy as np
from matplotlib import pyplot as plt
import os
import yaml
from yaml import Loader, Dumper

from blip.utils.logger import Logger

class ConfigParser:
    """
    """
    def __init__(self, 
        config_file:    str,
    ):
        self.logger = Logger('config_parser', output="both", file_mode='w')
        self.logger.info(f"setting up config file.")
        self.config_file = config_file
        self.nested_config_files = [config_file]
        self.nested_configs = []
        self.data = {}
        self.parse_config()

    def parse_config(self): 
        self.collect_nested_configs(self.config_file)
        self.nested_config_files.reverse()
        self.nested_configs.reverse()
        for config in self.nested_configs:
            self.data.update(config)
        self.logger.info(
            f'parsed {len(self.nested_configs)} files with the heirarchy: ' + 
            f'{self.nested_config_files}.'
        )

    def collect_nested_configs(self,
        config_file
    ):
        with open(config_file, 'r') as file:
            temp_data = yaml.safe_load(file)
            self.nested_configs.append(temp_data)
        if 'load_config' in temp_data.keys():
            self.nested_config_files.append(temp_data['load_config'])
            self.collect_nested_configs(temp_data['load_config'])