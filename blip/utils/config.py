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
        self.config_file = config_file
        with open(self.config_file, 'r') as file:
            self.data = yaml.safe_load(file)