from ctypes import sizeof
import uproot
import os
import getpass
import numpy as np
import socket
import matplotlib.pyplot as plt
from scipy import stats as st
from datetime import datetime

from blip.utils.logger import Logger
from blip.dataset.common import *

class ArrakisND:
    def __init__(self,
        name:   str="arrakis_nd",
        config: dict={},
        meta:   dict={}
    ):
        self.name = name + '_dataset'
        self.config = config
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(name, output="both", file_mode="w")
        else:
            self.logger = Logger(name, file_mode="w")
        self.logger.info(f"constructing arrakis_nd dataset.")

        self.simulation_files = []
        self.output_folders = {}
    
    def parse_config(self):
        pass

