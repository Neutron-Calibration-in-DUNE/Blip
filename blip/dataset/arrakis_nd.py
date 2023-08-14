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
            self.logger = Logger(name, level='warning', file_mode="w")
        self.logger.info(f"constructing arrakis_nd dataset.")

        self.simulation_files = []
        self.output_folders = {}

        """
        2x2 channel mappings for different
        TPCs.
        """
        self.nd_2x2_tpc_positions = {
            "tpc0": [[-376.8501, -366.8851],[0., 607.49875],[-0.49375, 231.16625]],
            "tpc1": [[-359.2651,   -0.1651],[0., 607.49875],[-0.49375, 231.16625]],
            "tpc2": [[0.1651, 359.2651],    [0., 607.49875],[-0.49375, 231.16625]],
            "tpc3": [[366.8851, 376.8501],  [0., 607.49875],[-0.49375, 231.16625]],
            "tpc4": [[-376.8501, -366.8851],[0., 607.49875],[231.56625, 463.22625]],
            "tpc5": [[-359.2651, -0.1651],  [0., 607.49875],[231.56625, 463.22625]],
            "tpc6": [[0.1651, 359.2651],    [0., 607.49875],[231.56625, 463.22625]],
            "tpc7": [[366.8851, 376.8501],  [0., 607.49875],[231.56625, 463.22625]],
            "tpc8": [[-376.8501, -366.8851],[0., 607.49875],[463.62625, 695.28625]],
            "tpc9": [[-359.2651, -0.1651],  [0., 607.49875],[463.62625, 695.28625]],
            "tpc10": [[0.1651, 359.2651],   [0., 607.49875],[463.62625, 695.28625]],
            "tpc11": [[366.8851, 376.8501], [0., 607.49875],[463.62625, 695.28625]],
        }
    
    def parse_config(self):
        pass

