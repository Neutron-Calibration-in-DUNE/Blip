"""
Blip main program
"""
import torch
import os
import csv
import getpass
from torch import nn
import torch.nn.functional as F
from time import time
from datetime import datetime
import argparse
os.environ["TQDM_NOTEBOOK"] = "false"

from blip.utils.logger import Logger, default_logger
from blip.utils.config import ConfigParser

from blip.dataset.arrakis import Arrakis
from blip.dataset.arrakis_nd import ArrakisND
from blip.dataset.mssm import MSSM
from blip.dataset.blip import BlipDataset
from blip.dataset.vanilla import VanillaDataset
from blip.utils.loader import Loader
from blip.module import ModuleHandler
from blip.module.common import module_types

from blip.programs.wrapper import parse_command_line_config

def run():
    """
    BLIP main program.
    """
    parser = argparse.ArgumentParser(
        prog='BLIP Module Runner',
        description='This program constructs a BLIP module '+
            'from a config file, and then runs the set of modules ' +
            'in the configuration.',
        epilog='...'
    )
    parser.add_argument(
        'config_file', metavar='<str>.yml', type=str,
        help='config file specification for a BLIP module.'
    )
    parser.add_argument(
        '-n', dest='name', default='blip',
        help='name for this run (default "blip").'
    )
    parser.add_argument(
        '-scratch', dest='local_scratch', default='/local_scratch',
        help='location for the local scratch directory.'
    )
    parser.add_argument(
        '-blip', dest='local_blip', default='/local_blip',
        help='location for the local blip directory.'
    )
    parser.add_argument(
        '-data', dest='local_data', default='/local_data',
        help='location for the local data directory.'
    )
    parser.add_argument(
        '-anomaly', dest='anomaly', default=False,
        help='enable anomaly detection in pytorch'
    )
    args = parser.parse_args()
    meta, module_handler = parse_command_line_config(args)
    module_handler.run_modules()

if __name__ == "__main__":
    run()