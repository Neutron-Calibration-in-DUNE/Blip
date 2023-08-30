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
from subprocess import call

"""
To run the bokeh server in a jupyter notebook, 
import the create_server function from this
script and run it in the notebook.

To run the bokeh server from a terminal + browser,
run the command:
    > bokeh server --show blip_http_server.py
"""
from bokeh.io import output_notebook, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.io import curdoc

from blip.utils.logger import Logger, default_logger
from blip.utils.config import ConfigParser

from blip.dataset.arrakis import Arrakis
from blip.dataset.arrakis_nd import ArrakisND
from blip.dataset.blip import BlipDataset
from blip.utils.loader import Loader
from blip.module import ModuleHandler
from blip.module.common import module_types
from blip.utils.event_display import BlipDisplay

def run():
    """
    BLIP DISPLAY main program.
    """
    parser = argparse.ArgumentParser(
        prog='BLIP Display Runner',
        description='This program constructs a BLIP display as '+
            'an html server.',
        epilog='...'
    )
    parser.add_argument(
        '-url', dest='url', type=str,
        default='localhost:8888',
        help='url specification for a BLIP display.'
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

    args = parser.parse_args()

    os.environ['LOCAL_SCRATCH'] = args.local_scratch
    os.environ['LOCAL_BLIP'] = args.local_blip
    os.environ['LOCAL_DATA'] = args.local_data
    
    p = os.path.realpath(__file__)
    prefix, _ = os.path.split(p)
    bokeh_server_file = os.path.join(prefix, "../utils/event_display/blip_http_server.py")
    call(["bokeh", "serve", "--show", bokeh_server_file])

if __name__ == "__main__":
    run()