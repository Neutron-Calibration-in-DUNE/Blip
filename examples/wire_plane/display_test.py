"""

"""
from ctypes import sizeof
import uproot
import os
import numpy as np
import socket
import matplotlib.pyplot as plt
from scipy import stats as st
from datetime import datetime

from blip.utils.logger import Logger
from blip.dataset.common import *
from blip.dataset.blip import BlipDataset
from blip.utils.event_display import BlipDisplay


if __name__ == "__main__":
    blip_dataset = BlipDataset(
        "test",
        input_files=[
            "data/blip_simulation_0/point_cloud_view0_tpc0.npz"
        ],
        classes=["source", "shape", "particle"]
    )
    display = BlipDisplay(blip_dataset)
    display.show()