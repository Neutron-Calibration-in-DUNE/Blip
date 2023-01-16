"""
Training loop for a BLIP model   
"""
# blip imports
from blip.dataset.arrakis import Arrakis
from blip.dataset.blip import BlipDataset
from blip.utils.display import BlipDisplay

import numpy as np
import os
import shutil
from datetime import datetime


if __name__ == "__main__":

    """
    Now we load our dataset files to be displayed.
    """
    blip_display = BlipDisplay(
        name = "blip_display",
        input_files={
            'view0': '../data/point_cloud_view0.npz',
            'view1': '../data/point_cloud_view1.npz',
            'view2': '../data/point_cloud_view2.npz'
        },
    )

    blip_display.create_class_gif(
        'view0', 'gamma', 100
    )
    blip_display.create_class_gif(
        'view0', 'neutron', 100
    )
    blip_display.create_class_gif(
        'view0', 'ar39', 100
    )