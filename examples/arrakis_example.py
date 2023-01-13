"""
Training loop for a BLIP model   
"""
# blip imports
from blip.dataset.arrakis import Arrakis

import numpy as np
import os
import shutil
from datetime import datetime


if __name__ == "__main__":

    arrakis_dataset = Arrakis(
        "../../ArrakisEventDisplay/data/multiple_neutron_arrakis2.root"
    )
    arrakis_dataset.generate_training_data()