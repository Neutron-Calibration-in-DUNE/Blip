"""
Training loop for a BLIP model   
"""
# blip imports
from blip.dataset.blip import BlipDataset
from blip.utils.loader import Loader
from blip.losses import LossHandler
from blip.optimizers import Optimizer
from blip.metrics import MetricHandler
from blip.trainer import Trainer
from blip.utils.callbacks import CallbackHandler
from blip.utils.utils import get_files, save_model
from blip.models import PointNetClassification 

import torch_geometric.transforms as T

if __name__ == "__main__":

    # clean up directories first
    save_model()

    """
    Now we load our dataset as a torch dataset (blipDataset),
    and then feed that into a dataloader.
    """
    blip_dataset = BlipDataset(
        name = "wire_plane_test_dataset",
        input_file='data/wire_plane_test/point_cloud_view0.npz',
        root="."
    )
    blip_loader = Loader(
        blip_dataset, 
        batch_size=4,
        test_split=0.3,
        test_seed=100,
        validation_split=0.3,
        validation_seed=100,
        num_workers=4
    )