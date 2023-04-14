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
from blip.utils.sampling import *
from blip.utils.grouping import *
from blip.utils.callbacks import CallbackHandler
from blip.utils.utils import get_files, save_model
from blip.models import SetAbstraction, PointNet

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
        batch_size=1,
        test_split=0.0,
        test_seed=100,
        validation_split=0.0,
        validation_seed=100,
        num_workers=4
    )
    set_abstraction_config = {
        'sampling': {
            'method':   farthest_point_sampling,
            'number_of_centroids':  None,
        },
        'grouping': {
            'method':       query_ball_point,
            'radii':                None,
            'number_of_samples':    None,
        },
        'pointnet': {
            'method':   PointNet(),
        },
    }

    blip_model = SetAbstraction("blip", set_abstraction_config)

    training_loop = enumerate(blip_loader.train_loader, 0)
    for ii, data in training_loop:
        print(data.pos)
        print(data.batch)
        print(data.category)
        output = blip_model(data.pos, None)
        print(output)