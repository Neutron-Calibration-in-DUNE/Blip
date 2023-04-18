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

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F

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
        batch_size=2,
        test_split=0.0,
        test_seed=100,
        validation_split=0.2,
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
            'method':   PointNet,
            'config': {
                'input_dimension':  3,
                'num_embedding':    2,
                'embedding_mlp_layers': [
                    [10, 10],
                    [10, 10]
                ],
                'number_of_neighbors':  20,
                'aggregation_operators': [
                    'max', 'max'
                ],
                'linear_output':    10,
                'mlp_output_layers': [10, 25, 10],
            },
        }
    }

    blip_model = SetAbstraction("blip", set_abstraction_config)
    device = torch.device("cuda")
    blip_model.device = device
    optimizer = torch.optim.Adam(
        blip_model.parameters(),
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.001
    )
    criterion = F.nll_loss
    training_loop = enumerate(blip_loader.train_loader, 0)
    
    for ii, data in training_loop:
        optimizer.zero_grad()
        output = blip_model(data, None)
        print(output)
        loss = criterion(data.category, output['sampled_embedding'])
        loss.backward()
        optimizer.step()
        