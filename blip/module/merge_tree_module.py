
"""
Generic model code.
"""
import warnings
from tqdm import tqdm

from blip.module.generic_module import GenericModule
from blip.topology.merge_tree import MergeTree

warnings.filterwarnings("ignore")

generic_config = {
    "no_params":    "no_values"
}


class MergeTreeModule(GenericModule):
    """
    Creates a merge tree from a pointcloud input:
    - pointCloud: a Euclidean point cloud of shape (num points) x (dimension). The merge tree
                    is generated from the Vietoris-Rips filtration of the point cloud
    Merge trees can be 'decorated' with higher-dimensional homological data by the 'fit_barcode' method.
    The result is a 'decorated merge tree'.
    """
    def __init__(
        self,
        name:   str,
        config: dict = {},
        mode:   str = '',
        meta:   dict = {}
    ):
        self.name = name
        super(MergeTreeModule, self).__init__(
            self.name, config, mode, meta
        )
        self.consumes = ['dataset']
        self.produces = ['merge_tree']

        self.parse_config()

    def parse_config(self):
        self.logger.info('setting up merge_tree')
        self.merge_tree = MergeTree()

    def run_module(self):
        self.logger.info('running merge_tree module.')
        """
        Set up progress bar.
        """
        if (progress_bar == True):
            inference_loop = tqdm(
                enumerate(inference_loader, 0), 
                total=len(list(inference_loader)), 
                # total=len(list(inference_indices)), 
                leave=rewrite_bar,
                colour='magenta'
            )
        else: 
            inference_loop = enumerate(inference_loader, 0)

        for ii, data in inference_loop:
            vietoris_rips, tree = self.merge_tree.create_merge_tree(data)
