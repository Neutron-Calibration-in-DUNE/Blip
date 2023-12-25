
"""
Generic model code.
"""
import torch
import os
import csv
import getpass
from torch import nn
import numpy as np
from datetime import datetime
from collections import OrderedDict

from blip.utils.logger import Logger

generic_config = {
    "no_params":    "no_values"
}


class GenericModel(nn.Module):
    """
    Wrapper of torch nn.Module that generates a GenericModel
    """
    def __init__(
        self,
        name:   str,
        config: dict = generic_config,
        meta:   dict = {}
    ):
        super(GenericModel, self).__init__()
        self.name = name
        self.config = config
        self.logger = Logger(self.name, file_mode='w')
        self.logger.info("configuring model.")

        # forward view maps
        self.forward_views = {}
        self.forward_view_map = {}

        self.input_shape = None
        self.output_shape = None

        # device for the model
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        self.to(self.device)

    def set_device(
        self,
        device
    ):
        self.device = device
        self.to(device)

    def forward_hook(self, m, i, o):
        """
        A forward hook for a particular module.
        It assigns the output to the views dictionary.
        """
        self.forward_views[self.forward_view_map[m]] = o

    def register_forward_hooks(self):
        """
        This function registers all forward hooks for the modules
        in ModuleDict.
        """
        for name, module in self._modules.items():
            if isinstance(module, nn.ModuleDict):
                for name, layer in module.items():
                    self.forward_view_map[layer] = name
                    layer.register_forward_hook(self.forward_hook)

    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.

        """
        self.logger.info(f"Attempting to build GenericModel architecture using config: {self.config}")

        _model_dict = OrderedDict()
        self.model_dict = nn.ModuleDict(_model_dict)

        # record the info
        self.logger.info(
            "Constructed GenericModel with dictionaries:"
        )

    def forward(self, x):
        self.logger.error('"forward" not implemented in Model!')

    def save_model(
        self,
        flag:   str = ''
    ):
        # save meta information
        if not os.path.isdir(f"{self.meta['local_scratch']}/.tmp/models/{self.name}/"):
            os.makedirs(f"{self.meta['local_scratch']}/.tmp/models/{self.name}/")
        output = f"{self.meta['local_scratch']}/.tmp/models/{self.name}/" + self.name
        if flag != '':
            output += "_" + flag
        meta_info = {
            'name':     self.name,
            'date':     datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
            'user':     getpass.getuser(),
            'user_id':  os.getuid()
        }
        system_info = self.logger.get_system_info()
        if len(system_info) > 0:
            for item in system_info:
                meta_info[item] = system_info[item]
        meta_info['model_config'] = self.config
        meta_info['num_parameters'] = self.total_parameters()
        meta_info['state_dict'] = self.state_dict()
        # save config
        config = [[item, self.config[item]] for item in self.config]
        with open(output + ".config", "w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows(config)
        np.savez(
            f'{output}_meta_info.npz',
            meta_info=meta_info
        )
        # save parameters
        torch.save(
            {
                'model_state_dict': self.state_dict(),
                'model_config':     self.config,
                'meta_info':        meta_info,
            },
            output + "_params.ckpt"
        )

    def total_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_checkpoint(
        self,
        checkpoint_file:    str = ''
    ):
        pass

    def load_model(
        self,
        model_file:   str = ''
    ):
        self.logger.info(f"attempting to load model checkpoint from file {model_file}.")
        try:
            checkpoint = torch.load(model_file)
            self.config = checkpoint['model_config']
            self.construct_model()
            # register hooks
            self.register_forward_hooks()
            self.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            self.logger.error(f"unable to load model file {model_file}: {e}.")
            raise ValueError(f"unable to load model file {model_file}: {e}.")
        self.logger.info("successfully loaded model checkpoint.")
