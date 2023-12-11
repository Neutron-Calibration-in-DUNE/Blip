"""
Container for generic losses
"""
import os
import importlib.util
import sys
import inspect
import torch

from blip.utils.logger import Logger
from blip.losses import GenericLoss
from blip.utils.utils import get_method_arguments

class LossHandler:
    """
    """
    def __init__(self,
        name:    str,
        config:  dict={},
        losses:  list=[],
        use_sample_weights: bool=False,
        meta:    dict={}
    ):
        self.name = name + '_loss_handler'
        self.use_sample_weights = use_sample_weights
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(self.name, output="both", file_mode="w")
        else:
            self.logger = Logger(self.name, level='warning', file_mode="w")
            
        self.losses = {}
        self.batch_loss = {}

        if bool(config) and len(losses) != 0:
            self.logger.error(
                f"handler received both a config and a list of losses! " + 
                f"The user should only provide one or the other!")
        elif bool(config):
            self.set_config(config)
        else:
            if len(losses) == 0:
                self.logger.error(f"handler received neither a config or losses!")
            self.losses = {
                loss.name: loss 
                for loss in losses
            }
            self.batch_loss = {
                loss.name: torch.empty(size=(0,1), dtype=torch.float, device=self.device) 
                for loss in losses
            }

    def set_config(self, config):
        self.config = config
        self.process_config()
    
    def collect_loss_functions(self):
        self.available_criterions = {}
        self.criterion_files = [
            os.path.dirname(__file__) + '/' + file 
            for file in os.listdir(path=os.path.dirname(__file__))
        ]
        self.criterion_files.extend(self.meta['local_blip_files'])
        for criterion_file in self.criterion_files:
            if (
                ("__init__.py" in criterion_file) or 
                ("__pycache__.py" in criterion_file) or 
                ("generic_criterion.py" in criterion_file) or 
                ("__pycache__" in criterion_file) or
                (".py" not in criterion_file)
            ):
                continue
            try:
                self.load_loss_function(criterion_file)
            except:
                self.logger.warn(f'problem loading criterion from file: {criterion_file}')
    
    def load_loss_function(self,
        criterion_file: str
    ):
        spec = importlib.util.spec_from_file_location(
            f'{criterion_file.removesuffix(".py")}.name', 
            criterion_file
        )
        custom_loss_file = importlib.util.module_from_spec(spec)
        sys.modules[f'{criterion_file.removesuffix(".py")}.name'] = custom_loss_file
        spec.loader.exec_module(custom_loss_file)
        for name, obj in inspect.getmembers(sys.modules[f'{criterion_file.removesuffix(".py")}.name']):
            if inspect.isclass(obj):
                custom_class = getattr(custom_loss_file, name)
                if issubclass(custom_class, GenericLoss):
                    self.available_criterions[name] = custom_class

    def process_config(self):
        # list of available criterions
        self.collect_loss_functions()
        # check config
        if "custom_loss_file" in self.config.keys():
            if os.path.isfile(self.config["custom_loss_file"]):
                try:
                    self.load_loss_function(self.config["custom_loss_file"])
                    self.logger.info(f'added custom loss function from file {self.config["custom_loss_file"]}.')
                except:
                    self.logger.error(
                        f'loading classes from file {self.config["custom_loss_file"]} failed!'
                    )
            else:
                self.logger.error(f'custom_loss_file {self.config["custom_loss_file"]} not found!')
        # process loss functions
        for item in self.config.keys():
            if item == "custom_loss_file":
                continue
            # check that loss function exists
            if item not in self.available_criterions.keys():
                self.logger.error(
                    f"specified loss function '{item}' is not an available type! " + 
                    f"Available types:\n{self.available_criterions.keys()}"
                )
            # check that function arguments are provided
            argdict = get_method_arguments(self.available_criterions[item])
            for value in self.config[item].keys():
                if value not in argdict.keys():
                    self.logger.error(
                        f"specified loss function value '{item}:{value}' " + 
                        f"not a constructor parameter for '{item}'! " + 
                        f"Constructor parameters:\n{argdict}"
                    )
            for value in argdict.keys():
                if argdict[value] == None:
                    if value not in self.config[item].keys():
                        self.logger.error(
                            f"required input parameters '{item}:{value}' "+
                            f"not specified! Constructor parameters:\n{argdict}"
                        )
        self.losses = {}
        self.batch_loss = {}
        for item in self.config.keys():
            if item == "custom_loss_file":
                continue
            if item in self.losses:
                self.logger.warn(f'duplicate loss specified in config! Attempting to arrange by target_type.')
                if self.losses[item].target_type == self.config[item]['target_type']:
                    self.logger.error(f'duplicate losses with the same target_type in config!')
            self.losses[item] = self.available_criterions[item](**self.config[item], meta=self.meta)
            self.batch_loss[item] = torch.empty(size=(0,1), dtype=torch.float, device=self.device)
            self.logger.info(f'added loss function "{item}" to LossHandler.')

    def set_device(self,
        device
    ):  
        self.logger.info(f'setting device to "{device}".')
        for name, loss in self.losses.items():
            loss.set_device(device)
            self.batch_loss[name] = torch.empty(size=(0,1), dtype=torch.float, device=self.device)
        self.device = device

    def reset_batch(self):  
        for name, loss in self.losses.items():
            self.batch_loss[name] = torch.empty(size=(0,1), dtype=torch.float, device=self.device)
            loss.reset_batch()

    def add_loss(self,
        loss:   GenericLoss
    ):
        if issubclass(type(loss), GenericLoss):
            self.logger.info(f'added loss function "{loss}" to LossHandler.')
            self.losses[loss.name] = loss
        else:
            self.logger.error(
                f'specified loss {loss} is not a child of "GenericLoss"!' + 
                f' Only loss functions which inherit from GenericLoss can' +
                f' be used by the LossHandler in BLIP.'
            )

    def remove_loss(
        self,
        loss:   str
    ):
        if loss in self.losses.keys():
            self.losses.pop(loss)
            self.logger.info(f'removed {loss} from losses.')
        if loss in self.batch_loss.keys():
            self.batch_loss.pop(loss)

    def loss(self,
        outputs,
        data,
    ):
        batch_loss = 0
        for name, loss in self.losses.items():
            temp_loss = loss.loss(outputs, data)
            self.batch_loss[name] = torch.cat(
                (self.batch_loss[name], torch.tensor([[temp_loss]], device=self.device)), dim=0
            )
            batch_loss += temp_loss
        return batch_loss