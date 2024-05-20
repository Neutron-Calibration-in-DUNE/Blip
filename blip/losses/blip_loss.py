"""
Container for generic losses
"""
import os
import importlib.util
import sys
import inspect
import torch

from blip.utils.logger import BlipError
from blip.losses.generic_loss import GenericLoss
from blip.utils.utils import get_method_arguments, profiler


class BlipLoss:
    """
    """
    def __init__(
        self,
        config:  dict = {},
        meta:    dict = {}
    ):

        self.config = config
        self.meta = meta

        if "device" in self.meta:
            self.device = self.meta["device"]

        self.parse_config()

    @profiler
    def collect_losses(self):
        # Ensure only Python files that are not standard special files are considered
        loss_dir = os.path.dirname(__file__)
        self.available_losses = {}
        for filename in os.listdir(loss_dir):
            if filename.endswith(".py") and filename not in {
                "__init__.py", 
                "__pycache__.py",
                "generic_loss.py"
            }:
                loss_path = os.path.join(loss_dir, filename)
                loss_name = filename[:-3]  # Strip .py from filename to get module name
                try:
                    self.load_loss(loss_path, loss_name)
                except Exception as e:
                    raise BlipError(f"Problem loading loss from {loss_path}: {str(e)}")

    @profiler
    def load_loss(self, loss_path: str, loss_name: str):
        full_module_name = f"blip.losses.{loss_name}"
        if full_module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(full_module_name, loss_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[full_module_name] = module  # Register the loaded module
        else:
            module = sys.modules[full_module_name]

        # Register classes
        self.register_classes(module, full_module_name)

    @profiler
    def register_classes(self, module, full_module_name):
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, GenericLoss) and obj.__module__ == full_module_name:
                self.available_losses[name] = obj

    def parse_config(self):
        # list of available criterions
        self.collect_losses()
        # check config
        if "custom_loss_file" in self.config.keys():
            if os.path.isfile(self.config["custom_loss_file"]):
                try:
                    self.load_loss(self.config["custom_loss_file"])
                except Exception:
                    raise BlipError(
                        f'loading classes from file {self.config["custom_loss_file"]} failed!'
                    )
            else:
                raise BlipError(f'custom_loss_file {self.config["custom_loss_file"]} not found!')
        # process loss functions
        for item in self.config.keys():
            if item == "custom_loss_file":
                continue
            # check that loss function exists
            if item not in self.available_losses.keys():
                raise BlipError(
                    f"specified loss function '{item}' is not an available type! " +
                    f"Available types:\n{self.available_losses.keys()}"
                )
        self.losses = {}
        self.batch_loss = {}
        self.batch_iteration = {}
        for item in self.config.keys():
            if item == "custom_loss_file":
                continue
            if self.config[item] is not None:
                self.losses[item] = self.available_losses[item](**self.config[item], meta=self.meta)
            else:
                self.losses[item] = self.available_losses[item](meta=self.meta)
            self.batch_loss[item] = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)
            self.batch_iteration[item] = []

    @profiler
    def report_tensorboard(
        self,
        iterations,
        train_type
    ):
        for name, loss in self.losses.items():
            self.meta['tensorboard'].add_scalar(
                f"Avg loss: {name} ({train_type})",
                torch.mean(self.batch_loss[name]),
                iterations
            )
            for ii, iteration in enumerate(self.batch_iteration[name]):
                self.meta['tensorboard'].add_scalar(
                    f"{name} ({train_type})",
                    self.batch_loss[name][ii],
                    iteration
                )
        self.reset_batch()

    def set_device(
        self,
        device
    ):
        for name, loss in self.losses.items():
            loss.set_device(device)
            self.batch_loss[name] = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)
            self.batch_iteration[name] = []
        self.device = device

    def reset_batch(self):
        for name, loss in self.losses.items():
            self.batch_loss[name] = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)
            self.batch_iteration[name] = []
            loss.reset_batch()

    def add_loss(
        self,
        loss:   GenericLoss
    ):
        if issubclass(type(loss), GenericLoss):
            self.losses[loss.name] = loss
        else:
            raise BlipError(
                f'specified loss {loss} is not a child of "GenericLoss"!' +
                ' Only loss functions which inherit from GenericLoss can' +
                ' be used by the LossHandler in BLIP.'
            )

    def remove_loss(
        self,
        loss:   str
    ):
        if loss in self.losses.keys():
            self.losses.pop(loss)
        if loss in self.batch_loss.keys():
            self.batch_loss.pop(loss)
            self.batch_iteration.pop(loss)

    def loss(
        self,
        data,
        iteration
    ):
        batch_loss = 0
        for name, loss in self.losses.items():
            temp_loss = loss.loss(data)
            self.batch_loss[name] = torch.cat(
                (self.batch_loss[name], torch.tensor([[temp_loss]], device=self.device)), dim=0
            )
            self.batch_iteration[name].append(iteration)
            batch_loss += temp_loss
        return batch_loss
