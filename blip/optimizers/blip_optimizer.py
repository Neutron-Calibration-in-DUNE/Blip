"""
Optimizers for blip.
"""
import torch.optim as optim
import torch.nn as nn

from blip.utils.logger import BlipError
from blip.utils.utils import profiler


class BlipOptimizer:
    """
    A standard optimizer for pytorch models.
    """
    def __init__(
        self,
        config: dict = {},
        meta:   dict = {}
    ):
        self.config = config
        self.meta = meta

        self.parse_config()

    def parse_config(self):
        if "model" not in self.meta:
            raise BlipError('no model specified in meta!')
        # set learning rate and momentum
        if "learning_rate" not in self.config.keys():
            self.config["learning_rate"] = 0.01
        self.learning_rate = self.config["learning_rate"]

        if self.config["optimizer_type"] == "Adam":
            if "betas" not in self.config.keys():
                self.config["betas"] = [0.9, 0.95]
            if "epsilon" not in self.config.keys():
                self.config["epsilon"] = 1e-08
            if "momentum" not in self.config.keys():
                self.config["momentum"] = 0.9
            if "weight_decay" not in self.config.keys():
                self.config["weight_decay"] = 0.001
            if "fused" not in self.config.keys():
                self.config["fused"] = False
            self.optimizer = optim.Adam(
                self.meta['model'].parameters(),
                lr=self.learning_rate,
                betas=self.config["betas"],
                eps=float(self.config["epsilon"]),
                weight_decay=self.config["weight_decay"],
                fused=self.config["fused"]
            )
        else:
            raise BlipError(
                f"specified optimizer_type: {self.config['optimizer_type']} not allowed!"
            )

        if "max_norm" not in self.config.keys():
            self.config["max_norm"] = 1.0
        self.max_norm = self.config["max_norm"]

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self):
        if self.max_norm:
            nn.utils.clip_grad_norm_(self.meta['model'].model.parameters(), max_norm=self.max_norm)
        return self.optimizer.step()

    @profiler
    def report_tensorboard(self, iterations, train_type):
        """Report learning rate to tensorboard"""
        self.meta['tensorboard'].add_scalar(
            f'Learning rate ({train_type})',
            self.optimizer.param_groups[0]['lr'],
            iterations
        )
