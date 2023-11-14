"""
Optimizers for blip.
"""
import torch.optim as optim

from blip.utils.logger import Logger
from blip.models.generic_model import GenericModel

class Optimizer:
    """
    A standard optimizer for pytorch models.
    """
    def __init__(self,
        name:   str='default',
        config: dict={},
        model:  GenericModel=None
    ):
        self.name = name + "_optimizer"
        self.logger = Logger(self.name, file_mode='w')
        self.config = config
        if model == None:
            self.logger.warn(f"no model given to optimizer!")
        self.model = model

        self.parse_config()

    def parse_config(self):
        # set learning rate and momentum
        if "learning_rate" not in self.config.keys():
            self.logger.warn("no learning_rate specified in config! Setting to 0.01")
            self.config["learning_rate"] = 0.01
        self.learning_rate = self.config["learning_rate"]

        if self.config["optimizer_type"] == "Adam":
            self.logger.info(f"setting optimizer_type to 'Adam'.")
            self.logger.info(f"learning rate set to {self.learning_rate}")
            if "betas" not in self.config.keys():
                self.logger.warn("no 'betas' specified in config! Setting to '[0.9, 0.999]'.")
                self.config["betas"] = [0.9, 0.999]
            self.logger.info(f"betas set to {self.config['betas']}")
            if "epsilon" not in self.config.keys():
                self.logger.warn("no 'epsilon' specified in config! Setting to '1e-08'.")
                self.config["epsilon"] = 1e-08
            self.logger.info(f"epsilon set to {self.config['epsilon']}")
            if "momentum" not in self.config.keys():
                self.logger.warn("no 'momentum' specified in config! Setting to '0.9'.")
                self.config["momentum"] = 0.9
            self.logger.info(f"momentum value set to {self.config['momentum']}")
            if "weight_decay" not in self.config.keys():
                self.logger.warn("no 'weight_decay' specified in config! Setting to '0.001'.")
                self.config["weight_decay"] = 0.001
            self.logger.info(f"weight decay set to {self.config['weight_decay']}")
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=self.config["betas"],
                eps=float(self.config["epsilon"]),
                weight_decay=self.config["weight_decay"]
            )
        else:
            self.logger.error(
                f"specified optimizer_type: {self.config['optimizer_type']} not allowed!"
            )
        
    def zero_grad(self):
        return self.optimizer.zero_grad()
    
    def step(self):
        return self.optimizer.step()