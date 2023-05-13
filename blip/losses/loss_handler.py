"""
Container for generic losses
"""
from blip.utils.logger import Logger
from blip.losses import GenericLoss
from blip.losses import L1Loss, L2Loss, NTXEntropyLoss
from blip.losses import CrossEntropyLoss
from blip.losses import NegativeLogLikelihoodLoss
from blip.losses import MultiClassNegativeLogLikelihoodLoss
from blip.losses import MultiClassCrossEntropyLoss
from blip.utils.utils import get_method_arguments

class LossHandler:
    """
    """
    def __init__(self,
        name:   str,
        config:    dict={},
        losses:  list=[],
        use_sample_weights: bool=False,
        device: str='cpu'
    ):
        self.name = name
        self.use_sample_weights = use_sample_weights
        self.logger = Logger(self.name, output="both", file_mode="w")
        self.device = device

        if bool(config) and len(losses) != 0:
            self.logger.error(f"handler received both a config and a list of losses! The user should only provide one or the other!")
        else:
            if bool(config):
                self.config = config
                self.process_config()
            else:
                self.losses = {loss.name: loss for loss in losses}        
    
    def process_config(self):
        # list of available criterions
        # TODO: Make this automatic
        self.available_criterions = {
            'l1_loss':                  L1Loss,
            'l2_loss':                  L2Loss,
            'ntx_entropy_loss':         NTXEntropyLoss,
            'ce_loss':                  CrossEntropyLoss,
            'nll_loss':                 NegativeLogLikelihoodLoss,
            'multiclass_nll_loss':      MultiClassNegativeLogLikelihoodLoss,
            'multiclass_ce_loss':       MultiClassCrossEntropyLoss
        }
        # check config
        for item in self.config.keys():
            if item == 'classes' or item == 'class_weights':
                continue
            elif item not in self.available_criterions.keys():
                self.logger.error(f"specified loss function '{item}' is not an available type! Available types:\n{self.available_criterions}")
            argdict = get_method_arguments(self.available_criterions[item])
            for value in self.config[item].keys():
                if value not in argdict.keys():
                    self.logger.error(f"specified loss function value '{item}:{value}' not a constructor parameter for '{item}'! Constructor parameters:\n{argdict}")
            for value in argdict.keys():
                if argdict[value] == None:
                    if value not in self.config[item].keys():
                        self.logger.error(f"required input parameters '{item}:{value}' not specified! Constructor parameters:\n{argdict}")
            if "classes" in self.config.keys():
                self.config[item]["classes"] = self.config["classes"]
            if "class_weights" in self.config.keys():
                self.config[item]["class_weights"] = self.config["class_weights"]
            self.config[item]["device"] = self.device
        self.losses = {}
        for item in self.config.keys():
            if item == 'classes' or item == 'class_weights':
                continue
            else:
                self.losses[item] = self.available_criterions[item](**self.config[item])

    def set_device(self,
        device
    ):  
        for name, loss in self.losses.items():
            loss.set_device(device)
            loss.reset_batch()
        self.device = device

    def reset_batch(self):  
        for name, loss in self.losses.items():
            loss.reset_batch()

    def add_loss(self,
        loss:   GenericLoss
    ):
        self.losses[loss.name] = loss
    
    def set_training_info(self,
        epochs: int,
        num_training_batches:   int,
        num_validation_batches:  int,
        num_test_batches:   int,
    ):
        for name, loss in self.losses.items():
            loss.set_training_info(
                epochs,
                num_training_batches,
                num_validation_batches,
                num_test_batches
            )
            loss.reset_batch()

    def loss(self,
        outputs,
        data,
    ):
        if self.use_sample_weights:
            weights = data[2].to(self.device)
            losses = [(loss.loss(outputs, data) * weights / weights.sum()).sum() for name, loss in self.losses.items()]
        else:
            losses = [loss.loss(outputs, data) for name, loss in self.losses.items()]
        return sum(losses)