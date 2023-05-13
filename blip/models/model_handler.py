"""
Container for models
"""
from blip.utils.logger import Logger
from blip.models import GenericModel, PointNetPlusPlus
from blip.models import VietorisRipsNet
from blip.models import SparseUNet, SparseUResNet, SparseUResNeXt
from blip.utils.utils import get_method_arguments

class ModelHandler:
    """
    """
    def __init__(self,
        name:   str,
        config:  dict={},
        models:  list=[],
        use_sample_weights: bool=False,
        device: str='cpu'
    ):
        self.name = name
        self.use_sample_weights = use_sample_weights
        self.logger = Logger(self.name, file_mode="w")
        self.device = device

        if bool(config) and len(models) != 0:
            self.logger.error(f"handler received both a config and a list of models! The user should only provide one or the other!")
        else:
            if bool(config):
                self.config = config
                self.process_config()
            else:
                self.models = {model.name: model for model in models}

    def process_config(self):
        # list of available models
        # TODO: Make this automatic
        self.available_models = {
            'PointNet++':       PointNetPlusPlus,
            'VietorisRipsNet':  VietorisRipsNet,
            'SparseUNet':       SparseUNet,
            'SparseUResNet':    SparseUResNet,
            'SparseUResNeXt':   SparseUResNeXt
        }
        # check config
        if self.config["model_type"] not in self.available_models.keys():
            self.logger.error(
                f"specified callback '{self.config['model_type']}'" +
                f"is not an available type! Available types:\n{self.available_models}"
            )
        self.model = self.available_models[self.config['model_type']](
            "blip_model", self.config, device=self.device
        )
        if 'load_model' in self.config.keys():
            self.model.load_model(self.config['load_model'])

    def set_device(self,
        device
    ):  
        for name, model in self.models.items():
            model.set_device(device)
            model.reset_batch()
        self.device = device