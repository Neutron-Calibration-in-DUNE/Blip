"""
Container for models
"""
from blip.utils.logger import Logger
from blip.models import GenericModel, PointNetPlusPlus
from blip.utils.utils import get_method_arguments

class ModelHandler:
    """
    """
    def __init__(self,
        name:   str,
        cfg:    dict={},
        models:  list=[],
        use_sample_weights: bool=False,
    ):
        self.name = name
        self.use_sample_weights = use_sample_weights
        self.logger = Logger(self.name, file_mode="w")
        if bool(cfg) and len(models) != 0:
            self.logger.error(f"handler received both a config and a list of models! The user should only provide one or the other!")
        else:
            if bool(cfg):
                self.cfg = cfg
                self.process_config()
            else:
                self.models = {model.name: model for model in models}

        # set to whatever the last call of set_device was.
        self.device = 'None'
    
    def process_config(self):
        # list of available models
        # TODO: Make this automatic
        self.available_models = {
            'PointNet++': PointNetPlusPlus,
        }
        # check config
        if self.cfg["model_type"] not in self.available_models.keys():
            self.logger.error(
                f"specified callback '{self.cfg['model_type']}'" +
                f"is not an available type! Available types:\n{self.available_models}"
            )
        self.model = self.available_models[self.cfg['model_type']](
            "blip_model", self.cfg
        )
        print(self.model)

    def set_device(self,
        device
    ):  
        for name, model in self.models.items():
            model.set_device(device)
            model.reset_batch()
        self.device = device