"""
Container for models
"""
import os 
import importlib.util
import sys
import inspect
from blip.utils.logger import Logger
from blip.models import GenericModel, PointNetPlusPlus
from blip.models import PointNet, VietorisRipsNet
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
        meta:   dict={}
    ):
        self.name = name + "_model_handler"
        self.use_sample_weights = use_sample_weights
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(name, output="both", file_mode="w")
        else:
            self.logger = Logger(name, file_mode="w")

        self.model_type = None
        self.single_model_name = ''
        self.models = {}

        if bool(config) and len(models) != 0:
            self.logger.error(
                f"handler received both a config and a list of models! " + 
                f"The user should only provide one or the other!")
        elif bool(config):
            self.set_config(config)
        else:
            if len(models) == 0:
                self.logger.error(f"handler received neither a config or models!")
            self.models = {
                model.name: model 
                for model in models
            }
        
    def set_config(self, config):
        self.config = config
        self.process_config()

    def collect_models(self):
        self.available_models = {}
        self.model_files = [
            os.path.dirname(__file__) + '/' + file 
            for file in os.listdir(path=os.path.dirname(__file__))
        ]
        for model_file in self.model_files:
            if model_file in ["__init__.py", "__pycache__.py", "generic_loss.py"]:
                continue
            try:
                self.load_model(model_file)
            except:
                pass
    
    def load_model(self,
        model_file: str
    ):
        spec = importlib.util.spec_from_file_location(
            f'{model_file.removesuffix(".py")}.name', 
            model_file
        )
        custom_model_file = importlib.util.module_from_spec(spec)
        sys.modules[f'{model_file.removesuffix(".py")}.name'] = custom_model_file
        spec.loader.exec_module(custom_model_file)
        for name, obj in inspect.getmembers(sys.modules[f'{model_file.removesuffix(".py")}.name']):
            if inspect.isclass(obj):
                custom_class = getattr(custom_model_file, name)
                if issubclass(custom_class, GenericModel):
                    self.available_models[name] = custom_class

    def process_config(self):
        # list of available criterions
        self.collect_models()
        # check config
        if "custom_model_file" in self.config.keys():
            if os.path.isfile(self.config["custom_model_file"]):
                try:
                    self.load_model(self.config["custom_model_file"])
                    self.logger.info(f'added custom model from file {self.config["custom_model_file"]}.')
                except:
                    self.logger.error(
                        f'loading classes from file {self.config["custom_model_file"]} failed!'
                    )
            else:
                self.logger.error(f'custom_model_file {self.config["custom_model_file"]} not found!')
        if "model_type" not in self.config.keys():
            self.logger.warn(f'model_type not specified in config! Setting to "single"!')
            self.model_type = 'single'
        # process models
        for item in self.config.keys():
            if item == "custom_model_file" or item == "load_model":
                continue
            if item == "model_type":
                self.model_type = self.config[item]
                continue
            # check that model exists
            if item not in self.available_models.keys():
                self.logger.error(
                    f"specified model '{item}' is not an available type! " + 
                    f"Available types:\n{self.available_models.keys()}"
                )
        self.models = {}
        self.batch_model = {}
        for item in self.config.keys():
            if item == "custom_model_file" or item == "load_model" or item == "model_type":
                continue
            self.models[item] = self.available_models[item](
                item, self.config[item], self.meta
            )
            self.logger.info(f'added model "{item}" to ModelHandler.')
        if self.model_type == 'single':
            if len(self.models.keys()) > 1:
                self.logger.error(f'model_type set to "single", but multiple models have been registered!')
            else:
                self.model = list(self.models.values())[0]

    def set_device(self,
        device
    ):  
        self.logger.info(f'setting device to "{device}".')
        for name, model in self.models.items():
            model.set_device(device)
        self.device = device

    def add_model(self,
        model:   GenericModel
    ):
        if issubclass(type(model), GenericModel):
            self.logger.info(f'added model function "{model}" to ModelHandler.')
            self.models[model.name] = model
        else:
            self.logger.error(
                f'specified model {model} is not a child of "GenericModel"!' + 
                f' Only models which inherit from GenericModel can' +
                f' be used by the ModelHandler in BLIP.'
            )
    
    def __call__(self, inputs):
        if self.model_type == "single":
            return self.models[self.single_model_name](inputs)