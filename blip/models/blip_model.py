"""
Container for models
"""
import os
import importlib.util
import sys
import inspect

from blip.utils.utils import profiler
from blip.utils.logger import BlipError
from blip.models.generic_model import GenericModel


class BlipModel:
    """
    """
    def __init__(
        self,
        config:  dict = {},
        meta:   dict = {}
    ):
        self.config = config
        self.meta = meta

        self.parse_config()

    @profiler
    def collect_models(self):
        # Ensure only Python files that are not standard special files are considered
        model_dir = os.path.dirname(__file__)
        self.available_models = {}
        for filename in os.listdir(model_dir):
            if filename.endswith(".py") and filename not in {
                "__init__.py",
                "__pycache__.py",
                "generic_model.py"
            }:
                model_path = os.path.join(model_dir, filename)
                model_name = filename[:-3]  # Strip .py from filename to get module name
                try:
                    self.load_model(model_path, model_name)
                except Exception as e:
                    raise BlipError(f"Problem loading model from {model_path}: {str(e)}")

    @profiler
    def load_model(self, model_path: str, model_name: str):
        full_module_name = f"blip.models.{model_name}"
        if full_module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(full_module_name, model_path)
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
            if issubclass(obj, GenericModel) and obj.__module__ == full_module_name:
                self.available_models[name] = obj

    @profiler
    def parse_config(self):
        # list of available models
        self.collect_models()
        # check config
        if "custom_model_file" in self.config.keys():
            if os.path.isfile(self.config["custom_model_file"]):
                try:
                    self.load_model(self.config["custom_model_file"])
                except Exception:
                    raise BlipError(
                        f'loading classes from file {self.config["custom_model_file"]} failed!'
                    )
            else:
                raise BlipError(f'custom_model_file {self.config["custom_model_file"]} not found!')
        if "model_type" not in self.config.keys():
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
                raise BlipError(
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
        if self.model_type == 'single':
            if len(self.models.keys()) > 1:
                raise BlipError('model_type set to "single", but multiple models have been registered!')
            else:
                self.model = list(self.models.values())[0]
        if 'load_model' in self.config.keys():
            self.model.load_model(self.config['load_model'])

    def set_device(
        self,
        device
    ):
        for name, model in self.models.items():
            model.set_device(device)
        self.device = device

    def add_model(
        self,
        model:   GenericModel
    ):
        if issubclass(type(model), GenericModel):
            self.models[model.name] = model
        else:
            raise BlipError(
                'specified model {model} is not a child of "GenericModel"!' +
                ' only models which inherit from GenericModel can' +
                ' be used by the ModelHandler in BLIP.'
            )

    def train(self):
        if self.model_type == 'single':
            self.model.train()
        else:
            for name, model in self.models.items():
                try:
                    model.train()
                except Exception:
                    self.logger.warn(f'problem with setting train for model {name}')

    def eval(self):
        if self.model_type == 'single':
            self.model.eval()
        else:
            for name, model in self.models.items():
                try:
                    model.eval()
                except Exception:
                    self.logger.warn(f'problem with setting eval for model {name}')

    def contrastive_learning(self):
        if self.model_type == 'single':
            self.model.contrastive_learning()
        else:
            for name, model in self.models.items():
                try:
                    model.contrastive_learning()
                except Exception:
                    self.logger.warn(f'problem with setting contrastive learning for model {name}')

    def linear_evaluation(self):
        if self.model_type == 'single':
            self.model.linear_evaluation()
        else:
            for name, model in self.models.items():
                try:
                    model.linear_evaluation()
                except Exception:
                    self.logger.warn(f'problem with setting linear_evaluation for model {name}')

    def parameters(self):
        if self.model_type == 'single':
            return self.model.parameters()
        else:
            parameters = []
            for name, model in self.models.items():
                parameters += model.parameters()
            return parameters

    def forward_views(self):
        if self.model_type == 'single':
            return self.model.forward_views()
        else:
            forward_views = {}
            for name, model in self.models.items():
                model_views = model.forward_views()
                for view_name, view in model_views.items():
                    forward_views[view_name] = view
            return forward_views
