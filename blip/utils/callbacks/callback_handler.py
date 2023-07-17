"""
Container for generic callbacks
"""
import os
import importlib
import sys
import inspect
from blip.utils.logger import Logger
from blip.utils.callbacks import GenericCallback
from blip.utils.utils import get_method_arguments

class CallbackHandler:
    """
    """

    def __init__(self,
        name:       str,
        config:     dict={},
        callbacks:  list=[],
        meta:       dict={}
    ):
        self.name = name + "_callback_handler"
        self.logger = Logger(self.name, output="both", file_mode="w")
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(name, output="both", file_mode="w")
        else:
            self.logger = Logger(name, file_mode="w")

        if bool(config) and len(callbacks) != 0:
            self.logger.error(
                f"handler received both a config and a list of callbacks! " + 
                f"The user should only provide one or the other!")
        elif bool(config):
            self.set_config(config)
        else:
            if len(callbacks) == 0:
                self.logger.warn(f"handler received neither a config or callbacks!")
            self.callbacks = {
                callback.name: callback 
                for callback in callbacks
            }       
        
    def set_config(self, config):
        self.config = config
        self.process_config()
    
    def collect_callbacks(self):
        self.available_callbacks = {}
        self.callback_files = [
            os.path.dirname(__file__) + '/' + file 
            for file in os.listdir(path=os.path.dirname(__file__))
        ]
        for callback_file in self.callback_files:
            if callback_file in ["__init__.py", "__pycache__.py", "generic_callback.py"]:
                continue
            try:
                self.load_callback(callback_file)
            except:
                pass
    
    def load_callback(self,
        callback_file: str
    ):
        spec = importlib.util.spec_from_file_location(
            f'{callback_file.removesuffix(".py")}.name', 
            callback_file
        )
        custom_callback_file = importlib.util.module_from_spec(spec)
        sys.modules[f'{callback_file.removesuffix(".py")}.name'] = custom_callback_file
        spec.loader.exec_module(custom_callback_file)
        for name, obj in inspect.getmembers(sys.modules[f'{callback_file.removesuffix(".py")}.name']):
            if inspect.isclass(obj):
                custom_class = getattr(custom_callback_file, name)
                if issubclass(custom_class, GenericCallback):
                    self.available_callbacks[name] = custom_class

    def process_config(self):
        # list of available criterions
        self.collect_callbacks()
        # check config
        if "custom_callback_file" in self.config.keys():
            if os.path.isfile(self.config["custom_callback_file"]):
                try:
                    self.load_callback(self.config["custom_callback_file"])
                    self.logger.info(f'added custom callback function from file {self.config["custom_callback_file"]}.')
                except:
                    self.logger.error(
                        f'loading classes from file {self.config["custom_callback_file"]} failed!'
                    )
            else:
                self.logger.error(f'custom_callback_file {self.config["custom_callback_file"]} not found!')
        # process callback functions
        for item in self.config.keys():
            if item == "custom_callback_file":
                continue
            # check that callback function exists
            if item not in self.available_callbacks.keys():
                self.logger.error(
                    f"specified callback function '{item}' is not an available type! " + 
                    f"Available types:\n{self.available_callbacks.keys()}"
                )
            # check that function arguments are provided
            argdict = get_method_arguments(self.available_callbacks[item])
            for value in self.config[item].keys():
                if value not in argdict.keys():
                    self.logger.error(
                        f"specified callback value '{item}:{value}' " + 
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
        self.callbacks = {}
        for item in self.config.keys():
            if item == "custom_callback_file":
                continue
            self.callbacks[item] = self.available_callbacks[item](**self.config[item], meta=self.meta)
            self.logger.info(f'added callback function "{item}" to CallbackHandler.')

    def set_device(self,
        device
    ):  
        for name, callback in self.callbacks.items():
            callback.set_device(device)
            callback.reset_batch()
        self.device = device

    def add_callback(self,
        callback:   GenericCallback
    ):
        if issubclass(type(callback), GenericCallback):
            self.logger.info(f'added callback function "{callback}" to CallbackHandler.')
            self.callbacks[callback.name] = callback
        else:
            self.logger.error(
                f'specified callback {callback} is not a child of "GenericCallback"!' + 
                f' Only callback functions which inherit from GenericCallback can' +
                f' be used by the callbackHandler in BLIP.'
            )
    
    def set_training_info(self,
        epochs: int,
        num_training_batches:   int,
        num_validation_batches:  int,
        num_test_batches:   int,
    ):
        for name, callback in self.callbacks.items():
            callback.set_training_info(
                epochs,
                num_training_batches,
                num_validation_batches,
                num_test_batches
            )

    def evaluate_epoch(self,
        train_type='train',
    ):
        if train_type not in ['training', 'validation', 'test', 'cluster']:
            self.logger.error(f"specified train_type: '{train_type}' not allowed!")
        for name, callback in self.callbacks.items():
            callback.evaluate_epoch(train_type)

    def evaluate_training(self):
        for name, callback in self.callbacks.items():
            callback.evaluate_training()

    def evaluate_testing(self):
        for name, callback in self.callbacks.items():
            callback.evaluate_testing()
    
    def evaluate_inference(self):
        for name, callback in self.callbacks.items():
            callback.evaluate_inference()
    
    def evaluate_clustering(self):
        for name, callback in self.callbacks.items():
            callback.evaluate_clustering()