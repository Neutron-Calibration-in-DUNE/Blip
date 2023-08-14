"""
"""
import os 
import importlib.util
import sys
import inspect
from tqdm import tqdm
from blip.utils.logger import Logger
from blip.module import GenericModule
from blip.module.common import *
from blip.utils.utils import get_method_arguments

class ModuleHandler:
    """
    """
    def __init__(self,
        name:   str,
        config: dict={},
        modules:    list=[],
        meta:   dict={}
    ):
        self.name = name + "_module_handler"
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(name, output="both", file_mode="w")
        else:
            self.logger = Logger(name, level='warning', file_mode="w")
        
        self.modules = {}

        if bool(config) and len(modules) != 0:
            self.logger.error(
                f"handler received both a config and a list of modules! " + 
                f"The user should only provide one or the other!")
        elif bool(config):
            self.set_config(config)
        else:
            if len(modules) == 0:
                self.logger.error(f"handler received neither a config or modules!")
            self.modules = {
                module.name: module 
                for module in modules
            }
        
    def set_config(self, config):
        self.config = config
        self.process_config()
    
    def collect_modules(self):
        self.available_modules = {}
        self.module_files = [
            os.path.dirname(__file__) + '/' + file 
            for file in os.listdir(path=os.path.dirname(__file__))
        ]
        for module_file in self.module_files:
            if module_file in ["__init__.py", "__pycache__.py", "generic_module.py"]:
                continue
            try:
                self.load_module(module_file)
            except:
                pass
    
    def load_module(self,
        module_file: str
    ):
        spec = importlib.util.spec_from_file_location(
            f'{module_file.removesuffix(".py")}.name', 
            module_file
        )
        custom_module_file = importlib.util.module_from_spec(spec)
        sys.modules[f'{module_file.removesuffix(".py")}.name'] = custom_module_file
        spec.loader.exec_module(custom_module_file)
        for name, obj in inspect.getmembers(sys.modules[f'{module_file.removesuffix(".py")}.name']):
            if inspect.isclass(obj):
                custom_class = getattr(custom_module_file, name)
                print(custom_class)
                if issubclass(custom_class, GenericModule):
                    self.available_modules[name] = custom_class
    
    def process_config(self):
        # list of available 
        self.collect_modules()
        # check config
        if "custom_module_file" in self.config["module"].keys():
            if os.path.isfile(self.config["module"]["custom_module_file"]):
                try:
                    self.load_module(self.config["module"]["custom_module_file"])
                    self.logger.info(f'added custom module from file {self.config["module"]["custom_module_file"]}.')
                except:
                    self.logger.error(
                        f'loading classes from file {self.config["module"]["custom_module_file"]} failed!'
                    )
            else:
                self.logger.error(f'custom_module_file {self.config["module"]["custom_module_file"]} not found!')
        if "module_type" not in self.config["module"].keys():
            self.logger.error(f'module_type not specified in config!')
        if "module_mode" not in self.config["module"].keys():
            self.logger.error(f'module_mode not specified in config!')
        self.module_type = self.config["module"]["module_type"]
        self.module_mode = self.config["module"]["module_mode"]
        if len(self.module_type) != len(self.module_mode):
            self.logger.error(f'module:module_type and module:module_mode must have the same number of entries!')
        
        # process modules
        for ii, item in enumerate(self.module_type):
            if item in module_aliases.keys():
                self.logger.info(f"converting module alias '{item}' to '{module_aliases[item]}'")
                self.module_type[ii] = module_aliases[item]
            # check that module exists
            if self.module_type[ii] not in self.available_modules.keys():
                self.logger.error(
                    f"specified module '{item}' is not an available type! " + 
                    f"Available types:\n{self.available_modules.keys()}"
                )
        self.modules = {}
        for ii, item in enumerate(self.module_type):
            self.modules[item] = self.available_modules[item](
                item, self.config, self.module_mode[ii], self.meta
            )
            self.modules[item].parse_config()
            self.logger.info(f'added module "{item}" to ModuleHandler.')

    def set_device(self,
        device
    ):  
        self.logger.info(f'setting device to "{device}".')
        for name, module in self.modules.items():
            module.set_device(device)
        self.device = device
    
    def add_module(self,
        module:   GenericModule
    ):
        if issubclass(type(module), GenericModule):
            self.logger.info(f'added module function "{module}" to ModuleHandler.')
            self.modules[module.name] = module
        else:
            self.logger.error(
                f'specified module {module} is not a child of "GenericModule"!' + 
                f' Only modules which inherit from GenericModule can' +
                f' be used by the ModuleHandler in BLIP.'
            )
    
    def run_modules(self):
        """
        Once everything is configured, we run the modules here.
        """
        module_loop = tqdm(
            enumerate(self.modules, 0), 
            total=len(self.modules), 
            leave=False,
            colour='white'
        )
        for ii, module in module_loop:
            module_loop.set_description(f"Running module: {self.modules[module].name} [{ii+1}/{len(self.modules)}]")
            self.modules[module].run_module()