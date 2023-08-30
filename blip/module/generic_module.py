
"""
Generic module code.
"""
from blip.utils.logger import Logger

generic_config = {
    "no_params":    "no_values"
}

class GenericModule:
    """
    """
    def __init__(self,
        name:   str,
        config: dict=generic_config,
        mode:   str='',
        meta:   dict={}
    ):
        self.name = name
        self.config = config
        self.mode = mode
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(name, output="both", file_mode="w")
        else:
            self.logger = Logger(name, file_mode="w")
        
        self.module_data_product = {}
        
    def set_device(self,
        device
    ):
        self.device = device
    
    def set_config(self,
        config_file:    str
    ):
        self.config_file = config_file
        self.parse_config()
    
    def parse_config(self):
        self.logger.error(f'"parse_config" not implemented in Module!')
    
    def run_module(self):
        self.logger.error(f'"run_module" not implemented in Module!')
        