
"""
Generic model analyzer code.
"""
from blip.utils.logger import Logger

generic_config = {
    "no_params":    "no_values"
}


class GenericModelAnalyzer:
    """
    """
    def __init__(
        self,
        name:   str,
        dataset_type:   str,
        layers:     list = [],
        outputs:    list = [],
        meta:       dict = {}
    ):
        self.name = name
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(name, output="both", file_mode="w")
        else:
            self.logger = Logger(name, file_mode="w")

        if dataset_type is None:
            self.logger.warning('dataset_type not specified! setting to "inference"')
            self.dataset_type = 'inference'
        else:
            self.dataset_type = dataset_type

        if layers is None:
            self.logger.error('"layers" not specified for model analyzer!')
        else:
            self.layers = layers

        if outputs is None:
            self.logger.error('"outputs" not specified for model analyzer!')
        else:
            self.outputs = outputs

    def set_device(
        self,
        device
    ):
        self.device = device

    def analyze(
        self,
        input,
        predictions
    ):
        self.logger.error('analyze function has not been defined!')
