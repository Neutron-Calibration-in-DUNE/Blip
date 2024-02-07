
"""
Generic analyzer code.
"""
from blip.utils.logger import Logger

generic_config = {
    "no_params":    "no_values"
}


class GenericAnalyzer:
    """
    """
    def __init__(
        self,
        name:   str,
        config: dict = {},
        meta:   dict = {},
    ):
        self.name = name
        self.config = config
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(name, output="both", file_mode="w")
        else:
            self.logger = Logger(name, file_mode="w")

    def set_device(
        self,
        device
    ):
        self.device = device

    def analyze_event(
        self,
        event
    ):
        self.logger.error('analyze_event function has not been defined!')

    def analyze_events(
        self
    ):
        self.logger.error('analyze_events function has not been defined!')
