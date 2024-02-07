
"""
Purity analyzer code.
"""
from blip.analysis.generic_analyzer import GenericAnalyzer

generic_config = {
    "no_params":    "no_values"
}


class PurityAnalyzer(GenericAnalyzer):
    """
    """
    def __init__(
        self,
        name:   str,
        config: dict = {},
        meta:   dict = {},
    ):
        super(PurityAnalyzer, self).__init__(
            name, config, meta
        )

    def set_device(
        self,
        device
    ):
        self.device = device

    def analyze_event(
        self,
        event
    ):
        event, charge = event
        print(charge.particles)
        self.logger.info("analyze_event")

    def analyze_events(
        self
    ):
        self.logger.info("analyze_events")
