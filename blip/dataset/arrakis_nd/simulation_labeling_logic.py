"""
"""
from blip.dataset.arrakis_nd.simulation_wrangler import SimulationWrangler

class SimulationLabelingLogic:
    """
    """
    def __init__(self,
        simulation_wrangler
    ):
        self.simulation_wrangler = simulation_wrangler

    def process_event(self):

        self.process_muons()

    def process_muons(self):
        pass