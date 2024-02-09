
"""
nArInelastic analyzer code.
"""
from blip.analysis.generic_analyzer import GenericAnalyzer

generic_config = {
    "no_params":    "no_values"
}


class nArInelasticAnalyzer(GenericAnalyzer):
    """
    This analyzer does several things:
        1. Generate nAr-inelastic plots from larnd-sim.
            a.
            b.
            c.
            d. 
        2. Perform the rest of the analysis chain, after
           segmentation and other reconstruction tasks.
           a.
           b.
           c.
           d.
    """
    def __init__(
        self,
        name:   str,
        config: dict = {},
        meta:   dict = {},
    ):
        super(nArInelasticAnalyzer, self).__init__(
            name, config, meta
        )

        self.mc_data = {
            "lar_fv":   [],
            "p_vtx":    [],
            "nu_vtx":   [],
            "proton_total_energy":  [],
            "proton_vis_energy":    [],
            "proton_length":        [],
            "proton_start_momentum":    [],
            "proton_end_momentum":      [],
            "parent_total_energy":      [],
            "parent_length":    [],
            "parent_start_momentum":    [],
            "parent_end_momentum":      [],
            "nu_proton_dt": [],
            "nu_proton_distance":   [],
            "parent_pdg":   [],
            "grandparent_pdg":  [],
            "primary_pdg":  [],
            "primary_length":   [],
        }

        self.process_config()

    def process_config(self):
        if "analyzer_modes" in self.config:
            self.analyzer_modes = self.config["analyzer_modes"]

    def set_device(
        self,
        device
    ):
        self.device = device

    def analyze_event(
        self,
        event
    ):
        for analyzer_mode in self.analyzer_modes:
            if analyzer_mode == "mc_truth":
                self.analyze_event_mc_truth(event)
            elif analyzer_mode == "analysis_chain":
                self.analyze_event_analysis_chain(event)

    def analyze_event_mc_truth(
        self,
        event
    ):
        self.logger.info("processing mc_truth")

    def analyze_event_analysis_chain(
        self,
        event
    ):
        pass

    def analyze_events(
        self
    ):
        self.logger.info("analyze_events")
