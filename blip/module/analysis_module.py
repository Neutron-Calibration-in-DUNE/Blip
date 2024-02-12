
"""
Analysis module code.
"""
from tqdm import tqdm

from blip.module import GenericModule
from blip.analysis.analysis_handler import AnalysisHandler

generic_config = {
    "no_params":    "no_values"
}


class AnalysisModule(GenericModule):
    """
    """
    def __init__(
        self,
        name:   str,
        config: dict = {},
        mode:   str = '',
        meta:   dict = {}
    ):
        self.name = name + "_ml_module"
        super(AnalysisModule, self).__init__(
            self.name, config, mode, meta
        )

        self.consumes = ['dataset', 'loader']
        self.produces = []

    def parse_config(self):
        self.check_config()

        self.meta['analysis'] = None

        self.parse_analysis()

    def check_config(self):
        if "analysis" not in self.config.keys():
            self.logger.error('"analysis" section not specified in config!')

    def parse_analysis(self):
        self.logger.info("configuring analysis")
        analysis_config = self.config["analysis"]
        self.meta["analysis"] = AnalysisHandler(
            self.name,
            analysis_config,
            meta=self.meta
        )

    def run_module(self):
        # self.meta["dataset"].set_load_type(True)
        inference_loader = self.meta['loader'].mc_truth_loader
        inference_indices = self.meta['loader'].all_indices
        inference_loop = tqdm(
            enumerate(inference_loader, 0),
            total=len(list(inference_indices)),
            leave=True,
            position=0,
            colour='magenta'
        )
        for ii, data in inference_loop:
            self.meta["analysis"].analyze_event(data)
            inference_loop.set_description(f"Analysis: Event [{ii+1}/{len(list(inference_indices))}]")
        self.meta["analysis"].analyze_events()
