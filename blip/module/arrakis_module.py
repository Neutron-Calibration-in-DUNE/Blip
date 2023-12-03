
"""
Arrakis module code.
"""
from tqdm import tqdm

from blip.module import GenericModule
from blip.dataset.arrakis import Arrakis

arrakis_config = {
    "process_simulation": True,
    "simulation_type":    'LArSoft',
    "simulation_folder":  "/local_data/data/",
    "process_type":       [],
    "simulation_files":   [],
}


class ArrakisModule(GenericModule):
    """
    The module class helps to organize meta data and objects related to different tasks
    and execute those tasks based on a configuration file.  The spirit of the 'Module' class
    is to mimic some of the functionality of LArSoft, e.g. where you can specify a chain
    of tasks to be completed, the ability to have nested config files where default parameters
    can be overwritten.

    The Arrakis specific module runs in several different modes,
    """
    def __init__(
        self,
        name:   str,
        config: dict = {},
        mode:   str = '',
        meta:   dict = {}
    ):
        self.name = name
        super(ArrakisModule, self).__init__(
            self.name, config, mode, meta
        )
        self.consumes = [None]
        self.produces = [None]

    def parse_config(self):
        """
        """
        self.check_config()
    
    def check_config(self):
        if "arrakis" not in self.config.keys():
            self.logger.error('"arrakis" section not specified in config!')
        if "simulation_folder" not in self.config["arrakis"].keys():
            self.logger.warn('"arrakis:simulation_folder" not specified in config! setting to "/local_data/"')
            self.config["arrakis"]["simulation_folder"] = "/local_data/"
        if "process_type" not in self.config["arrakis"].keys():
            self.logger.warn('"arrakis:process_type" not specified in config! setting to "[all]"')
            self.config["arrakis"]["process_type"] = ["all"]
        if "simulation_files" not in self.config["arrakis"].keys():
            self.logger.warn('"arrakis:simulation_files" not specfied in config! setting to "[]"')
            self.config["arrakis"]["simulation_files"] = []

    def run_module(self):
        self.arrakis = Arrakis(
            self.name,
            self.config['arrakis'],
            self.meta
        )
        if self.mode == 'larsoft':
            self.run_larsoft()
        elif self.mode == 'ndlar_flow':
            self.run_ndlar_flow()
        elif self.mode == 'larsoft_singles':
            self.run_larsoft_singles()
        elif self.mode == 'ndlar_flow_singles':
            self.run_ndlar_flow_singles()
        else:
            self.logger.warning(f"current mode {self.mode} not an available type!")

    def run_larsoft(self):
        self.logger.info('running larsoft arrakis')
        simulation_file_loop = tqdm(
            enumerate(self.arrakis.simulation_files, 0),
            total=len(self.arrakis.simulation_files),
            leave=False,
            colour='white'
        )
        for ii, simulation_file in simulation_file_loop:
            self.arrakis.load_root_arrays(simulation_file)
            self.arrakis.generate_wire_training_data(simulation_file)
            simulation_file_loop.set_description(f"Running LArSoft Arrakis: File [{ii+1}/{len(self.arrakis.simulation_files)}]")
        self.logger.info('larsoft arrakis finished')

    def run_ndlar_flow(self):
        self.logger.info('running ndlar_flow arrakis')
        simulation_file_loop = tqdm(
            enumerate(self.arrakis.simulation_files, 0),
            total=len(self.arrakis.simulation_files),
            leave=False,
            colour='white'
        )
        for ii, simulation_file in simulation_file_loop:
            self.arrakis.load_flow_arrays(simulation_file)
            self.arrakis.generate_larpix_training_data(simulation_file)
            simulation_file_loop.set_description(f"Running ndlar-flow Arrakis: File [{ii+1}/{len(self.arrakis.simulation_files)}]")
        self.logger.info('ndlar_flow arrakis finished')
    
    def run_larsoft_singles(self):
        self.logger.info('running larsoft singles arrakis')
        simulation_file_loop = tqdm(
            enumerate(self.arrakis.simulation_files, 0),
            total=len(self.arrakis.simulation_files),
            leave=False,
            colour='white'
        )
        for ii, simulation_file in simulation_file_loop:
            self.arrakis.load_root_arrays(simulation_file)
            self.arrakis.generate_wire_training_data(simulation_file)
            simulation_file_loop.set_description(f"Running LArSoft Arrakis: File [{ii+1}/{len(self.arrakis.simulation_files)}]")
        self.logger.info('larsoft arrakis sinles finished')
