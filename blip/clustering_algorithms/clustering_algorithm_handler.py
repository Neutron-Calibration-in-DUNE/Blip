"""
Container for clustering algorithms
"""
import os
import importlib.util
import sys
import inspect
import torch

from blip.utils.logger import Logger
from blip.clustering_algorithms import GenericClusteringAlgorithm
from blip.utils.utils import get_method_arguments


class ClusteringAlgorithmHandler:
    """
    """
    def __init__(
        self,
        name:    str,
        config:  dict = {},
        clustering_algorithms:  list = [],
        use_sample_weights: bool = False,
        meta:   dict = {}
    ):
        self.name = name + '_clustering_algorithm_handler'
        self.use_sample_weights = use_sample_weights
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(name, output="both", file_mode="w")
        else:
            self.logger = Logger(name, file_mode="w")
        self.clustering_algorithms = {}

        if bool(config) and len(clustering_algorithms) != 0:
            self.logger.error(
                "handler received both a config and a list of clustering_algorithms! " +
                "The user should only provide one or the other!"
            )
        elif bool(config):
            self.set_config(config)
        else:
            if len(clustering_algorithms) == 0:
                self.logger.error("handler received neither a config or clustering_algorithms!")
            self.clustering_algorithms = {
                clustering_algorithm.name: clustering_algorithm
                for clustering_algorithm in clustering_algorithms
            }

    def set_config(self, config):
        self.config = config
        self.process_config()

    def collect_clustering_algorithm_functions(self):
        self.available_criterions = {}
        self.criterion_files = [
            os.path.dirname(__file__) + '/' + file
            for file in os.listdir(path=os.path.dirname(__file__))
        ]
        for criterion_file in self.criterion_files:
            if criterion_file in ["__init__.py", "__pycache__.py", "generic_clustering_algorithm.py"]:
                continue
            try:
                self.load_clustering_algorithm_function(criterion_file)
            except:
                pass

    def load_clustering_algorithm_function(
        self,
        criterion_file: str
    ):
        spec = importlib.util.spec_from_file_location(
            f'{criterion_file.removesuffix(".py")}.name',
            criterion_file
        )
        custom_clustering_algorithm_file = importlib.util.module_from_spec(spec)
        sys.modules[f'{criterion_file.removesuffix(".py")}.name'] = custom_clustering_algorithm_file
        spec.loader.exec_module(custom_clustering_algorithm_file)
        for name, obj in inspect.getmembers(sys.modules[f'{criterion_file.removesuffix(".py")}.name']):
            if inspect.isclass(obj):
                custom_class = getattr(custom_clustering_algorithm_file, name)
                if issubclass(custom_class, GenericClusteringAlgorithm):
                    self.available_criterions[name] = custom_class

    def process_config(self):
        # list of available criterions
        self.collect_clustering_algorithm_functions()
        # check config
        if "custom_clustering_algorithm_file" in self.config.keys():
            if os.path.isfile(self.config["custom_clustering_algorithm_file"]):
                try:
                    self.load_clustering_algorithm_function(self.config["custom_clustering_algorithm_file"])
                    self.logger.info(
                        'added custom clustering_algorithm function from file ' +
                        f'{self.config["custom_clustering_algorithm_file"]}.'
                    )
                except:
                    self.logger.error(
                        f'loading classes from file {self.config["custom_clustering_algorithm_file"]} failed!'
                    )
            else:
                self.logger.error(
                    f'custom_clustering_algorithm_file {self.config["custom_clustering_algorithm_file"]} not found!'
                )
        # process clustering_algorithm functions
        for item in self.config.keys():
            if item == "custom_clustering_algorithm_file":
                continue
            # check that clustering_algorithm function exists
            if item not in self.available_criterions.keys():
                self.logger.error(
                    f"specified clustering_algorithm function '{item}' is not an available type! " +
                    f"Available types:\n{self.available_criterions.keys()}"
                )
            # check that function arguments are provided
            argdict = get_method_arguments(self.available_criterions[item])
            for value in self.config[item].keys():
                if value not in argdict.keys():
                    self.logger.error(
                        f"specified clustering_algorithm function value '{item}:{value}' " +
                        f"not a constructor parameter for '{item}'! " +
                        f"Constructor parameters:\n{argdict}"
                    )
            for value in argdict.keys():
                if argdict[value] is None:
                    if value not in self.config[item].keys():
                        self.logger.error(
                            f"required input parameters '{item}:{value}' " +
                            f"not specified! Constructor parameters:\n{argdict}"
                        )
            self.config[item]["device"] = self.device
        self.clustering_algorithms = {}
        self.batch_clustering_algorithm = {}
        for item in self.config.keys():
            if item == "custom_clustering_algorithm_file":
                continue
            self.clustering_algorithms[item] = self.available_criterions[item](**self.config[item])
            self.batch_clustering_algorithm[item] = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)
            self.logger.info(f'added clustering_algorithm function "{item}" to Clusteringclustering_AlgorithmHandler.')

    def set_device(
        self,
        device
    ):
        self.logger.info(f'setting device to "{device}".')
        for name, clustering_algorithm in self.clustering_algorithms.items():
            clustering_algorithm.set_device(device)
            self.batch_clustering_algorithm[name] = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)
        self.device = device

    def add_clustering_algorithm(
        self,
        clustering_algorithm:   GenericClusteringAlgorithm
    ):
        if issubclass(clustering_algorithm, GenericClusteringAlgorithm):
            self.logger.info(
                f'added clustering_algorithm function "{clustering_algorithm}" to Clusteringclustering_AlgorithmHandler.'
            )
            self.clustering_algorithms[clustering_algorithm.name] = clustering_algorithm
        else:
            self.logger.error(
                f'specified clustering_algorithm {clustering_algorithm} is not a child of "GenericClusteringAlgorithm"!' +
                ' Only clustering_algorithm functions which inherit from GenericClusteringAlgorithm can' +
                ' be used by the ClusteringAlgorithmHandler in BLIP.'
            )

    def cluster(
        self,
        parameters,
        data,
    ):
        clustering = {}
        for name, clustering_algorithm in self.clustering_algorithms.items():
            clustering[name] = clustering_algorithm.cluster(parameters, data)
        return clustering
