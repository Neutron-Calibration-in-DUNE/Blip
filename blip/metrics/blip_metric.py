"""
Container for generic metrics
"""
import os
import importlib.util
import sys
import inspect
import torch

from blip.utils.logger import BlipError
from blip.metrics.generic_metric import GenericMetric
from blip.utils.utils import profiler


class BlipMetric:
    """
    """
    def __init__(
        self,
        config:  dict = {},
        meta:    dict = {}
    ):

        self.config = config
        self.meta = meta

        if "device" in self.meta:
            self.device = self.meta["device"]

        self.parse_config()

    @profiler
    def collect_metrics(self):
        # Ensure only Python files that are not standard special files are considered
        metric_dir = os.path.dirname(__file__)
        self.available_metrics = {}
        for filename in os.listdir(metric_dir):
            if filename.endswith(".py") and filename not in {
                "__init__.py",
                "__pycache__.py",
                "generic_metric.py"
            }:
                metric_path = os.path.join(metric_dir, filename)
                metric_name = filename[:-3]  # Strip .py from filename to get module name
                try:
                    self.load_metric(metric_path, metric_name)
                except Exception as e:
                    raise BlipError(f"Problem loading metric from {metric_path}: {str(e)}")

    @profiler
    def load_metric(self, metric_path: str, metric_name: str):
        full_module_name = f"blip.metrics.{metric_name}"
        if full_module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(full_module_name, metric_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[full_module_name] = module  # Register the loaded module
        else:
            module = sys.modules[full_module_name]

        # Register classes
        self.register_classes(module, full_module_name)

    @profiler
    def register_classes(self, module, full_module_name):
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, GenericMetric) and obj.__module__ == full_module_name:
                self.available_metrics[name] = obj

    @profiler
    def parse_config(self):
        # list of available criterions
        self.collect_metrics()
        # check config
        if "custom_metric_file" in self.config.keys():
            if os.path.isfile(self.config["custom_metric_file"]):
                try:
                    self.load_metric(self.config["custom_metric_file"])
                except Exception:
                    raise BlipError(
                        f'loading classes from file {self.config["custom_metric_file"]} failed!'
                    )
            else:
                raise BlipError(f'custom_metric_file {self.config["custom_metric_file"]} not found!')
        # process metric functions
        for item in self.config.keys():
            if item == "custom_metric_file":
                continue
            # check that metric function exists
            if item not in self.available_metrics.keys():
                raise BlipError(
                    f"specified metric function '{item}' is not an available type! " +
                    f"Available types:\n{self.available_metrics.keys()}"
                )
        self.metrics = {}
        self.batch_metric = {}
        self.batch_iteration = {}
        for item in self.config.keys():
            if item == "custom_metric_file":
                continue
            if self.config[item] is not None:
                self.metrics[item] = self.available_metrics[item](**self.config[item], meta=self.meta)
            else:
                self.metrics[item] = self.available_metrics[item](meta=self.meta)
            self.batch_metric[item] = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)
            self.batch_iteration[item] = []

    def report_tensorboard(
        self,
        iterations,
        train_type
    ):
        for name, metric in self.metrics.items():
            metric.report_tensorboard(iterations=iterations, train_type=train_type)
        self.reset_batch()

    def set_device(
        self,
        device
    ):
        for name, metric in self.metrics.items():
            metric.set_device(device)
            self.batch_metric[name] = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)
            self.batch_iteration[name] = []
        self.device = device

    def reset_batch(self):
        for name, metric in self.metrics.items():
            self.batch_metric[name] = torch.empty(size=(0, 1), dtype=torch.float, device=self.device)
            self.batch_iteration[name] = []
            metric.reset_batch()

    def add_metric(
        self,
        metric:   GenericMetric
    ):
        if issubclass(type(metric), GenericMetric):
            self.metrics[metric.name] = metric
        else:
            raise BlipError(
                f'specified metric {metric} is not a child of "GenericMetric"!' +
                ' Only metric functions which inherit from GenericMetric can' +
                ' be used by the metricHandler in BLIP.'
            )

    def remove_metric(
        self,
        metric:   str
    ):
        if metric in self.metrics.keys():
            self.metrics.pop(metric)
        if metric in self.batch_metric.keys():
            self.batch_metric.pop(metric)
            self.batch_iteration.pop(metric)

    def update(
        self,
        data,
    ):
        for name, metric in self.metrics.items():
            metric.update(data)

    def compute(
        self,
    ):
        metrics = {
            name: metric.compute()
            for name, metric in self.metrics.items()
        }
        return metrics
