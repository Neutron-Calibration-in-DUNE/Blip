"""
Container for generic callbacks
"""
from blip.utils.logger import Logger
from blip.metrics import GenericMetric
from blip.metrics import AUROCMetric
from blip.metrics import ConfusionMatrixMetric
from blip.metrics import DiceScoreMetric
from blip.metrics import JaccardIndexMetric
from blip.metrics import PrecisionMetric
from blip.metrics import RecallMetric
from blip.utils.utils import get_method_arguments

class MetricHandler:
    """
    """
    def __init__(self,
        name:   str,
        cfg:    dict={},
        metrics:  list=[],
    ):
        self.name = name
        self.logger = Logger(self.name, file_mode="w")
        if bool(cfg) and len(metrics) != 0:
            self.logger.error(f"handler received both a config and a list of metrics! The user should only provide one or the other!")
        else:
            if bool(cfg):
                self.cfg = cfg
                self.process_config()
            else:
                self.metrics = {metric.name: metric for metric in metrics}

        # set to whatever the last call of set_device was.
        self.device = 'None'
    
    def process_config(self):
        # list of available criterions
        # TODO: Make this automatic
        # list of available metrics
        self.available_metrics = {
            'auroc':            AUROCMetric,
            'confusion_matrix': ConfusionMatrixMetric,
            'dice_score':       DiceScoreMetric,
            'jaccard_index':    JaccardIndexMetric,
            'precision':        PrecisionMetric,
            'recall':           RecallMetric,
        }

        # check config
        for item in self.cfg.keys():
            if item not in self.available_metrics.keys():
                self.logger.error(f"specified metric '{item}' is not an available type! Available types:\n{self.available_metrics}")
            argdict = get_method_arguments(self.available_metrics[item])
            for value in self.cfg[item].keys():
                if value == "metric":
                    continue
                if value not in argdict.keys():
                    self.logger.error(f"specified metric value '{item}:{value}' not a constructor parameter for '{item}'! Constructor parameters:\n{argdict}")
            for value in argdict.keys():
                if argdict[value] == None:
                    if value not in self.cfg[item].keys():
                        self.logger.error(f"required input parameters '{item}:{value}' not specified! Constructor parameters:\n{argdict}")
        
        self.metrics = {}
        for item in self.cfg.keys():
            self.metrics[item] = self.available_metrics[item](**self.cfg[item])

    def set_device(self,
        device
    ):  
        for name, metric in self.metrics.items():
            metric.metric.to(device)
            metric.reset()
        self.device = device
    
    def set_shapes(self,
        input_shapes,
    ):
        pass

    def reset(self):  
        for name, metric in self.metrics.items():
            metric.reset()

    def add_metric(self,
        metric:   GenericMetric
    ):
        self.metrics.append(metric)
    
    def set_training_info(self,
        epochs: int,
        num_training_batches:   int,
        num_validation_batches:  int,
        num_test_batches:   int,
    ):
        for name, metric in self.metrics.items():
            metric.set_training_info(
                epochs,
                num_training_batches,
                num_validation_batches,
                num_test_batches
            )
    
    def update(self,
        outputs,
        data,
        train_type: str='all',
    ):
        for name, metric in self.metrics.items():
            if train_type == metric.when_compute or metric.when_compute == 'all':
                metric.update(outputs, data)
    
    def compute(self,
        outputs,
        data
    ):
        metrics = [metric.compute(outputs, data) for name, metric in self.metrics.items()]
        return 