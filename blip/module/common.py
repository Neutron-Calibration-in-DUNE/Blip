"""
Module data
"""
import numpy as np

module_types = {
    "clustering": ["parameter_scan"],
    "data_prep":  [],
    "ml":   ["training", "inference"],
    "tda":  []
}

module_aliases = {
    "ml":   "MachineLearningModule",
    "machine_learning": "MachineLearningModule",
    "machinelearning":  "MachineLearningModule",
    "MachineLearning":  "MachineLearningModule",
    "clustering":   "ClusteringModule"
}