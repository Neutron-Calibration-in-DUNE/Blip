"""
Module data
"""
import numpy as np

module_types = {
    "clustering": ["parameter_scan"],
    "data_prep":  [],
    "ml":   ["training", "inference", "hyper_parameter_scan"],
    "tda":  ["merge_tree"],
    "mcts": ["playout"]
}

module_aliases = {
    "ml":   "MachineLearningModule",
    "machine_learning": "MachineLearningModule",
    "machinelearning":  "MachineLearningModule",
    "MachineLearning":  "MachineLearningModule",
    "clustering":   "ClusteringModule",
    "blip_net":     "BlipNetModule"
}