"""
Module data
"""

module_types = {
    "clustering": ["parameter_scan"],
    "arrakis":    ["larsoft", "ndlar_flow"],
    "dataset":    ["dataset_prep", "dataset_load"],
    "ml":   [
        "training", "contrastive_training",
        "inference",
        "hyper_parameter_scan", "contrastive_hyper_parameter_scan",
        "linear_evaluation",
        "model_analyzer"
    ],
    "tda":  ["merge_tree"],
    "mcts": ["playout"]
}

module_aliases = {
    "arrakis":          "ArrakisModule",
    "Arrakis":          "ArrakisModule",
    "arrakis_module":   "ArrakisModule",
    "dataset":          "DatasetModule",
    "Dataset":          "DatasetModule",
    "data_prep":        "DatasetModule",
    "ml":               "MachineLearningModule",
    "machine_learning": "MachineLearningModule",
    "machinelearning":  "MachineLearningModule",
    "MachineLearning":  "MachineLearningModule",
    "clustering":       "ClusteringModule",
    "blip_net":         "BlipNetModule"
}
