"""
Module data
"""

module_types = {
    "clustering": ["parameter_scan"],
    "arrakis":    ["larsoft", "ndlar_flow"],
    "data_prep":  [],
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
    "ml":               "MachineLearningModule",
    "machine_learning": "MachineLearningModule",
    "machinelearning":  "MachineLearningModule",
    "MachineLearning":  "MachineLearningModule",
    "clustering":       "ClusteringModule",
    "blip_net":         "BlipNetModule"
}
