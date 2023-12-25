"""
"""
import MinkowskiEngine as ME


quantization_modes = {
    "random_subsample":     ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
    "unweighted_average":   ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
    "unweighted_sum":       ME.SparseTensorQuantizationMode.UNWEIGHTED_SUM,
    "no_quantization":      ME.SparseTensorQuantizationMode.NO_QUANTIZATION,
}

minkowski_algorithms = {
    "memory_efficient":     ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
    "speed_optimized":   ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
}
