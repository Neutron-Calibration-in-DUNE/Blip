"""
Generic clustering algorithm for blip.
"""


class GenericClusteringAlgorithm:
    """
    """
    def __init__(
        self,
        name:   str = 'generic',
        alpha:  float = 0.0,
        device: str = 'cpu'
    ):
        self.name = name
        self.alpha = alpha
        self.classes = []
        self.device = device

    def set_device(
        self,
        device
    ):
        self.device = device

    def cluster(
        self,
        parameters,
        data,
    ):
        pass
