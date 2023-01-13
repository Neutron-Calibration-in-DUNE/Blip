"""
Confusion matrix metric.
"""
import torch
import torch.nn as nn
from tsne_torch import TorchTSNE as TSNE

from blip.metrics import GenericMetric

class EmbeddingMetric(GenericMetric):
    
    def __init__(self,
        name:       str='confusion_matrix',
        shape:      tuple=(),
        input:      str='reductions',
        num_classes:    int=2,
    ):
        """
        """
        super(EmbeddingMetric, self).__init__(
            name,
            (0,2),
            input
        )
        self.num_classes = num_classes
        self.embedding = TSNE(
            n_components=2, 
            n_iter=1000, 
            perplexity=15,
        )

    def update(self,
        outputs,
        data,
    ):
        # set predictions using cutoff
        embedding = self.embedding.fit_transform(
            outputs[self.input]
        )
        self.batch_metric = torch.cat(
            (self.batch_metric, torch.tensor(embedding, device=self.device)),
            dim=0
        )

    def compute(self):
        pass