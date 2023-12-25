
"""
contrastive learning model analyzer code.
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn import cluster
import kmapper as km
import networkx as nx

from blip.analysis.generic_model_analyzer import GenericModelAnalyzer


class ContrastiveModelAnalyzer(GenericModelAnalyzer):
    """
    """
    def __init__(
        self,
        name:           str = 'contrastive_model_analyzer',
        dataset_type:   str = 'inference',
        layers:         list = [],
        outputs:        list = [],
        perplexities:   list = [5, 10, 25, 50],
        eps_values:     list = [1.0, 10.0, 25.0],
        min_samples:    int = 5,
        num_cubes:      int = 15,
        perc_overlap:   float = 0.25,
        meta:           dict = {}
    ):
        super(ContrastiveModelAnalyzer, self).__init__(
            name, dataset_type, layers, outputs, meta
        )
        self.perplexities = perplexities
        self.eps_values = eps_values
        self.min_samples = min_samples
        self.num_cubes = num_cubes
        self.perc_overlap = perc_overlap

        if not os.path.isdir(f"{self.meta['local_scratch']}/.tmp/manifold"):
            os.makedirs(f"{self.meta['local_scratch']}/.tmp/manifold")

    def analyze(
        self,
        input,
        predictions
    ):
        for jj, output in enumerate(self.outputs):
            for kk, perplexity in enumerate(self.perplexities):
                # create TSNE embedding
                embedding = manifold.TSNE(
                    n_components=2,
                    learning_rate='auto',
                    init='random',
                    perplexity=perplexity,
                ).fit_transform(predictions[output])

                target_names = self.meta['dataset'].meta['classes_labels_names']
                for ii, class_target in enumerate(self.meta['dataset'].meta['blip_classes']):
                    temp_targets = input['category'][:, ii]
                    unique_targets = np.unique(temp_targets)

                    fig, axs = plt.subplots(figsize=(10, 6))
                    for target in unique_targets:
                        axs.scatter(
                            embedding[:, 0][(temp_targets == target)],
                            embedding[:, 1][(temp_targets == target)],
                            label=f"{target_names[class_target][target]}"
                        )

                    axs.set_xlabel("tSNE embedding 'x'")
                    axs.set_ylabel("tSNE embedding 'y'")
                    axs.set_title(f"tSNE embedding for category '{class_target}'")
                    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
                    plt.tight_layout()
                    plt.savefig(
                        f"{self.meta['local_scratch']}/.tmp/manifold/{output}_{class_target}_" +
                        f"p={perplexity}_tsne_projection.png"
                    )
                    plt.close()

                    for mm, eps in enumerate(self.eps_values):
                        mapper = km.KeplerMapper(verbose=0)
                        graph = mapper.map(
                            embedding,
                            clusterer=cluster.DBSCAN(
                                eps=eps,
                                min_samples=self.min_samples
                            ),
                            cover=km.Cover(
                                self.num_cubes,
                                self.perc_overlap
                            ),
                        )

                        # TODO: KeplerMapper is broken, numpy incompatibility problems!
                        # mapper.visualize(
                        #     graph,
                        #     title=f"{output} {class_target} Mapper",
                        #     path_html=f"{self.meta['local_scratch']}/.tmp/manifold/{output}_{class_target}_p={perplexity}_eps={eps}_mapper.html",
                        #     color_values=temp_targets,
                        #     color_function_name="labels",
                        #     custom_tooltips=temp_targets,
                        # )

                        # # Tooltips with the target y-labels for every cluster member
                        # mapper.visualize(
                        #     graph,
                        #     title=f"{output} {class_target} Mapper",
                        #     path_html=f"{self.meta['local_scratch']}/.tmp/manifold/{output}_{class_target}_p={perplexity}_eps={eps}_tooltips_mapper.html",
                        #     custom_tooltips=temp_targets,
                        # )
                        fig, axs = plt.subplots(figsize=(25, 25))
                        if not isinstance(graph, nx.Graph):
                            graph = km.adapter.to_networkx(graph)
                        positions = nx.spring_layout(graph)

                        nodes = nx.draw_networkx_nodes(graph, node_size=50, pos=positions, ax=axs)
                        edges = nx.draw_networkx_edges(graph, pos=positions, ax=axs)

                        nodes.set_edgecolor("w")
                        nodes.set_linewidth(3)

                        axs.axis("square")
                        axs.axis("off")
                        axs.set_title(f"Mapper graph with DBSCAN({eps},{self.min_samples}) on tSNE embedding")
                        plt.savefig(
                            f"{self.meta['local_scratch']}/.tmp/manifold/{output}_{class_target}_" +
                            f"p={perplexity}_eps={eps}_mapper.png"
                        )
                        plt.close()
