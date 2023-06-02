from ctypes import sizeof
import uproot
import os
import numpy as np
import socket
import matplotlib.pyplot as plt
from scipy import stats as st
from datetime import datetime

from blip.utils.logger import Logger
from blip.dataset.common import *

class WirePlanePointCloud:
    def __init__(self,
        name:       str="wire_plane",
        input_file: str=""
    ):
        self.name = name
        self.logger = Logger(self.name, output='both', file_mode='w')
        self.logger.info(f"constructing wire_plane dataset.")
        self.input_file = input_file
        self.uproot_file = uproot.open(self.input_file)
        self.point_cloud_data = self.uproot_file['ana/wire_plane_point_cloud'].arrays(library="np")

        if not os.path.isdir(f"data/{self.name}"):
            os.makedirs(f"data/{self.name}")
        
        self.protodune_tpc_channels = {
            "tpc0": [[0,799],[800,1599],[1600,2079]],
            "tpc1": [[0,799],[800,1599],[2080,2559]],
            "tpc2": [[2560,3359],[3360,4159],[4160,4639]],
            "tpc3": [[2560,3359],[3360,4159],[4640,5119]],
            "tpc4": [[5120,5919],[5920,6719],[6720,7199]],
            "tpc5": [[5120,5919],[5920,6719],[7200,7679]],
            "tpc6": [[7680,8479],[8480,9279],[9280,9759]],
            "tpc7": [[7680,8479],[8480,9279],[9760,10239]],
            "tpc8": [[10240,11039],[11040,11839],[11840,12319]],
            "tpc9": [[10240,11039],[11040,11839],[12320,12799]],
            "tpc10": [[12800,13599],[13600,14399],[14400,14879]],
            "tpc11": [[12800,13599],[13600,14399],[14880,15359]],
        }

    def generate_training_data(self,
        plot_group_statistics:  bool=True
    ):
        """
        We iterate over each view (wire plane) and collect all
        (tdc,channel,adc) points for each point cloud into a position
        array, together with (group_label,group_name,label,name) as
        the categorical information.
        """
        self.logger.info(f"generating training data from file: {self.input_file}")
        view = self.point_cloud_data['view']
        channel = self.point_cloud_data['channel']
        tdc = self.point_cloud_data['tdc']
        energy = self.point_cloud_data['energy'] * 10e5
        adc = self.point_cloud_data['adc']
        # construct ids and names for source, shape and particle labels
        source_label = self.point_cloud_data['source_label']
        shape_label = self.point_cloud_data['shape_label']
        particle_label = self.point_cloud_data['particle_label']
        unique_shape_label = self.point_cloud_data['unique_shape']
        unique_particle_label = self.point_cloud_data['unique_particle']

        #for v in np.unique(np.concatenate(view)):
        for tpc, tpc_ranges in self.protodune_tpc_channels.items():
            for v, tpc_view in enumerate(tpc_ranges):
                """
                For each point cloud, we want to normalize adc against
                all point clouds in the data set, so that it is independent 
                of the specific detector readout.
                """
                channel_view = []
                tdc_view = []
                adc_view = []
                energy_view = []
                source_label_view = []
                shape_label_view = []
                particle_label_view = []
                unique_shape_label_view = []
                unique_particle_label_view = []

                for event in range(len(channel)):
                    view_mask = (
                        (channel[event] >= tpc_view[0]) & 
                        (channel[event] < tpc_view[1]) & 
                        (source_label[event] >= 0) &        # we don't want 'undefined' points in our dataset.
                        (shape_label[event] >= 0) &         # i.e., things with a label == -1
                        (particle_label[event] >= 0)
                    )
                    if np.sum(view_mask) > 0:
                        channel_view.append(channel[event][view_mask])
                        tdc_view.append(tdc[event][view_mask])
                        adc_view.append(adc[event][view_mask])
                        energy_view.append(energy[event][view_mask])
                        source_label_view.append(source_label[event][view_mask])
                        shape_label_view.append(shape_label[event][view_mask])
                        particle_label_view.append(particle_label[event][view_mask])
                        unique_shape_label_view.append(unique_shape_label[event][view_mask])
                        unique_particle_label_view.append(unique_particle_label[event][view_mask])

                channel_view = np.array(channel_view, dtype=object)
                tdc_view = np.array(tdc_view, dtype=object)
                adc_view = np.array(adc_view, dtype=object)
                energy_view = np.array(energy_view, dtype=object)
                source_label_view = np.array(source_label_view, dtype=object)
                shape_label_view = np.array(shape_label_view, dtype=object)
                particle_label_view = np.array(particle_label_view, dtype=object)
                unique_shape_label_view = np.array(unique_shape_label_view, dtype=object)
                unique_particle_label_view = np.array(unique_particle_label_view, dtype=object)

                adc_view_sum = np.array([sum(a) for a in adc_view])
                adc_view_normalized = adc_view / adc_view_sum
                
                features = np.array([
                    np.vstack((channel_view[ii], tdc_view[ii], adc_view_normalized[ii])).T
                    for ii in range(len(channel_view))],
                    dtype=object
                )
                classes = np.array([
                    np.vstack((source_label_view[ii], shape_label_view[ii], particle_label_view[ii])).T
                    for ii in range(len(channel_view))],
                    dtype=object
                )          
                clusters = np.array([
                    np.vstack((unique_shape_label_view[ii], unique_particle_label_view[ii])).T
                    for ii in range(len(channel_view))],
                    dtype=object
                )

                meta = {
                    "who_created":      "me",
                    "when_created":     datetime.now().strftime("%m-%d-%Y-%H:%M:%S"),
                    "where_created":    socket.gethostname(),
                    "num_events":       len(features),
                    "view":             v,
                    "features": {
                        "channel": 0, "tdc": 1, "adc": 2
                    },
                    "classes": {
                        "source": 0, "shape": 1, "particle": 2
                    },
                    "source_labels": {
                        key: value
                        for key, value in classification_labels["source"].items()
                    },
                    "source_points": {
                        key: np.count_nonzero(np.concatenate(source_label_view) == key)
                        for key, value in classification_labels["source"].items()
                    },
                    "shape_labels": {
                        key: value
                        for key, value in classification_labels["shape"].items()
                    },
                    "shape_points": {
                        key: np.count_nonzero(np.concatenate(shape_label_view) == key)
                        for key, value in classification_labels["shape"].items()
                    },
                    "particle_labels": {
                        key: value
                        for key, value in classification_labels["particle"].items()
                    },      
                    "particle_points": {
                        key: np.count_nonzero(np.concatenate(particle_label_view) == key)
                        for key, value in classification_labels["particle"].items()
                    },
                    "total_points":     len(np.concatenate(features)),
                    "adc_view_sum":     adc_view_sum,    
                }
                    
                np.savez(
                    f"data/{self.name}/view{v}_{tpc}.npz",
                    features=features,
                    classes=classes,
                    clusters=clusters,
                    meta=meta
                )

            # source_label_hist, _ = np.histogram(np.concatenate(labels), bins=len(unique_source_labels))
            # source_label_hist = np.divide(source_label_hist, np.sum(source_label_hist, dtype=float), dtype=float)

            # if plot_group_statistics:
            #     adc_view = np.concatenate(adc_view)
            #     source_label_view = np.concatenate(source_label_view)
            #     fig, axs = plt.subplots(figsize=(10,6))
            #     for label in unique_source_labels:
            #         source_adc = adc_view[(source_label_view == label)]
            #         axs.hist(
            #             source_adc, 
            #             bins=100, 
            #             range=[np.min(source_adc),np.max(source_adc)], 
            #             histtype='step',
            #             stacked=True,
            #             density=True,
            #             label=f"{source_label_map[label]}",
            #             log=True
            #         )
            #     axs.set_xlabel("Summed ADC [counts]")
            #     axs.set_ylabel("Point Clouds")
            #     axs.set_title(f"Summed ADC Distribution for View {v}")
            #     axs.ticklabel_format(axis='x', style='sci')
            #     plt.legend()
            #     plt.tight_layout()
            #     plt.savefig(self.wire_plane_dir + f"source_adc_view{v}.png")