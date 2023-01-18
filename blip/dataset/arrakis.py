from ctypes import sizeof
import uproot
import os
import numpy as np
import socket
import matplotlib.pyplot as plt
from scipy import stats as st
from datetime import datetime

from blip.utils.logger import Logger

class Arrakis:
    def __init__(self,
        input_file
    ):
        self.name = "arrakis"
        self.logger = Logger(self.name, output='both', file_mode='w')
        self.logger.info(f"constructing arrakis dataset.")
        self.input_file = input_file
        self.uproot_file = uproot.open(self.input_file)
        self.point_cloud_data = self.uproot_file['ana/solo_point_cloud'].arrays(library="np")
        self.arrakis_dir = "plots/arrakis/"

        # create directories
        if not os.path.isdir(self.arrakis_dir):
            self.logger.info(f"creating arrakis plots directory '{self.arrakis_dir}'")
            os.makedirs(self.arrakis_dir)
        if not os.path.isdir("data/"):
            os.makedirs("data/")

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
        views = np.concatenate(view)


        for v in np.unique(np.concatenate(view)):
            """
            For each point cloud, we want to normalize (tdc,channel,adc) against
            all point clouds in the data set, so that it is independent of the specific 
            geometry and detector readout.
            """
            channel = self.point_cloud_data['channel']
            channel_mean = np.mean(np.concatenate(channel)[(views == v)])
            channel_std = np.std(np.concatenate(channel)[(views == v)])

            tdc = self.point_cloud_data['tdc']
            tdc_mean = np.mean(np.concatenate(tdc)[(views == v)])
            tdc_std = np.std(np.concatenate(tdc)[(views == v)])

            adc = self.point_cloud_data['adc']
            adc_mean = np.mean(np.concatenate(adc)[(views == v)])
            adc_std = np.std(np.concatenate(adc)[(views == v)])
            adc_sum = np.array([sum(a) for a in adc])

            # construct ids and names for group labels
            group_label = self.point_cloud_data['group_label']
            unique_group_labels = np.unique(group_label)
            group_label_map = {
                unique_group_labels[ii]: ii 
                for ii in range(len(unique_group_labels))
            }
            group_label_ids = np.array([group_label_map[l] for l in group_label])

            # construct ids and names for individual labels
            label = self.point_cloud_data['label']
            unique_labels = np.unique(label)
            label_map = {unique_labels[ii]: ii for ii in range(len(unique_labels))}
            label_ids = np.array([label_map[l] for l in label])

            energy = self.point_cloud_data['total_energy'] * 10e5

            positions = []
            group_labels = []
            group_names = []
            labels = []
            label_names = []
            total_energy = []
            summed_adc = []

            for ii in range(len(self.point_cloud_data['tdc'])):
                temp_pos = []
                temp_view = self.point_cloud_data['view'][ii]
                temp_tdc = (self.point_cloud_data['tdc'][ii][(temp_view == v)] - tdc_mean)/tdc_std
                temp_channel = (self.point_cloud_data['channel'][ii][(temp_view == v)] - channel_mean)/channel_std
                temp_adc = (self.point_cloud_data['adc'][ii][(temp_view == v)] - adc_mean)/adc_std
                temp_summed_adc = np.sum(self.point_cloud_data['adc'][ii][(temp_view == v)])

                """
                We then need to impose translational symmetry by moving 
                each point cloud to the origin.
                """
                temp_tdc_mean = np.mean(temp_tdc)
                temp_channel_mean = np.mean(temp_channel)
                temp_adc_mean = np.mean(temp_adc)

                if len(temp_tdc) < 1:
                    continue

                for jj in range(len(temp_tdc)):
                    temp_pos.append([
                        (temp_tdc[jj] - temp_tdc_mean),
                        (temp_channel[jj] - temp_channel_mean),
                        (temp_adc[jj] - temp_adc_mean)
                    ])
                if (temp_pos != []):
                    positions.append(np.array(temp_pos))
                    group_labels.append(group_label_ids[ii])
                    group_names.append(group_label[ii])
                    labels.append(label_ids[ii])
                    label_names.append(label[ii])
                    total_energy.append(energy[ii])
                    summed_adc.append(temp_summed_adc)

            positions = np.array(positions)
            total_energy = np.array(total_energy)
            group_labels = np.array(group_labels)
            group_names = np.array(group_names)
            labels = np.array(labels)
            label_names = np.array(label_names)
            summed_adc = np.array(summed_adc)

            meta = {
                "who_created":      "me",
                "when_created":     datetime.now().strftime("%m-%d-%Y-%H:%M:%S"),
                "where_created":    socket.gethostname(),
                "num_events":       len(positions),
                "view":             v,
                "features": {
                    "tdc": 0, "channel": 1, "adc": 2
                },
                "group_classes": {
                    key: value
                    for key, value in group_label_map.items()
                },
                "classes": {
                    key: value
                    for key, value in label_map.items()
                },          
                "num_group_classes":    len(np.unique(group_labels)),
                "num_classes":          len(np.unique(labels)),
            }

            np.savez(
                f"data/point_cloud_view{v}.npz",
                positions=positions,
                energies=total_energy,
                summed_adc=summed_adc,
                group_labels=group_labels,
                group_names=group_names,
                labels=labels,
                label_names=label_names,
                meta=meta
            )

            label_hist, _ = np.histogram(labels, bins=len(unique_labels))
            label_hist = np.divide(label_hist, np.sum(label_hist, dtype=float), dtype=float)

            if plot_group_statistics:
                fig, axs = plt.subplots(figsize=(10,6))
                for label in unique_group_labels:
                    group_adc = summed_adc[(group_names == label)]
                    axs.hist(
                        group_adc, 
                        bins=100, 
                        range=[np.min(summed_adc),np.max(summed_adc)], 
                        histtype='step',
                        stacked=True,
                        density=True,
                        label=f"{label}"
                    )
                axs.set_xlabel("Summed ADC [counts]")
                axs.set_ylabel("Point Clouds")
                axs.set_title(f"Summed ADC Distribution for View {v}")
                axs.ticklabel_format(axis='x', style='sci')
                plt.legend()
                plt.tight_layout()
                plt.savefig(self.arrakis_dir + f"group_summed_adc_view{v}.png")

                fig, axs = plt.subplots(figsize=(10,6))
                for label in unique_labels[(label_hist > .01)]:
                    group_adc = summed_adc[(label_names == label)]
                    axs.hist(
                        group_adc, 
                        bins=100,
                        range=[np.min(summed_adc),np.max(summed_adc)], 
                        histtype='step',
                        stacked=True,
                        density=True,
                        label=f"{label}"
                    )
                axs.set_xlabel("Summed ADC [counts]")
                axs.set_ylabel("Point Clouds")
                axs.set_title(f"Summed ADC Distribution for View {v}")
                axs.ticklabel_format(axis='x', style='sci')
                plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
                plt.tight_layout()
                plt.savefig(self.arrakis_dir + f"individual_summed_adc_view{v}.png")

                # fig, axs = plt.subplots(figsize=(15,10))
                # axs.scatter(unique_labels[(label_hist > 10)], label_hist[(label_hist > 10)])
                # axs.set_xticks([ii+1 for ii in range(len(unique_labels[(label_hist > 10)]))])
                # axs.set_xticklabels(unique_labels[(label_hist > 10)], rotation=45, ha='right')
                # plt.show()
