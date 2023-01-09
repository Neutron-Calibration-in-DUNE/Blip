from ctypes import sizeof
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st


class Arrakis:
    def __init__(self,input_file):
        self.input_file = uproot.open(input_file)
        self.point_cloud_data = self.input_file['ana/solo_point_cloud'].arrays(library="np")

    def generate_training_data(self):
        view = self.point_cloud_data['view']
        views = np.concatenate(view)
        for v in np.unique(np.concatenate(view)):
            channel = self.point_cloud_data['channel']
            channel_mean = np.mean(np.concatenate(channel)[(views == v)])
            channel_std = np.std(np.concatenate(channel)[(views == v)])

            tdc = self.point_cloud_data['tdc']
            tdc_mean = np.mean(np.concatenate(tdc)[(views == v)])
            tdc_std = np.std(np.concatenate(tdc)[(views == v)])

            adc = self.point_cloud_data['adc']
            adc_mean = np.mean(np.concatenate(adc)[(views == v)])
            adc_std = np.std(np.concatenate(adc)[(views == v)])

            label = self.point_cloud_data['label']
            energy = self.point_cloud_data['total_energy'] * 10e5

            positions = []
            labels = []
            total_energy = []

            for ii in range(len(self.point_cloud_data['tdc'])):
                temp_pos = []
                temp_view = self.point_cloud_data['view'][ii]
                temp_tdc = (self.point_cloud_data['tdc'][ii][(temp_view == v)] - tdc_mean)/tdc_std
                temp_channel = (self.point_cloud_data['channel'][ii][(temp_view == v)] - channel_mean)/channel_std
                temp_adc = (self.point_cloud_data['adc'][ii][(temp_view == v)] - adc_mean)/adc_std

                temp_tdc_mean = np.mean(temp_tdc)
                temp_channel_mean = np.mean(temp_channel)
                temp_adc_mean = np.mean(temp_adc)

                for jj in range(len(temp_tdc)):
                    temp_pos.append([
                        (temp_tdc[jj] - temp_tdc_mean),
                        (temp_channel[jj] - temp_channel_mean),
                        (temp_adc[jj] - temp_adc_mean)
                    ])
                if (temp_pos != []):
                    positions.append(np.array(temp_pos))
                    labels.append(label[ii])
                    total_energy.append(energy[ii])

            np.savez(
                f"data/point_cloud_view{v}.npz",
                positions=np.array(positions),
                energies=np.array(total_energy),
                labels=np.array(labels)
            )