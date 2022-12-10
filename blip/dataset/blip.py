from ctypes import sizeof
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st

file = uproot.open("data/full_sim.root")

array_Tree = file["ana/single_neutron"]
array_Branches = array_Tree.arrays(library="np")
print(array_Branches['tdc'])

nCap_Tree = file["ana/NeutronCapture"]
nCap_Branches = nCap_Tree.arrays(library="np")
print(nCap_Branches)

gamma_energies_real = [] #energies of the gammas produced in geant
gamma_number_real = [] #number of gammas produced in each event
# gamma_xMomen_real = []
# gamma_yMomen_real = []
# gamma_zMomen_real = []
with open('data/gamma_pointCloud_input_10000_0.dat') as f:
    num_gammas = -1
    gamma_e_temp = []
    # gamma_Px_temp = []
    # gamma_Py_temp = []
    # gamma_Pz_temp = []
    for line in f:
        data = line.split()
        data = np.array([eval(i) for i in data])
        if data.size == 2:
            num_gammas = data[1]
            gamma_number_real.append(data[1])
        if data.size != 2:
            gamma_e_temp.append(data[9])
            # gamma_Px_temp.append(data[6]/data[9])
            # gamma_Py_temp.append(data[7]/data[9])
            # gamma_Pz_temp.append(data[8]/data[9])
            num_gammas -= 1
        if num_gammas == 0:
            gamma_energies_real.append(gamma_e_temp[:])
            # gamma_xMomen_real.append(gamma_Px_temp[:])
            # gamma_yMomen_real.append(gamma_Py_temp[:])
            # gamma_zMomen_real.append(gamma_Pz_temp[:])
            gamma_e_temp.clear()
            # gamma_Px_temp.clear()
            # gamma_Py_temp.clear()
            # gamma_Pz_temp.clear()
gamma_energies_real = np.round(gamma_energies_real, 6)
unique_labels = np.unique(gamma_energies_real)
print(unique_labels)
labels = []
for gamma in gamma_energies_real:
    for ii, label in enumerate(unique_labels):
        if gamma == label:
            labels.append(ii)
labels = np.array(labels)
pos = []
tdc = np.concatenate(array_Branches['tdc'])
adc = np.concatenate(array_Branches['adc'])
channel = np.concatenate(array_Branches['channel'])


tdc_mean = np.mean(tdc)
tdc_std = np.std(tdc)
adc_mean = np.mean(adc)
adc_std = np.std(adc)
channel_mean = np.mean(channel)
channel_std = np.std(channel)

for ii in range(len(array_Branches['tdc'])):
    temp_pos = []
    for jj in range(len(array_Branches['tdc'][ii])):
        temp_pos.append([
            (array_Branches['tdc'][ii][jj] - tdc_mean)/tdc_std,
            (array_Branches['channel'][ii][jj] - channel_mean)/channel_std,
            (array_Branches['adc'][ii][jj] - adc_mean)/adc_std
        ])
    pos.append(np.array(temp_pos))
print(pos[0])

np.savez(
    "data/gamma_pointcloud.npz",
    pos=np.array(pos),
    energies=np.array(gamma_energies_real[:5000]),
    labels=np.array(labels[:5000])
)

import json
import os
import os.path as osp
import shutil
from typing import Callable, List, Optional, Union
import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import read_txt_array


class GammaDataset(InMemoryDataset):
    def __init__(self, root, num_samples, transform=None, pre_transform=None, pre_filter=None):
        self.num_samples = num_samples
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['../data/gamma_pointcloud.npz']

    @property
    def processed_file_names(self):
        return ['data.pt']
        ...

    def process(self):
        # Read data into huge `Data` list.
        data = np.load('../data/gamma_pointcloud.npz', allow_pickle=True)
        pos = data['pos'][:self.num_samples]
        y = data['labels'][:self.num_samples]
        data_list = [
            Data(
                pos=torch.tensor(pos[ii]).type(torch.float),
                x=torch.zeros(pos[ii].shape).type(torch.float),
                y=torch.full((len(pos[ii]),1),y[ii]).type(torch.long), 
                category=torch.tensor([y[ii]]).type(torch.long)
            )
            for ii in range(self.num_samples)
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])