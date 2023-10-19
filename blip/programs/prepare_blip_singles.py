"""
This script takes Arrakis outputs and constructs "singles" datasets
of Blip types to be used in optimizing BlipGraph. 
"""

import numpy as np
import matplotlib.pyplot as plt
import uproot
import os
import imageio
import csv
import argparse


def create_class_gif(self,
    input_file:     str,
    class_label:    str,
    num_events:     int=10,
):
    pos = self.data[input_file]['positions']
    summed_adc = self.data[input_file]['summed_adc']
    y = self.data[input_file]['group_labels']

    pos = pos[(y == self.labels[input_file][class_label])]
    summed_adc = summed_adc[(y == self.labels[input_file][class_label])]
    mins = np.min(np.concatenate(pos[:num_events]),axis=0)
    maxs = np.max(np.concatenate(pos[:num_events]),axis=0)

    gif_frames = []
    for ii in range(min(num_events,len(pos))):
        self._create_class_gif_frame(
            pos[ii],
            class_label,
            summed_adc[ii],
            ii,
            [mins[1], maxs[1]],
            [mins[0], maxs[0]]
        )
        gif_frames.append(
            imageio.v2.imread(
                f"plots/.img/img_{ii}.png"
            )
        )
    imageio.mimsave(
        f"plots/{input_file}_{class_label}.gif",
        gif_frames,
        duration=2000
    )

def create_class_gif_frame(
    pos,
    class_label,
    summed_adc,
    image_number,
    xlim:   list=[-1.0,1.0],
    ylim:   list=[-1.0,1.0],
):
    fig, axs = plt.subplots(figsize=(8,8))
    scatter = axs.scatter(
        pos[0],   # channel
        pos[1],   # tdc
        marker='o',
        s=pos[2],
        c=pos[2],
        label=r"$\Sigma$"+f" ADC: {summed_adc:.2f}"
    )
    axs.set_xlim(-1.2,1.2)
    axs.set_ylim(-1.2,1.2)
    axs.set_xlabel(f"Channel [id normalized]")
    axs.set_ylabel(f"TDC (ns normalized)")
    plt.title(f"Point cloud {image_number} for class {class_label}")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(
        f"plots/.img/img_{image_number}.png",
        transparent = False,
        facecolor = 'white'
    )
    plt.close()

def generate_decay_singles(
    files,
    decay_label,
    max_events:     int=10000,
    channel_min:    int=4200,
    channel_max:    int=4600,
    tdc_min:        int=1000,
    tdc_max:        int=5000,
    consolidate:    bool=False,
):
    """
    This function takes in a set of radioactive decay files generated
    from Arrakis, and separates out individual instances
    of a particular decay into different events.
    """
    if not os.path.isdir("plots/"):
        os.makedirs("plots/")
    if not os.path.isdir("plots/.img"):
        os.makedirs("plots/.img")

    decay_types = {
        16: "ar39",
        17: "ar42",
        18: "k42",
        19: "kr85",
        20: "rn222",
        21: "po218a",
        22: "po218b",
        23: "at218a",
        24: "at218b",
        25: "rn218",
        26: "pb214",
        27: "bi214a",
        28: "bi214b",
        29: "po214",
        30: "tl210",
        31: "pb210a",
        32: "pb210b",
        33: "bi210a",
        34: "bi210b",
        35: "po210",
    }

    decay_energies = {
        16: "0.565",
        17: "0.598",
        18: "3.525",
        19: "0.687",
        20: "5.590",
        21: "6.114",
        22: "0.259",
        23: "6.874",
        24: "2.881",
        25: "7.262",
        26: "1.018",
        27: "5.621",
        28: "3.269",
        29: "7.833",
        30: "5.482",
        31: "3.792",
        32: "0.063",
        33: "5.036",
        34: "1.161",
        35: "5.407",
    } 

    decay_type = decay_types[decay_label]
    decay_energy = decay_energies[decay_label]
    s_adc = []
    energies = []
    if consolidate:
        new_wire_plane = {
            'channel':  [],
            'wire':     [],
            'tick':     [],
            'tdc':      [],
            'adc':      [],
            'view':     [],
            'energy':   [],
            'source_label': [],
            'topology_label':  [],
            'particle_label':   [],
            'physics_label':   [],
            'unique_topology':     [],
            'unique_particle':  [],
            'unique_physics':   [],
            'hit_mean': [],
            'hit_rms':  [],
            'hit_amplitude': [],
            'hit_charge':   [],
        }
    for ii, file in enumerate(files):
        with uproot.open(file) as f:
            wire_plane = f['ana/mc_wire_plane_point_cloud'].arrays(library="np")
            
            channel = wire_plane['channel']
            wire = wire_plane['wire']
            tick = wire_plane['tick']
            tdc = wire_plane['tdc']
            adc = wire_plane['adc']
            view = wire_plane['view']
            energy = wire_plane['energy']
            source_label = wire_plane['source_label']
            topology_label = wire_plane['topology_label']
            particle_label = wire_plane['particle_label']
            physics_label = wire_plane['physics_label']
            unique_topology = wire_plane['unique_topology']
            unique_particle = wire_plane['unique_particle']
            unique_physics = wire_plane['unique_physics']
            hit_mean = wire_plane['hit_mean']
            hit_rms = wire_plane['hit_rms']
            hit_amplitude = wire_plane['hit_amplitude']
            hit_charge = wire_plane['hit_charge']

            num_events = len(channel)
            if not consolidate:
                new_wire_plane = {
                    'channel':  [],
                    'wire':     [],
                    'tick':     [],
                    'tdc':      [],
                    'adc':      [],
                    'view':     [],
                    'energy':   [],
                    'source_label': [],
                    'topology_label':  [],
                    'particle_label':   [],
                    'physics_label':   [],
                    'unique_topology':     [],
                    'unique_particle':  [],
                    'unique_physics':   [],
                    'hit_mean': [],
                    'hit_rms':  [],
                    'hit_amplitude': [],
                    'hit_charge':   [],
                }
            class_label = f'Radioactive decay - {decay_type} {decay_energy} MeV'
            gif_frames = []

            # select events within the channel/tdc cuts
            saved_events = 0
            for event in range(num_events):
                if saved_events >= max_events:
                    break
                mask = (
                    (view[event] == 2) & 
                    (channel[event] >= channel_min) & 
                    (channel[event] <= channel_max) & 
                    (tdc[event] >= tdc_min) & 
                    (tdc[event] <= tdc_max)
                )
                channel[event] = channel[event][mask]
                wire[event] = wire[event][mask]
                tick[event] = tick[event][mask]
                tdc[event] = tdc[event][mask]
                adc[event] = adc[event][mask]
                view[event] = view[event][mask]
                energy[event] = energy[event][mask]
                source_label[event] = source_label[event][mask]
                topology_label[event] = topology_label[event][mask]
                particle_label[event] = particle_label[event][mask]
                physics_label[event] = physics_label[event][mask]
                unique_topology[event] = unique_topology[event][mask]
                unique_particle[event] = unique_particle[event][mask]
                unique_physics[event] = unique_physics[event][mask]
                hit_mean[event] = hit_mean[event][mask]
                hit_rms[event] = hit_rms[event][mask]
                hit_amplitude[event] = hit_amplitude[event][mask]
                hit_charge[event] = hit_charge[event][mask]

                # separate each unique instance of decays
                # into different events in the new file.
                for shape in np.unique(unique_topology[event]):
                    mask = (unique_topology[event] == shape)
                    if shape == -1:
                        continue
                    if sum(mask) < 3:
                        continue
                    if np.sum(adc[event][mask]) == 0:
                        continue
                    saved_events += 1
                    if saved_events >= max_events:
                        break
                    energies.append([np.sum(energy[event][mask])])
                    new_wire_plane['channel'].append(channel[event][mask])
                    new_wire_plane['wire'].append(wire[event][mask])
                    new_wire_plane['tick'].append(tick[event][mask])
                    new_wire_plane['tdc'].append(tdc[event][mask])
                    new_wire_plane['adc'].append(adc[event][mask])
                    new_wire_plane['view'].append(view[event][mask])
                    new_wire_plane['energy'].append(energy[event][mask])
                    new_wire_plane['source_label'].append(source_label[event][mask])
                    new_wire_plane['topology_label'].append(topology_label[event][mask])
                    new_wire_plane['particle_label'].append(particle_label[event][mask])
                    new_wire_plane['physics_label'].append(physics_label[event][mask])
                    new_wire_plane['unique_topology'].append(unique_topology[event][mask])
                    new_wire_plane['unique_particle'].append(unique_particle[event][mask])
                    new_wire_plane['unique_physics'].append(unique_physics[event][mask])
                    new_wire_plane['hit_mean'].append(hit_mean[event][mask])
                    new_wire_plane['hit_rms'].append(hit_rms[event][mask])
                    new_wire_plane['hit_amplitude'].append(hit_amplitude[event][mask])
                    new_wire_plane['hit_charge'].append(hit_charge[event][mask])

                    pos = np.vstack((channel[event][mask], tdc[event][mask], np.abs(adc[event][mask]))).astype(float)
                    summed_adc = np.sum(adc[event][mask])
                    s_adc.append([summed_adc])

                    # create animated GIF of events
                    mins = np.min(pos, axis=1)
                    maxs = np.max(pos, axis=1)
                    for kk in range(len(pos)-1):
                        denom = (maxs[kk] - mins[kk])
                        if denom == 0:
                            pos[kk] = 0 * pos[kk]
                        else:
                            pos[kk] = 2 * (pos[kk] - mins[kk])/(denom) - 1

                    create_class_gif_frame(
                        pos,
                        class_label,
                        summed_adc,
                        event,
                        [mins[1], maxs[1]],
                        [mins[0], maxs[0]]
                    )
                    gif_frames.append(
                        imageio.v2.imread(
                            f"plots/.img/img_{event}.png"
                        )
                    )

            if len(gif_frames) == 0:
                continue
            imageio.mimsave(
                f"plots/single_decay_{decay_type}.{ii}_{class_label}.gif",
                gif_frames,
                duration=2000
            )

            # save new events to new tree
            if not consolidate:
                with uproot.recreate(f"single_decay_{decay_type}.{ii}.root") as r:
                    r['ana/mc_wire_plane_point_cloud'] = new_wire_plane

    if consolidate:
        with uproot.recreate(f"single_decay_{decay_type}.root") as r:
            r['ana/mc_wire_plane_point_cloud'] = new_wire_plane

    fig, axs = plt.subplots()
    axs.hist(energies, bins=50)
    axs.set_xlabel("Energy [MeV]")
    axs.set_ylabel("Counts")
    axs.set_title(f"Energy Distribution for {class_label}")
    plt.tight_layout()
    plt.savefig(f'plots/single_decay_{decay_type}_{decay_energy}_energies.png')

    with open(f"single_decay_{decay_type}_summed_adc.csv", "w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerows(s_adc)
    with open(f"single_decay_{decay_type}_summed_energy.csv", "w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerows(energies)


def generate_capture_gamma_singles(
    files,
    gamma_label,
    channel_min:    int=4200,
    channel_max:    int=4600,
    tdc_min:        int=1000,
    tdc_max:        int=5000,
    consolidate:    bool=False,
):
    """
    This function takes in a set of capture gamma files of
    a specific energy and generates a new root file where each gamma is 
    separated into a different event.

    It is assumed that Arrakis does not properly label single gammas
    with respect to the neutron captures, since they are generated
    individually.  This is why we have to tell this function
    what the label of the gamma is.
    """
    if not os.path.isdir("plots/"):
        os.makedirs("plots/")
    if not os.path.isdir("plots/.img"):
        os.makedirs("plots/.img")

    gamma_energies = {
        8:  "4.745",
        9:  "3.365",
        10: "2.566",
        11: "1.186",
        12: "0.837",
        13: "0.516",
        14: "0.167",
    } 

    gamma_energy = gamma_energies[gamma_label]
    s_adc = []
    energies = []

    if consolidate:
        new_wire_plane = {
            'channel':  [],
            'wire':     [],
            'tick':     [],
            'tdc':      [],
            'adc':      [],
            'view':     [],
            'energy':   [],
            'source_label': [],
            'topology_label':  [],
            'particle_label':   [],
            'physics_label':   [],
            'unique_topology':     [],
            'unique_particle':  [],
            'unique_physics':   [],
            'hit_mean': [],
            'hit_rms':  [],
            'hit_amplitude': [],
            'hit_charge':   [],
        }

    for ii, file in enumerate(files):
        with uproot.open(file) as f:
            wire_plane = f['ana/mc_wire_plane_point_cloud'].arrays(library="np")
            
            channel = wire_plane['channel']
            wire = wire_plane['wire']
            tick = wire_plane['tick']
            tdc = wire_plane['tdc']
            adc = wire_plane['adc']
            view = wire_plane['view']
            energy = wire_plane['energy']
            source_label = wire_plane['source_label']
            topology_label = wire_plane['topology_label']
            particle_label = wire_plane['particle_label']
            physics_label = wire_plane['physics_label']
            unique_topology = wire_plane['unique_topology']
            unique_particle = wire_plane['unique_particle']
            unique_physics = wire_plane['unique_physics']
            hit_mean = wire_plane['hit_mean']
            hit_rms = wire_plane['hit_rms']
            hit_amplitude = wire_plane['hit_amplitude']
            hit_charge = wire_plane['hit_charge']

            num_events = len(channel)
            for event in range(num_events):
                mask = (physics_label[event] != 0) & (physics_label[event] != -1)
                physics_label[event][mask] = gamma_label
            
            if not consolidate:
                new_wire_plane = {
                    'channel':  [],
                    'wire':     [],
                    'tick':     [],
                    'tdc':      [],
                    'adc':      [],
                    'view':     [],
                    'energy':   [],
                    'source_label': [],
                    'topology_label':  [],
                    'particle_label':   [],
                    'physics_label':   [],
                    'unique_topology':     [],
                    'unique_particle':  [],
                    'unique_physics':   [],
                    'hit_mean': [],
                    'hit_rms':  [],
                    'hit_amplitude': [],
                    'hit_charge':   [],
                }
            class_label = f'Neutron Capture Gamma {gamma_energy} MeV'
            gif_frames = []

            # select events within the channel/tdc cuts
            for event in range(num_events):
                mask = (
                    (view[event] == 2) & 
                    (channel[event] >= channel_min) & 
                    (channel[event] <= channel_max) & 
                    (tdc[event] >= tdc_min) & 
                    (tdc[event] <= tdc_max)
                )
                channel[event] = channel[event][mask]
                wire[event] = wire[event][mask]
                tick[event] = tick[event][mask]
                tdc[event] = tdc[event][mask]
                adc[event] = adc[event][mask]
                view[event] = view[event][mask]
                energy[event] = energy[event][mask]
                source_label[event] = source_label[event][mask]
                topology_label[event] = topology_label[event][mask]
                particle_label[event] = particle_label[event][mask]
                physics_label[event] = physics_label[event][mask]
                unique_topology[event] = unique_topology[event][mask]
                unique_particle[event] = unique_particle[event][mask]
                unique_physics[event] = unique_physics[event][mask]
                hit_mean[event] = hit_mean[event][mask]
                hit_rms[event] = hit_rms[event][mask]
                hit_amplitude[event] = hit_amplitude[event][mask]
                hit_charge[event] = hit_charge[event][mask]

                # separate each unique instance of gammas
                # into different events in the new file.
                for shape in np.unique(unique_topology[event]):
                    mask = (unique_topology[event] == shape)
                    if shape == -1:
                        continue
                    if sum(mask) < 3:
                        continue
                    if np.sum(adc[event][mask]) == 0:
                        continue
                    energies.append([np.sum(energy[event][mask])])
                    new_wire_plane['channel'].append(channel[event][mask])
                    new_wire_plane['wire'].append(wire[event][mask])
                    new_wire_plane['tick'].append(tick[event][mask])
                    new_wire_plane['tdc'].append(tdc[event][mask])
                    new_wire_plane['adc'].append(adc[event][mask])
                    new_wire_plane['view'].append(view[event][mask])
                    new_wire_plane['energy'].append(energy[event][mask])
                    new_wire_plane['source_label'].append(source_label[event][mask])
                    new_wire_plane['topology_label'].append(topology_label[event][mask])
                    new_wire_plane['particle_label'].append(particle_label[event][mask])
                    new_wire_plane['physics_label'].append(physics_label[event][mask])
                    new_wire_plane['unique_topology'].append(unique_topology[event][mask])
                    new_wire_plane['unique_particle'].append(unique_particle[event][mask])
                    new_wire_plane['unique_physics'].append(unique_physics[event][mask])
                    new_wire_plane['hit_mean'].append(hit_mean[event][mask])
                    new_wire_plane['hit_rms'].append(hit_rms[event][mask])
                    new_wire_plane['hit_amplitude'].append(hit_amplitude[event][mask])
                    new_wire_plane['hit_charge'].append(hit_charge[event][mask])

                    pos = np.vstack((channel[event][mask], tdc[event][mask], np.abs(adc[event][mask]))).astype(float)
                    summed_adc = np.sum(adc[event][mask])
                    s_adc.append([summed_adc])

                    # create animated GIF of events
                    mins = np.min(pos, axis=1)
                    maxs = np.max(pos, axis=1)
                    for kk in range(len(pos)-1):
                        denom = (maxs[kk] - mins[kk])
                        if denom == 0:
                            pos[kk] = 0 * pos[kk]
                        else:
                            pos[kk] = 2 * (pos[kk] - mins[kk])/(denom) - 1

                    create_class_gif_frame(
                        pos,
                        class_label,
                        summed_adc,
                        event,
                        [mins[1], maxs[1]],
                        [mins[0], maxs[0]]
                    )
                    gif_frames.append(
                        imageio.v2.imread(
                            f"plots/.img/img_{event}.png"
                        )
                    )

            if len(gif_frames) == 0:
                continue
            imageio.mimsave(
                f"plots/single_capture_gamma_{gamma_energy}.{ii}_{class_label}.gif",
                gif_frames,
                duration=2000
            )

            # save new events to new tree
            if not consolidate:
                with uproot.recreate(f"single_capture_gamma_{gamma_energy}.{ii}.root") as r:
                    r['ana/mc_wire_plane_point_cloud'] = new_wire_plane

    if consolidate:
        with uproot.recreate(f"single_capture_gamma_{gamma_energy}.root") as r:
            r['ana/mc_wire_plane_point_cloud'] = new_wire_plane

    fig, axs = plt.subplots()
    axs.hist(energies, bins=50)
    axs.set_xlabel("Energy [MeV]")
    axs.set_ylabel("Counts")
    axs.set_title(f"Energy Distribution for {class_label}")
    plt.tight_layout()
    plt.savefig(f'plots/single_capture_gamma_{gamma_energy}_energies.png')

    with open(f"single_capture_gamma_{gamma_energy}_summed_adc.csv", "w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerows(s_adc)
    with open(f"single_capture_gamma_{gamma_energy}_summed_energy.csv", "w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerows(energies)