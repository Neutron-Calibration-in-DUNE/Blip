"""
Tools for displaying events
"""
import numpy as np
from matplotlib import pyplot as plt
import os
import imageio

from blip.utils.logger import Logger

class BlipDisplay:
    """
    """
    def __init__(self,
        name:   str,
        input_files: dict,
    ):
        self.name = name
        self.input_files = input_files

        self.logger = Logger(self.name, file_mode='w')
        self.logger.info(f"constructing blip display.")

        if not os.path.isdir("plots/.img/"):
            os.makedirs("plots/.img/")

        self.data = {
            key: np.load(self.input_files[key], allow_pickle=True)
            for key, value in self.input_files.items()
        }
        self.meta = {
            key: self.data[key]['meta'].item()
            for key, value in self.input_files.items()
        }

        self.number_classes = {
            key: self.meta[key]['num_group_classes']
            for key, value in self.input_files.items()
        }
        self.labels = {
            key: self.meta[key]['group_classes']
            for key, value in self.input_files.items()
        }
    
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
            fps=.5
        )
        

    def _create_class_gif_frame(self,
        pos,
        class_label,
        summed_adc,
        image_number,
        xlim:   list=[-1.0,1.0],
        ylim:   list=[-1.0,1.0],
    ):
        fig, axs = plt.subplots(figsize=(5,5))
        scatter = axs.scatter(
            pos[:,1],   # channel
            pos[:,0],   # tdc
            marker='o',
            s=50*pos[:,2],
            c=pos[:,2],
            label=r"$\Sigma$"+f" ADC: {summed_adc:.2f}"
        )
        axs.set_xlim(xmin=xlim[0] - 0.1 * (xlim[1] - xlim[0]),xmax=xlim[1] + 0.1 * (xlim[1] - xlim[0]))
        axs.set_ylim(ymin=ylim[0] - 0.1 * (ylim[1] - ylim[0]),ymax=ylim[1] + 0.1 * (ylim[1] - ylim[0]))
        axs.set_xlabel(f"Channel [id normalized]")
        axs.set_ylabel(f"TDC (ns normalized)")
        plt.title(f"Point cloud {image_number} for class {class_label}")
        plt.legend(loc='upper right')
        #plt.colorbar(scatter, ax=axs)
        plt.tight_layout()
        plt.savefig(
            f"plots/.img/img_{image_number}.png",
            transparent = False,
            facecolor = 'white'
        )
        plt.close()


