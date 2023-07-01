"""
Tools for displaying events
"""
import numpy as np
from matplotlib import pyplot as plt

from bokeh.io import curdoc, output_notebook, show
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.layouts import row, column, layout
from bokeh.plotting import figure, show
from bokeh.models import Div, RangeSlider, Spinner
from bokeh.models import Select, MultiSelect, FileInput
from bokeh.models import Button, CheckboxGroup, TextInput
from bokeh.models import CategoricalColorMapper, Toggle
from bokeh.models import CheckboxButtonGroup, CustomJS
from bokeh.models import Paragraph, PreText, Dropdown
from bokeh.models import ColumnDataSource, RadioGroup
from bokeh.palettes import Turbo256, Category20, Category20b, TolRainbow, Magma256
from bokeh.transform import linear_cmap
from bokeh.transform import factor_cmap, factor_mark
from bokeh.server.server import Server
from bokeh.command.util import build_single_handler_applications
from bokeh.document import Document

import pandas as pd

import os
from pathlib import Path
import imageio

from blip.utils.logger import Logger

class BlipDisplay:
    """
    """
    def __init__(self,
        document = None
    ):
        self.file_folder = str(Path().absolute())
        self.available_folders = []
        self.update_available_folders()
        self.available_files = []
        self.input_file = ''
        self.update_available_files()
        self.meta = {}
        self.available_events = []
        self.event = -1
        self.meta_vars = [
            "input_file", "who_created", "when_created",
            "where_created", "num_events", "view",
            "features", "classes"
        ]
        self.meta_vals = [
            '...', '...', '...', '...', '...', '...', '...', '...'
        ]
        self.meta_string = ''
        self.update_meta_string()
        self.features = []
        self.classes = []
        self.predictions = {}

        self.plot_options = ["Truth", "Predictions"]

        # parameters for first plot
        self.available_truth_labels = [
            'adc', 'source', 'shape', 'particle'
        ]
        self.first_figure_label = 'adc'
        self.first_scatter = {}
        self.first_figure_plot_type = "Truth"

        # parameters for second plot
        self.available_prediction_labels = []
        self.second_figure_label = ''
        self.second_scatter = {}
        self.second_figure_plot_type = "Predictions"
        
        if document == None:
            self.document = curdoc()
        else:
            self.document = document

        self.construct_widgets(self.document)

    def update_available_folders(self):
        self.available_folders = ['.', '..']
        folders = [
            f.parts[-1] for f in Path(self.file_folder).iterdir() if f.is_dir()
        ]
        if len(folders) > 0:
            folders.sort()
            self.available_folders += folders
        
    def update_available_files(self):
        self.available_files = [
            f.parts[-1] for f in Path(self.file_folder).iterdir() if f.is_file()
        ]
        if len(self.available_files) > 0:
            self.available_files.sort()

    def update_meta_string(self):
        self.meta_string = ''
        for ii, item in enumerate(self.meta_vars):
            self.meta_string += item
            self.meta_string += ":\t"
            self.meta_string += str(self.meta_vals[ii])
            self.meta_string += "\n"

    def update_available_events(self):
        self.available_events = [
            str(ii) for ii in range(int(self.meta["num_events"]))
        ]
        if len(self.available_events) > 0:
            self.event = 0

    def construct_widgets(self,
        document
    ):
        # Left hand column
        self.file_folder_select = Select(
            title=f"Blip folder: ~/{Path(self.file_folder).parts[-1]}",
            value=".",
            options=self.available_folders,
            width_policy='fixed', width=350
        )
        self.file_folder_select.on_change(
            "value", self.update_file_folder
        )
        self.file_select = Select(
            title="Blip file:", value="", 
            options=self.available_files,
            width_policy='fixed', width=350
        )
        if len(self.available_files) > 0:
            self.file_select.value = self.available_files[0]
            self.input_file = self.file_select.value
        self.file_select.on_change(
            "value", self.update_input_file
        )
        self.load_file_button = Button(
            label="Load file", 
            button_type="success",
            width_policy='fixed', width=100
        )
        self.load_file_button.on_click(
            self.load_input_file
        )
        self.meta_pretext = PreText(
            text=self.meta_string,
            width=200,
            height=200
        )
        self.event_select = Select(
            title="Event:", value="",
            options=self.available_events,
            width_policy='fixed', width=100
        )
        self.event_select.on_change(
            "value", self.update_event
        )
        self.load_event_button = Button(
            label="Load event", 
            button_type="success",
            width_policy='fixed', width=100
        )
        self.load_event_button.on_click(
            self.load_event
        )
        self.link_axes_toggle = Toggle(
            label="Link plots", 
            button_type="success"
        )
        self.link_axes_toggle.on_click(
            self.update_link_axes
        )

        # First plot column
        self.first_figure = figure(
            title="Plot I [Truth]",
            x_axis_label="x []",
            y_axis_label="y []"
        )
        self.first_figure.legend.click_policy="hide"
        self.first_figure_radio_text = PreText(
            text="Label type:"
        )
        self.first_figure_radio_group = RadioGroup(
            labels = self.plot_options, active=0
        )
        self.first_figure_radio_group.on_change(
            "active", self.update_first_figure_radio_group
        )
        self.first_figure_color_select = Select(
            title="Plot I labeling:", value="",
            options=self.available_truth_labels,
            width_policy='fixed', width=100
        )
        self.first_figure_color_select.on_change(
            "value", self.update_first_figure_color
        )
        self.first_figure_plot_button = Button(
            label="Plot event",
            button_type="success",
            width_policy='fixed', width=100
        )
        self.first_figure_plot_button.on_click(
            self.plot_first_event
        )

        # Second plot column
        self.second_figure = figure(
            title="Plot II [Predictions]",
            x_axis_label="x []",
            y_axis_label="y []",
            x_range=self.first_figure.x_range,
            y_range=self.first_figure.y_range
        )
        self.second_figure.legend.click_policy="hide"
        self.second_figure_radio_group = RadioGroup(
            labels = self.plot_options, active=1
        )
        self.second_figure_radio_group.on_change(
            "active", self.update_second_figure_radio_group
        )
        self.second_figure_color_select = Select(
            title="Plot II labeling:", value="",
            options=self.available_prediction_labels,
            width_policy='fixed', width=100
        )
        self.second_figure_color_select.on_change(
            "value", self.update_second_figure_color
        )
        self.second_figure_plot_button = Button(
            label="Plot event",
            button_type="success",
            width_policy='fixed', width=100
        )
        self.second_figure_plot_button = Button(
            label="Plot event",
            button_type="success",
            width_policy='fixed', width=100
        )
        self.second_figure_plot_button.on_click(
            self.plot_second_event
        )
        
        # construct the layout
        self.layout = row(
            column(
                self.file_folder_select,
                self.file_select,
                self.load_file_button,
                self.meta_pretext,
                self.event_select,
                self.load_event_button,
                self.link_axes_toggle,
                width_policy = 'fixed', width=400
            ),
            column(
                self.first_figure,
                self.first_figure_color_select,
                self.first_figure_radio_group,
                self.first_figure_plot_button,
                width_policy='fixed', width=600,
                height_policy='fixed', height=1000
            ),
            column(
                self.second_figure,
                self.second_figure_color_select,
                self.second_figure_radio_group,
                self.second_figure_plot_button,
                width_policy='fixed', width=600,
                height_policy='fixed', height=1000
            )
        )

        document.add_root(self.layout)
        document.title = "Blip Display"
    
    def update_file_folder(self, attr, old, new):
        if new == '..':
            self.file_folder = str(Path(self.file_folder).parent)
        elif new == '.':
            pass
        else:
            self.file_folder = str(Path(self.file_folder)) + "/" + new
        self.update_available_folders()
        self.file_folder_select.options = self.available_folders
        self.file_folder_select.title = title=f"Blip folder: ~/{Path(self.file_folder).parts[-1]}"
        self.file_folder_select.value = '.'

        self.update_available_files()
        self.file_select.options = self.available_files
        if len(self.available_files) > 0:
            self.file_select.value = self.available_files[0]
    
    def update_input_file(self, attr, old, new):
        self.input_file = new

    def update_meta(self):
        for ii, item in enumerate(self.meta_vars):
            if item in self.meta.keys():
                self.meta_vals[ii] = self.meta[item]
        self.meta_vals[0] = self.input_file
        self.update_meta_string()
        self.meta_pretext.text = self.meta_string
    
    def update_events(self):
        self.update_available_events()
        self.event_select.options = self.available_events
        self.event_select.value = str(self.event)

    def update_event(self, attr, old, new):
        self.event = int(self.event_select.value)
    
    def update_link_axes(self, new):
        if self.link_axes_toggle.active:
            self.second_figure.x_range = self.first_figure.x_range
            self.second_figure.y_range = self.first_figure.y_range

    def update_first_figure_radio_group(self, attr, old, new):
        if self.first_figure_radio_group.active == 0:
            self.first_figure_plot_type = "Truth"
            self.first_figure_color_select.options = self.available_truth_labels
            self.first_figure_color_select.value = self.available_truth_labels[0]
            self.first_figure_label = self.available_truth_labels[0]
        else:
            self.first_figure_plot_type = "Predictions"
            self.first_figure_color_select.options = self.available_prediction_labels
            if len(self.available_prediction_labels) > 0:
                self.first_figure_color_select.value = self.available_prediction_labels[0]
                self.first_figure_label = self.available_prediction_labels[0]
        self.first_figure.title.text = f"Plot I [{self.first_figure_plot_type}]:"
        
    def update_first_figure_color(self, attr, old, new):
        self.first_figure_label = self.first_figure_color_select.value

    def update_second_figure_radio_group(self, attr, old, new):
        if self.second_figure_radio_group.active == 0:
            self.second_figure_plot_type = "Truth"
            self.second_figure_color_select.options = self.available_truth_labels
            self.second_figure_color_select.value = self.available_truth_labels[0]
            self.second_figure_label = self.available_truth_labels[0]
        else:
            self.second_figure_plot_type = "Predictions"
            self.second_figure_color_select.options = self.available_prediction_labels
            if len(self.available_prediction_labels) > 0:
                self.second_figure_color_select.value = self.available_prediction_labels[0]
                self.second_figure_label = self.available_prediction_labels[0]
        self.second_figure.title.text = f"Plot II [{self.second_figure_plot_type}]:"
        
    def update_second_figure_color(self, attr, old, new):
        self.second_figure_label = self.second_figure_color_select.value

    def load_input_file(self):
        if self.input_file.endswith(".npz"):
            self.load_npz_file()
        elif self.input_file.endswith(".root"):
            self.load_root_file()
        else:
            print(f"Can't load file {self.input_file}.")
    
    def load_npz_file(self):
        input_file = np.load(
            self.file_folder + "/" + self.input_file, 
            allow_pickle=True
        )
        self.available_prediction_labels = []
        if 'meta' in input_file.files:
            self.meta = input_file['meta'].item()
            self.update_meta()
            self.update_events()
        if 'features' in input_file.files:
            self.features = input_file['features']
        if 'classes' in input_file.files:
            self.classes = input_file['classes']
        if 'source' in input_file.files:
            self.predictions['source'] = input_file['source']
            self.available_prediction_labels.append('source')
        if 'shape' in input_file.files:
            self.predictions['shape'] = input_file['shape']
            self.available_prediction_labels.append('shape')
        if 'particle' in input_file.files:
            self.predictions['particle'] = input_file['particle']
            self.available_prediction_labels.append('particle')
        if self.first_figure_plot_type == "Predictions":
            self.first_figure_color_select.options = self.available_prediction_labels
            if len(self.available_prediction_labels) > 0:
                self.first_figure_color_select.value = self.available_prediction_labels[0]
                self.first_figure_label = self.available_prediction_labels[0]
        if self.second_figure_plot_type == "Predictions":
            self.second_figure_color_select.options = self.available_prediction_labels
            if len(self.available_prediction_labels) > 0:
                self.second_figure_color_select.value = self.available_prediction_labels[0]
                self.second_figure_label = self.available_prediction_labels[0]

    
    def load_root_file(self):
        pass

    def load_event(self):
        if str(self.event) in self.available_events:
            self.event_features = self.features[self.event]
            self.event_classes = self.classes[self.event]
            self.event_predictions = {
                key: val[self.event][0]
                for key, val in self.predictions.items()
            }
        else:
            pass
    
    def plot_first_event(self):
        self.first_figure.renderers = []
        self.first_figure.legend.items = []
        if self.first_figure_label == 'adc':
            pass
        else:

            label_index = self.meta['classes'][self.first_figure_label]
            label_vals = self.meta[f"{self.first_figure_label}_labels"]
            self.first_scatter = {}
            self.first_scatter_colors = {
                #val: Magma256[len(label_vals)][ii]
                val: Magma256[int(ii*256/len(label_vals))]
                for ii, val in enumerate(label_vals.values())
            }
            for key, val in label_vals.items():   
                if self.first_figure_plot_type == "Truth": 
                    mask = (self.event_classes[:, label_index] == key)
                else:
                    if self.first_figure_label not in self.available_prediction_labels:
                        continue
                    labels = np.argmax(self.event_predictions[self.first_figure_label], axis=1)
                    mask = (labels == key)
                if np.sum(mask) == 0:
                    continue
                self.first_scatter[val] = self.first_figure.circle(
                    self.event_features[:,0][mask],
                    self.event_features[:,1][mask],
                    legend_label=val,
                    color=self.first_scatter_colors[val]
                )
        self.first_figure.legend.click_policy="hide"
        self.first_figure.xaxis[0].axis_label = "Channel [n]"
        self.first_figure.yaxis[0].axis_label = "TDC [10ns]"
    
    def plot_second_event(self):
        self.second_figure.renderers = []
        self.second_figure.legend.items = []
        if self.second_figure_label == 'adc':
            pass
        else:

            label_index = self.meta['classes'][self.second_figure_label]
            label_vals = self.meta[f"{self.second_figure_label}_labels"]
            self.second_scatter = {}
            self.second_scatter_colors = {
                #val: Magma256[len(label_vals)][ii]
                val: Magma256[int(ii*256/len(label_vals))]
                for ii, val in enumerate(label_vals.values())
            }
            for key, val in label_vals.items():   
                if self.second_figure_plot_type == "Truth": 
                    mask = (self.event_classes[:, label_index] == key)
                else:
                    if self.second_figure_label not in self.available_prediction_labels:
                        continue
                    labels = np.argmax(self.event_predictions[self.second_figure_label], axis=1)
                    mask = (labels == key)
                if np.sum(mask) == 0:
                    continue
                self.second_scatter[val] = self.second_figure.circle(
                    self.event_features[:,0][mask],
                    self.event_features[:,1][mask],
                    legend_label=val,
                    color=self.second_scatter_colors[val]
                )
        self.second_figure.legend.click_policy="hide"
        self.second_figure.xaxis[0].axis_label = "Channel [n]"
        self.second_figure.yaxis[0].axis_label = "TDC [10ns]"
            
