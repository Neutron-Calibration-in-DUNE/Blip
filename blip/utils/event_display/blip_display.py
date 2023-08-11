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
from bokeh.models import TabPanel, Tabs, TapTool
from bokeh.models import Div, RangeSlider, Spinner
from bokeh.models import Select, MultiSelect, FileInput
from bokeh.models import Button, CheckboxGroup, TextInput
from bokeh.models import CategoricalColorMapper, Toggle
from bokeh.models import CheckboxButtonGroup, CustomJS
from bokeh.models import Paragraph, PreText, Dropdown
from bokeh.models import ColumnDataSource, RadioGroup
from bokeh.events import Tap
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
        # Wire plane display
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

        self.simulation_wrangler_vars = [
            "pdg_code", "generator", "generator_label", 
            "particle_id", "particle_energy",
            "parent_track_id", "parent_pdg_code",
            "daughter_track_id", "progeny_track_id", "ancestry_track_id",
            "edep_id", "edep_process", "detsim", "random_detsim", "edep_detsim",
            "detsim_edep"
        ]
        self.simulation_wrangler_vals = [
            '...', '...', '...', '...','...',
            '...','...','...','...','...','...','...',
            '...','...','...','...'
        ]
        self.simulation_wrangler_string = ''
        self.update_simulation_wrangler_string()
        
        self.features = []
        self.classes = []
        self.clusters = []
        self.predictions = {}

        self.plot_options = ["Truth", "Predictions"]

        # parameters for first plot
        self.available_truth_labels = [
            'adc', 
            'source', 'topology', 'particle', 'physics', 
            'cluster_topology', 'cluster_particle', 'cluster_physics',
            'hit_mean', 'hit_rms', 'hit_amplitude', 'hit_charge'
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
    
    def update_simulation_wrangler_string(self):
        self.simulation_wrangler_string = ''
        for ii, item in enumerate(self.simulation_wrangler_vars):
            self.simulation_wrangler_string += item
            self.simulation_wrangler_string += ":\t"
            self.simulation_wrangler_string += str(self.simulation_wrangler_vals[ii])
            self.simulation_wrangler_string += "\n"

    def update_available_events(self):
        self.available_events = [
            str(ii) for ii in range(int(self.meta["num_events"]))
        ]
        if len(self.available_events) > 0:
            self.event = 0
    
    def update_first_figure_taptool(self, event):
        print(event.x, event.y)

    def update_second_figure_taptool(self):
        pass

    def construct_widgets(self,
        document
    ):
        self.construct_header_widgets(document)
        self.construct_blip_widgets(document)
        self.construct_edep_widgets(document)
        self.construct_wire_plane_widgets(document)
        self.construct_wire_plane_channel_widgets(document)
        self.construct_semantic_widgets(document)
        self.construct_point_net_embedding_widgets(document)

        # enumerate the different tabs
        self.header_tab = TabPanel(
            child=self.header_layout, title="Blip Display"
        )
        self.wire_plane_display_tab = TabPanel(
            child=self.wire_plane_layout, title="Wire-Plane"
        )
        self.wire_plane_channel_tab = TabPanel(
            child=self.wire_plane_channel_layout, title="Wire-Plane Channel"
        )
        self.edep_tab = TabPanel(
            child=self.edep_layout, title="Energy Deposit"
        )
        self.semantic_model_tab = TabPanel(
            child=self.semantic_model_layout, title="Semantic Model"
        )
        self.point_net_embedding_tab = TabPanel(
            child=self.point_net_embedding_layout, title="PointNet Embedding"
        )
        self.blip_tab = TabPanel(
            child=self.blip_layout, title="Blip Runner"
        )
        self.tab_layout = Tabs(tabs=[
            self.header_tab,
            self.blip_tab,
            self.edep_tab,
            self.wire_plane_display_tab,
            self.wire_plane_channel_tab,
            self.semantic_model_tab,
            self.point_net_embedding_tab
        ])
        document.add_root(self.tab_layout)
        document.title = "Blip Display"

    def construct_header_widgets(self,
        document
    ):
        self.neutrino_image = Div(text="""<img src="data/neutrino.png" alt="div_image">""", width=100, height=100)
        self.header_layout = row(self.neutrino_image)

    def construct_blip_widgets(self,
        document
    ):
        self.blip_layout = row()

    def construct_edep_widgets(self,
        document
    ):
        self.edep_layout = row()

    def construct_wire_plane_widgets(self,
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
            y_axis_label="y []",
            tools='pan,wheel_zoom,box_zoom,lasso_select,tap,reset,save'
        )
        self.first_figure.on_event(Tap, self.update_first_figure_taptool)
        # self.first_figure_taptool = TapTool(callback=self.update_first_figure_taptool)
        # self.first_figure.add_tools(self.first_figure_taptool)
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
        self.simulation_wrangler_pretext = PreText(
            text=self.simulation_wrangler_string,
            width=200,
            height=200
        )

        # Second plot column
        self.second_figure = figure(
            title="Plot II [Predictions]",
            x_axis_label="x []",
            y_axis_label="y []",
            x_range=self.first_figure.x_range,
            y_range=self.first_figure.y_range,
            tools='pan,wheel_zoom,box_zoom,lasso_select,tap,reset,save'
        )
        self.second_figure_taptool = self.second_figure.select(type=TapTool)
        self.second_figure_taptool.callback = self.update_second_figure_taptool()
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
        
        # construct the wire plane layout
        self.wire_plane_layout = row(
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
                self.simulation_wrangler_pretext,
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
    
    def construct_wire_plane_channel_widgets(self,
        document
    ):
        self.wire_plane_channel_layout = row()
    
    def construct_semantic_widgets(self,
        document
    ):
        self.semantic_model_layout = row()

    def construct_point_net_embedding_widgets(self,
        document
    ):
        self.point_net_embedding_layout = row()
    
    ######################## Header Display ##########################

    ######################### Blip Display ###########################

    ######################### Edep Display ###########################

    ###################### Wire Plane Display ########################
    """
    functions here are for updating the Wire Plane display left panel.
    """
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

    """
    functions here are for updating the Wire Plane display plots.
    """
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
        if 'clusters' in input_file.files:
            self.clusters = input_file['clusters']
        if 'hits' in input_file.files:
            self.hits = input_file['hits']
        if 'source' in input_file.files:
            self.predictions['source'] = input_file['source']
            self.available_prediction_labels.append('source')
        if 'topology' in input_file.files:
            self.predictions['topology'] = input_file['topology']
            self.available_prediction_labels.append('topology')
        if 'particle' in input_file.files:
            self.predictions['particle'] = input_file['particle']
            self.available_prediction_labels.append('particle')
        if 'physics' in input_file.files:
            self.predictions['physics'] = input_file['physics']
            self.available_prediction_labels.append('physics')
        if 'mc_maps' in self.meta.keys():
            self.mc_maps = self.meta['mc_maps']
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
            self.event_clusters = self.clusters[self.event]
            self.event_hits = self.hits[self.event]
            self.event_predictions = {
                key: val[self.event][0]
                for key, val in self.predictions.items()
            }
            if 'mc_maps' in self.meta.keys():
                self.event_pdg_maps = self.mc_maps['pdg_code'][self.event]
                self.event_parent_track_id_maps = self.mc_maps['parent_track_id'][self.event]
                self.event_ancestor_track_id_maps = self.mc_maps['ancestor_track_id'][self.event]
                self.event_ancestor_level_maps = self.mc_maps['ancestor_level'][self.event]
        else:
            pass
    
    def plot_first_event(self):
        self.first_figure.renderers = []
        self.first_figure.legend.items = []
        if self.first_figure_label == 'adc':
            pass
        else:
            if 'cluster' in self.first_figure_label:
                label_index = self.meta['clusters'][self.first_figure_label.replace('cluster_','')]
                label_vals = np.unique(self.event_clusters[:, label_index])
                self.first_scatter = {}
                self.first_scatter_colors = {
                    #val: Magma256[len(label_vals)][ii]
                    val: Magma256[int(ii % 256)]
                    for ii, val in enumerate(label_vals)
                }
                for val in label_vals:   
                    if self.first_figure_plot_type == "Truth": 
                        mask = (self.event_clusters[:, label_index] == val)
                    else:
                        if self.first_figure_label not in self.available_prediction_labels:
                            continue
                        labels = np.argmax(self.event_predictions[self.first_figure_label], axis=1)
                        mask = (labels == val)
                    if np.sum(mask) == 0:
                        continue
                    self.first_scatter[val] = self.first_figure.circle(
                        self.event_features[:,0][mask],
                        self.event_features[:,1][mask],
                        legend_label=str(val),
                        color=self.first_scatter_colors[val]
                    )
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
            if 'cluster' in self.second_figure_label:
                label_index = self.meta['clusters'][self.second_figure_label.replace('cluster_','')]
                label_vals = np.unique(self.event_clusters[:, label_index])
                self.second_scatter = {}
                self.second_scatter_colors = {
                    #val: Magma256[len(label_vals)][ii]
                    val: Magma256[int(ii % 256)]
                    for ii, val in enumerate(label_vals)
                }
                print(label_vals)
                for val in label_vals:   
                    if self.second_figure_plot_type == "Truth": 
                        mask = (self.event_clusters[:, label_index] == val)
                    else:
                        if self.second_figure_label not in self.available_prediction_labels:
                            continue
                        labels = np.argmax(self.event_predictions[self.second_figure_label], axis=1)
                        mask = (labels == val)
                    if np.sum(mask) == 0:
                        continue
                    self.second_scatter[val] = self.second_figure.circle(
                        self.event_features[:,0][mask],
                        self.event_features[:,1][mask],
                        legend_label=str(val),
                        color=self.second_scatter_colors[val]
                    )
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
    
    ###################### Wire Plane Display ########################
            
