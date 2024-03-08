"""
Tools for displaying events
"""
import os.path as osp
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
import torch
from bokeh.events import Tap
from bokeh.models import ColorBar, LinearColorMapper, PreText, Slider, Toggle
from bokeh.palettes import Category20, Category20b, Magma256, TolRainbow, Turbo256
from panel.layout import Column, Row
from panel.widgets import (  # Replaces `Slider`, `Toggle`, `PreText`
    Button,
    Checkbox,
    FloatSlider,
    RadioButtonGroup,
    Select,
    TextInput,
    # Toggle,
)

from blip.utils.logger import Logger


class PixelLArTPCPanelDisplay:
    """
    Tools for displaying events
    """

    def __init__(self, document=None):
        # File folder and file select.
        # These act as drop down menus which update upon
        # selecting, one for TPCs and another for
        # edeps.
        self.file_folder = str(Path().absolute())
        self.input_file = ""
        self.available_folders = []
        self.available_files = []
        self.update_available_folders()
        self.update_available_files()

        # Meta information from blip dataset files (.npz)
        self.tpc_meta = {}
        self.available_events = []
        self.event = -1
        self.tpc_meta_vars = [
            "input_file",
            "who_created",
            "when_created",
            "where_created",
            "num_events",
            "view",
            "features",
            "classes",
        ]
        self.tpc_meta_vals = ["...", "...", "...", "...", "...", "...", "...", "..."]
        self.tpc_meta_string = ""
        self.update_meta_string()

        # Arrakis simulation wrangler mc truth maps
        # which allow us to query variables through
        # the track id of the particle.
        self.simulation_wrangler_vars = [
            "pdg_code",
            "generator",
            "generator_label",
            "particle_id",
            "particle_energy",
            "parent_track_id",
            "parent_pdg_code",
            "daughter_track_id",
            "progeny_track_id",
            "ancestry_track_id",
            "id",
            "process",
            "detsim",
            "random_detsim",
            "detsim",
            "detsim_edep",
        ]
        self.simulation_wrangler_vals = [
            "...",
            "...",
            "...",
            "...",
            "...",
            "...",
            "...",
            "...",
            "...",
            "...",
            "...",
            "...",
            "...",
            "...",
            "...",
            "...",
        ]
        self.simulation_wrangler_string = ""
        self.update_simulation_wrangler_string()

        # data from blip dataset files for a given
        # event.
        self.features = {0: 0}
        self.classes = {0: 0}
        self.clusters = {0: 0}
        self.hits = {0: 0}
        self.edeps = {0: 0}
        self.predictions = {}

        self.plot_types = ["TPC", "Merge Tree"]
        self.plot_options = ["Truth", "Predictions"]
        self.options = []
        self.tpc_options = []
        self.merge_tree_options = []

        # parameters for wire_plane plots
        self.available_truth_labels = [
            "adc",
            "source",
            "topology",
            "particle",
            "physics",
            "cluster_topology",
            "cluster_particle",
            "cluster_physics",
            "hit_mean",
            "hit_rms",
            "hit_amplitude",
            "hit_charge",
        ]
        self.available_prediction_labels = ["None"]
        self.available_wire_channel_truth_labels = ["None"]
        self.available_wire_channel_prediction_labels = ["None"]
        self.available_truth_labels = [
            "energy",
            "num_photons",
            "num_electrons",
            "source",
            "topology",
            "particle",
            "physics",
            "cluster_topology",
            "cluster_particle",
            "cluster_physics",
        ]
        self.available_prediction_labels = ["None"]
        self.available_merge_tree_truth_labels = ["cluster_particle", "MergeTree"]
        self.available_merge_tree_prediction_labels = ["None"]

        self.figure1_label = "adc"
        self.first_scatter = {}
        self.figure1_plot_type = "TPC"

        # parameters for second plot
        self.available_prediction_labels = []
        self.figure2_label = ""
        self.second_scatter = {}
        self.figure2_plot_type = "TPC"

        self.document = document

        self.construct_widgets(self.document)

    def update_available_folders(self):
        self.available_folders = [".", ".."]
        folders = [
            f.parts[-1]
            for f in Path(self.file_folder).iterdir()
            if f.is_dir()
        ]
        if len(folders) > 0:
            folders.sort()
            self.available_folders += folders

    def update_available_files(self):
        self.available_files = [
            f.parts[-1]
            for f in Path(self.file_folder).iterdir()
            if f.is_file()
        ]
        if len(self.available_files) > 0:
            self.available_files.sort()

    def update_meta_string(self):
        self.tpc_meta_string = ""
        for ii, item in enumerate(self.tpc_meta_vars):
            self.tpc_meta_string += item
            self.tpc_meta_string += ":\t"
            self.tpc_meta_string += str(self.tpc_meta_vals[ii])
            self.tpc_meta_string += "\n"

    def update_simulation_wrangler_string(self):
        self.simulation_wrangler_string = ""
        for ii, item in enumerate(self.simulation_wrangler_vars):
            self.simulation_wrangler_string += item
            self.simulation_wrangler_string += ":\t"
            self.simulation_wrangler_string += str(self.simulation_wrangler_vals[ii])
            self.simulation_wrangler_string += "\n"

    def update_available_events(self):
        self.available_events = [
            str(ii) for ii in range(int(self.tpc_meta["num_events"]))
        ]
        if len(self.available_events) > 0:
            self.event = 0

    def update_figure1_taptool(self, event):
        print(event.x, event.y)

    def update_figure2_taptool(self):
        pass

    def construct_widgets(self, document):
        # Left hand column
        self.construct_left_hand_column()
        self.construct_figure1_column()
        self.construct_figure2_column()

    def construct_left_hand_column(self):
        # Folder select
        self.file_folder_select = Select(
            name=f"Blip folder: ~/{Path(self.file_folder).parts[-1]}",
            value=".",
            options=self.available_folders,
            width_policy="fixed",
            width=350,
        )
        self.file_folder_select.param.watch(self.update_file_folder, "value")
        # File select
        self.file_select = Select(
            name="Blip file:",
            value="",
            options=self.available_files,
            width_policy="fixed",
            width=350,
        )
        if len(self.available_files) > 0:
            self.file_select.value = self.available_files[0]
            self.input_file = self.file_select.value
        self.file_select.param.watch(self.update_input_file, "value")
        # Load File button
        self.load_file_button = Button(
            name="Load file", button_type="primary", width_policy="fixed", width=100
        )
        self.load_file_button.on_click(self.load_input_file)
        # Meta information
        self.tpc_meta_pretext = PreText(
            text=self.tpc_meta_string, width=200, height=200
        )
        self.event_select = Select(
            name="Event:",
            value="",
            options=self.available_events,
            width_policy="fixed",
            width=100,
        )
        self.event_select.param.watch(self.update_event, "value")
        self.load_event_button = Button(
            name="Load event", button_type="primary", width_policy="fixed", width=100
        )
        self.load_event_button.on_click(self.load_event)
        self.link_axes_toggle = Toggle(label="Link plots", button_type="success")
        self.link_axes_toggle.on_click(self.update_link_axes)

        # First plot column
    def construct_figure1_column(self):
        self.figure1_event_features = []
        self.figure1_event_classes = []
        self.figure1_event_clusters = []
        self.figure1_event_hits = []
        self.figure1 = go.Figure(
            layout=dict(
                template="presentation",
                title="Plot I [TPC Truth]",
                scene=dict(
                    xaxis=dict(title="x []"),
                    yaxis=dict(title="y []"),
                    zaxis=dict(title="z []")
                ),
                coloraxis=dict(colorbar=dict(title=""), colorscale="Viridis"),
            )
        )
        self.figure1 = self.figure1.add_trace(go.Scatter3d())
        self.figure1.data[0].on_click(self.update_figure1_taptool)
        self.figure1_adc_slider_option = Checkbox(
            name="Use ADC Slider", value=False
        )
        self.figure1_adc_slider_option.param.watch(
            self.update_figure1_adc_slider_option, "value"
        )
        self.figure1_adc_slider_option_bool = False
        self.figure1_slider = Slider(start=0.1, end=1, step=0.1, value=0.1)
        # Plot type radio group
        self.figure1_plot_type = "TPC"
        self.figure1_radio_text = PreText(text="Plot I type:")
        self.figure1_radio_group = RadioButtonGroup(options=self.plot_types)
        self.figure1_radio_group.param.watch(
            self.update_figure1_radio_group, "value"
        )
        # Plot type labeling options
        self.figure1_color_select = Select(
            name="Plot I labeling:",
            value="",
            options=self.available_truth_labels,
            width_policy="fixed",
            width=150,
        )
        self.figure1_color_select.param.watch(
            self.update_figure1_color, "value"
        )
        # Plot options (truth/predictions)
        self.figure1_plot_option = "Truth"
        self.figure1_plot_option_text = PreText(text="Truth/Predictions:")
        self.figure1_plot_options = RadioButtonGroup(
            options=self.plot_options, value=0
        )
        self.figure1_plot_options.param.watch(
            self.update_figure1_radio_group, "value"
        )
        # Plot type options
        self.figure1_plot_type_options = Select(
            name="Plot I options:",
            value="",
            options=self.options,
            width_policy="fixed",
            width=150,
        )
        self.figure1_plot_type_options.param.watch(
            self.update_figure1_plot_type_options, "value"
        )
        # Plot button
        self.figure1_plot_button = Button(
            name="Plot event", button_type="primary", width_policy="fixed", width=100
        )
        self.figure1_plot_button.on_click(self.plot_first_event)
        # Plot text information
        self.simulation_wrangler_pretext = PreText(
            text=self.simulation_wrangler_string, width=200, height=200
        )
        self.figure1_pane = pn.pane.Plotly(self.figure1)

        # Second plot column
    def construct_figure2_column(self):
        self.figure2_event_features = []
        self.figure2_event_classes = []
        self.figure2_event_clusters = []
        self.figure2_event_hits = []
        self.figure2 = go.Figure(
            layout=dict(
                template="presentation",
                title="Plot II [Predictions]",
                scene=dict(
                    xaxis=dict(title="x []"),
                    yaxis=dict(title="y []"),
                    zaxis=dict(title="z []")
                ),
                coloraxis=dict(colorbar=dict(title=""), colorscale="Viridis"),
            )
        )

        self.figure2 = self.figure2.add_trace(go.Scatter3d())
        self.figure2.data[0].on_click(self.update_figure2_taptool)

        self.figure2_adc_slider_option = Checkbox(
            name="Use ADC Slider", value=False
        )
        self.figure2_adc_slider_option.param.watch(
            self.update_figure2_adc_slider_option, "value"
        )
        self.figure2_adc_slider_option_bool = False
        self.figure2_slider = Slider(start=0.1, end=1, step=0.1, value=0.1)
        # self.figure2.legend.click_policy="hide"
        # Plot II type
        self.figure2_plot_type = "TPC"
        self.figure2_radio_text = PreText(text="Plot II type:")
        self.figure2_radio_group = RadioButtonGroup(
            options=self.plot_types, value=1
        )
        self.figure2_radio_group.param.watch(
            self.update_figure2_radio_group, "value"
        )
        # Plot II labeling
        self.figure2_color_select = Select(
            name="Plot II labeling:",
            value="",
            options=self.available_prediction_labels,
            width_policy="fixed",
            width=150,
        )
        self.figure2_color_select.param.watch(
            self.update_figure2_color, "value"
        )
        # Plot options (truth/predictions)
        self.figure2_plot_option = "Truth"
        self.figure2_plot_option_text = PreText(text="Truth/Predictions:")
        self.figure2_plot_options = RadioButtonGroup(
            options=self.plot_options, value=0
        )
        self.figure2_plot_options.param.watch(
            self.update_figure2_radio_group, "value"
        )
        # Plot type options
        self.figure2_plot_type_options = Select(
            name="Plot I options:",
            value="",
            options=self.options,
            width_policy="fixed",
            width=150,
        )
        self.figure2_plot_type_options.param.watch(
            self.update_figure2_plot_type_options, "value"
        )
        self.figure2_plot_button = Button(
            name="Plot event", button_type="primary", width_policy="fixed", width=100
        )
        self.figure2_plot_button = Button(
            name="Plot event", button_type="primary", width_policy="fixed", width=100
        )
        self.figure2_plot_button.on_click(self.plot_second_event)
        self.figure2_pane = pn.pane.Plotly(self.figure2)

        # construct the TPC layout
        self.layout = Row(
            Column(
                self.file_folder_select,
                self.file_select,
                self.load_file_button,
                self.tpc_meta_pretext,
                self.event_select,
                self.load_event_button,
                self.link_axes_toggle,
                self.simulation_wrangler_pretext,
            ),
            Column(
                self.figure1_pane,
                Row(
                    self.figure1_adc_slider_option,
                    self.figure1_slider,
                    width_policy="fixed",
                    width=600,
                ),
                Row(
                    Column(
                        self.figure1_radio_text,
                        self.figure1_radio_group,
                        self.figure1_plot_option_text,
                        self.figure1_plot_options,
                        self.figure1_color_select,
                        self.figure1_plot_type_options,
                        self.figure1_plot_button,
                        width_policy="fixed",
                        width=300,
                    ),
                ),
                width_policy="fixed",
                width=600,
                height_policy="fixed",
                height=1000,
            ),
            Column(
                self.figure2_pane,
                Row(
                    self.figure2_adc_slider_option,
                    self.figure2_slider,
                    width_policy="fixed",
                    width=600,
                ),
                Row(
                    Column(
                        self.figure2_radio_text,
                        self.figure2_radio_group,
                        self.figure2_plot_option_text,
                        self.figure2_plot_options,
                        self.figure2_color_select,
                        self.figure2_plot_type_options,
                        self.figure2_plot_button,
                        width_policy="fixed",
                        width=300,
                    ),
                ),
                width_policy="fixed",
                width=600,
                height_policy="fixed",
                height=1000,
            ),
        )

    """
    functions here are for updating the TPC display left panel.
    """

    def update_file_folder(self, event):
        new = event.new
        if new == "..":
            self.file_folder = str(Path(self.file_folder).parent)
        elif new == ".":
            pass
        else:
            self.file_folder = (
                str(Path(self.file_folder)) + "/" + new
            )

        self.update_available_folders()
        self.file_folder_select.options = self.available_folders
        self.file_folder_select.title = (
            title
        ) = f"Blip folder: ~/{Path(self.file_folder).parts[-1]}"
        self.file_folder_select.value = "."

        self.update_available_files()
        self.file_select.options = self.available_files
        if len(self.available_files) > 0:
            self.file_select.value = self.available_files[0]

    def update_input_file(self, new):
        self.input_file = new

    def update_meta(self):
        for ii, item in enumerate(self.tpc_meta_vars):
            if item in self.tpc_meta.keys():
                self.tpc_meta_vals[ii] = self.tpc_meta[item]
        self.tpc_meta_vals[0] = self.input_file
        self.update_meta_string()
        self.tpc_meta_pretext.text = self.tpc_meta_string

    def update_events(self):
        self.update_available_events()
        self.event_select.options = self.available_events
        self.event_select.value = str(self.event)

    def update_event(self, event):
        if event.new is None:
            self.event = 0
        else:
            self.event = int(event.new)

    def update_link_axes(self, new):
        if self.link_axes_toggle.value:
            self.figure2.x_range = self.figure1.x_range
            self.figure2.y_range = self.figure1.y_range

    """
    functions here are for updating the TPC display plots.
    """

    def update_figure1_adc_slider_option(self, new):
        if self.figure1_adc_slider_option.value == True:
            self.figure1_adc_slider_option_bool = True
        else:
            self.figure1_adc_slider_option_bool = False

    def update_figure1_radio_group(self, figure):
        if self.figure1_radio_group.value == "TPC":
            self.figure1_plot_type = "TPC"
            self.figure1_plot_type_options.options = self.options
            if self.figure1_plot_options.value == "Truth":
                self.figure1_plot_option = "Truth"
                self.figure1_color_select.options = (
                    self.available_truth_labels
                )
                self.figure1_color_select.value = (
                    self.available_truth_labels[0]
                )
                self.figure1_label = self.available_truth_labels[0]
            elif self.figure1_plot_options.value == "Predictions":
                self.figure1_plot_option = "Predictions"
                self.figure1_color_select.options = (
                    self.available_prediction_labels
                )
                if len(self.available_prediction_labels) > 0:
                    self.figure1_color_select.value = (
                        self.available_prediction_labels[0]
                    )
                    self.figure1_label = (
                        self.available_prediction_labels[0]
                    )

        elif self.figure1_radio_group.value == "Merge Tree":
            self.figure1_plot_type = "Merge Tree"
            self.figure1_plot_type_options.options = self.merge_tree_options
            if self.figure1_plot_options.value == "Truth":
                self.figure1_plot_option = "Truth"
                self.figure1_color_select.options = (
                    self.available_merge_tree_truth_labels
                )
                self.figure1_color_select.value = (
                    self.available_merge_tree_truth_labels[0]
                )
                self.figure1_label = self.available_merge_tree_truth_labels[0]
            elif self.figure1_plot_options.value == "Predictions":
                self.figure1_plot_option = "Predictions"
                self.figure1_color_select.options = (
                    self.available_merge_tree_prediction_labels
                )
                if len(self.available_merge_tree_prediction_labels) > 0:
                    self.figure1_color_select.value = (
                        self.available_merge_tree_prediction_labels[0]
                    )
                    self.figure1_label = (
                        self.available_merge_tree_prediction_labels[0]
                    )

        self.figure1["layout"].update(
            template="presentation",
            title=f"Plot I [{self.figure1_plot_type} {self.figure1_plot_option}]:",
        )

    def update_figure1_color(self, new):
        self.figure1_label = self.figure1_color_select.value

    def update_figure1_plot_type_options(self, new):
        self.figure1_plot_type_option = self.figure1_plot_type_options.value

        if self.figure1_plot_type == "TPC":
            self.figure1_event_features = self.features[self.event]
            self.figure1_event_classes = self.classes[self.event]
            self.figure1_event_clusters = self.clusters[self.event]
            self.figure1_event_hits = []

        elif self.figure1_plot_type == "Merge Tree":
            self.figure1_event_features = []
            self.figure1_event_classes = []
            self.figure1_event_clusters = []
            self.figure1_event_hits = []

    def update_figure2_adc_slider_option(self, new):
        if self.figure2_adc_slider_option.value == True:
            self.figure2_adc_slider_option_bool = True
        else:
            self.figure2_adc_slider_option_bool = False

    def update_figure2_radio_group(self, figure):
        if self.figure2_radio_group.value == "TPC":
            self.figure2_plot_type = "TPC"
            self.figure2_plot_type_options.options = self.options
            if self.figure2_plot_options.value == "Truth":
                self.figure2_plot_option = "Truth"
                self.figure2_color_select.options = (
                    self.available_truth_labels
                )
                self.figure2_color_select.value = (
                    self.available_truth_labels[0]
                )
                self.figure2_label = self.available_truth_labels[0]
            elif self.figure2_plot_options.value == "Predictions":
                self.figure2_plot_option = "Predictions"
                self.figure2_color_select.options = (
                    self.available_prediction_labels
                )
                if len(self.available_prediction_labels) > 0:
                    self.figure2_color_select.value = (
                        self.available_prediction_labels[0]
                    )
                    self.figure2_label = (
                        self.available_prediction_labels[0]
                    )

        if self.figure2_radio_group.value == "Merge Tree":
            self.figure2_plot_type = "Merge Tree"
            self.figure2_plot_type_options.options = self.merge_tree_options
            if self.figure2_plot_options.value == "Truth":
                self.figure2_plot_option = "Truth"
                self.figure2_color_select.options = (
                    self.available_merge_tree_truth_labels
                )
                self.figure2_color_select.value = (
                    self.available_merge_tree_truth_labels[0]
                )
                self.figure2_label = self.available_merge_tree_truth_labels[0]
            elif self.figure2_plot_options.value == "Predictions":
                self.figure2_plot_option = "Predictions"
                self.figure2_color_select.options = (
                    self.available_merge_tree_prediction_labels
                )
                if len(self.available_merge_tree_prediction_labels) > 0:
                    self.figure2_color_select.value = (
                        self.available_merge_tree_prediction_labels[0]
                    )
                    self.figure2_label = (
                        self.available_merge_tree_prediction_labels[0]
                    )

        self.figure1["layout"].update(
            template="presentation",
            title=f"Plot II [{self.figure2_plot_type} {self.figure2_plot_option}]:",
        )

    def update_figure2_color(self, new):
        self.figure2_label = self.figure2_color_select.value

    def update_figure2_plot_type_options(self, new):
        self.figure2_plot_type_option = self.figure2_plot_type_options.value

        if self.figure2_plot_type == "TPC":
            self.figure2_event_features = self.features[self.event]
            self.figure2_event_classes = self.classes[self.event]
            self.figure2_event_clusters = self.clusters[self.event]
            self.figure2_event_hits = []

        elif self.figure2_plot_type == "Merge Tree":
            self.figure2_event_features = []
            self.figure2_event_classes = []
            self.figure2_event_clusters = []
            self.figure2_event_hits = []

    def load_input_file(self, event):
        """
        Load a blip file into the viewer.
        """
        self.input_file = self.file_select.value
        if self.input_file.endswith(".npz"):
            self.load_npz_file()
        elif self.input_file.endswith(".root"):
            self.load_root_file()
        else:
            print(f"Can't load file {self.input_file}.")

    def load_npz_file(self):
        """
        Load a blip npz file into the viewer.
        """
        input_file = np.load(
            self.file_folder + "/" + self.input_file,
            allow_pickle=True,
        )
        self.available_prediction_labels = []
        print(input_file.files)
        if "meta" in input_file.files:
            self.tpc_meta = input_file["meta"].item()
            self.update_meta()
            self.update_events()
        if "features" in input_file.files:
            self.features = input_file["features"]
        if "classes" in input_file.files:
            self.classes = input_file["classes"]
        if "clusters" in input_file.files:
            self.clusters = input_file["clusters"]
        if "view_0_features" in input_file.files:
            self.view_0_features = input_file["view_0_features"]
        if "view_0_classes" in input_file.files:
            self.view_0_classes = input_file["view_0_classes"]
        if "view_0_clusters" in input_file.files:
            self.view_0_clusters = input_file["view_0_clusters"]
        if "view_0_hits" in input_file.files:
            self.view_0_hits = input_file["view_0_hits"]
        if "view_1_features" in input_file.files:
            self.view_1_features = input_file["view_1_features"]
        if "view_1_classes" in input_file.files:
            self.view_1_classes = input_file["view_1_classes"]
        if "view_1_clusters" in input_file.files:
            self.view_1_clusters = input_file["view_1_clusters"]
        if "view_1_hits" in input_file.files:
            self.view_1_hits = input_file["view_1_hits"]
        if "view_2_features" in input_file.files:
            self.view_2_features = input_file["view_2_features"]
        if "view_2_classes" in input_file.files:
            self.view_2_classes = input_file["view_2_classes"]
        if "view_2_clusters" in input_file.files:
            self.view_2_clusters = input_file["view_2_clusters"]
        if "view_2_hits" in input_file.files:
            self.view_2_hits = input_file["view_2_hits"]
        if "merge_features" in input_file.files:
            self.merge_features = input_file["merge_features"]
        if "merge_classes" in input_file.files:
            self.merge_classes = input_file["merge_classes"]
        if "mc_maps" in self.tpc_meta.keys():
            self.mc_maps = self.tpc_meta["mc_maps"]

        if "source" in input_file.files:
            self.predictions["source"] = input_file["source"]
            self.available_prediction_labels.append("source")
        if "topology" in input_file.files:
            self.predictions["topology"] = input_file["topology"]
            self.available_prediction_labels.append("topology")
        if "particle" in input_file.files:
            self.predictions["particle"] = input_file["particle"]
            self.available_prediction_labels.append("particle")
        if "physics" in input_file.files:
            self.predictions["physics"] = input_file["physics"]
            self.available_prediction_labels.append("physics")
        if self.figure1_plot_type == "Predictions":
            self.figure1_color_select.options = (
                self.available_prediction_labels
            )
            if len(self.available_prediction_labels) > 0:
                self.figure1_color_select.value = (
                    self.available_prediction_labels[0]
                )
                self.figure1_label = self.available_prediction_labels[0]
        if self.figure2_plot_type == "Predictions":
            self.figure2_color_select.options = (
                self.available_prediction_labels
            )
            if len(self.available_prediction_labels) > 0:
                self.figure2_color_select.value = (
                    self.available_prediction_labels[0]
                )
                self.figure2_label = self.available_prediction_labels[
                    0
                ]

    def load_root_file(self):
        pass

    def load_event(self, event):
        if str(self.event) in self.available_events:
            self.figure1_event_features = self.view_0_features[self.event]
            self.figure1_event_classes = self.view_0_classes[self.event]
            self.figure1_event_clusters = self.view_0_clusters[self.event]
            self.figure1_event_hits = self.view_0_hits[self.event]
            self.figure2_event_features = self.view_0_features[self.event]
            self.figure2_event_classes = self.view_0_classes[self.event]
            self.figure2_event_clusters = self.view_0_clusters[self.event]
            self.figure2_event_hits = self.view_0_hits[self.event]
            self.event_predictions = {
                key: val[self.event][0] for key, val in self.predictions.items()
            }
            if "mc_maps" in self.tpc_meta.keys():
                self.event_pdg_maps = self.mc_maps["pdg_code"][self.event]
                self.event_parent_track_id_maps = self.mc_maps["parent_track_id"][
                    self.event
                ]
                self.event_ancestor_track_id_maps = self.mc_maps["ancestor_track_id"][
                    self.event
                ]
                self.event_ancestor_level_maps = self.mc_maps["ancestor_level"][
                    self.event
                ]
        else:
            pass

    def plot_first_event(self, event):
        if self.figure1_label == "adc":
            print("Plotting adc (in progress)")
            pass
        if self.figure1_label == "MergeTree":
            print("Plotting MergeTree computed")
            # Load the saved data
            try:
                event = torch.load(
                    osp.join(self.processed_dir, "data_0.pt")
                )  # Replace 'data_0.pt' with your filename
                print("MergeTree loaded")
            except FileNotFoundError:
                print("MergeTree not found")

            merge_tree_data = event.merge_tree
            G = nx.Graph(merge_tree_data)
            pos = nx.spring_layout(G)
            edge_trace = go.Scatter(
                x=[],
                y=[],
                line=dict(width=0.5, color="#888"),
                hoverinfo="none",
                mode="lines",
            )

            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace["x"] += tuple([x0, x1, None])
                edge_trace["y"] += tuple([y0, y1, None])

            # Create a trace for the nodes
            node_trace = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode="markers",
                hoverinfo="text",
                marker=dict(
                    showscale=True,
                    colorscale="YlGnBu",
                    reversescale=True,
                    color=[],
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        title="Node Connections",
                        xanchor="left",
                        titleside="right",
                    ),
                    line=dict(width=2),
                ),
            )

            for node in G.nodes():
                x, y = pos[node]
                node_trace["x"] += tuple([x])
                node_trace["y"] += tuple([y])

            # Create a Figure and add the traces
            plotly_figure = go.Figure(data=[edge_trace, node_trace])
            title = f"Plot I [Merge Tree]:"
            xaxis = dict(title="Nodes")
            yaxis = dict(title="Height")

        else:
            label_index = self.tpc_meta["classes"][self.figure1_label]
            print(label_index)
            label_vals = self.tpc_meta[f"{self.figure1_label}_labels"]
            print(label_vals)

            self.first_scatter_random_colors = [ii for ii in range(256)]
            random.shuffle(self.first_scatter_random_colors)
            self.first_scatter_colors = {
                val: Magma256[int(self.first_scatter_random_colors[ii] % 256)]
                for ii, val in enumerate(label_vals.values())
            }
            for key, val in label_vals.items():
                if self.figure1_plot_option == "Truth":
                    mask = self.figure1_event_classes[:, label_index] == key
                else:
                    if (
                        self.figure1_label
                        not in self.available_prediction_labels
                    ):
                        continue
                    labels = np.argmax(
                        self.event_predictions[self.figure1_label], axis=1
                    )
                    mask = labels == key
                if np.sum(mask) == 0:
                    continue
                if self.figure1_adc_slider_option_bool is True:
                    self.first_scatter[str(val)] = go.Scatter3d(
                        x=self.figure1_event_features[:, 0][mask],
                        y=self.figure1_event_features[:, 1][mask],
                        z=self.figure1_event_features[:, 2][mask],
                        mode="markers",
                        marker=dict(
                            size=abs(self.figure1_event_features[:, 2][mask])
                            * 1000,
                            color=self.first_scatter_colors[val],
                        ),
                        name=str(val),
                    )
                    self.figure1_slider = pn.widgets.FloatSlider()

                    def update_figure1_marker_size(event):
                        for trace in self.figure1.object.data:
                            trace.marker.size = event.new * 1000
                        self.figure1.param.trigger("object")

                    self.figure1_slider.param.watch(
                        update_figure1_marker_size, "value"
                    )
                else:
                    self.first_scatter[str(val)] = go.Scatter3d(
                        x=self.figure1_event_features[:, 0][mask],
                        y=self.figure1_event_features[:, 1][mask],
                        z=self.figure1_event_features[:, 2][mask],
                        mode="markers",
                        marker=dict(size=10, color=self.first_scatter_colors[val]),
                        name=str(val),
                    )

            plotly_figure = go.Figure(data=list(self.first_scatter.values()))
            title = f"Plot I [{self.figure1_plot_type} {self.figure1_plot_option}]:"
            xaxis = dict(title="Channel [n]")
            yaxis = dict(title="TDC [10ns]")

        self.figure1 = plotly_figure.to_dict()
        self.figure1["layout"].update(
            template="presentation",
            title=title,
            xaxis=xaxis,
            yaxis=yaxis,
            showlegend=True,
        )
        self.figure1_pane.object = self.figure1

    def plot_second_event(self, event):
        if self.figure2_label == "adc":
            pass
        if self.figure2_label == "MergeTree":
            print("Plotting MergeTree computed")
            # Load the saved data
            try:
                event = torch.load(
                    osp.join(self.processed_dir, "data_0.pt")
                )  # Replace 'data_0.pt' with your filename
                print("MergeTree loaded")
            except FileNotFoundError:
                print("MergeTree not found")

            merge_tree_data = event.merge_tree
            G = nx.Graph(merge_tree_data)
            pos = nx.spring_layout(G)
            edge_trace = go.Scatter(
                x=[],
                y=[],
                line=dict(width=0.5, color="#888"),
                hoverinfo="none",
                mode="lines",
            )

            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace["x"] += tuple([x0, x1, None])
                edge_trace["y"] += tuple([y0, y1, None])

            # Create a trace for the nodes
            node_trace = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode="markers",
                hoverinfo="text",
                marker=dict(
                    showscale=True,
                    colorscale="YlGnBu",
                    reversescale=True,
                    color=[],
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        title="Node Connections",
                        xanchor="left",
                        titleside="right",
                    ),
                    line=dict(width=2),
                ),
            )

            for node in G.nodes():
                x, y = pos[node]
                node_trace["x"] += tuple([x])
                node_trace["y"] += tuple([y])

            # Create a Figure and add the traces
            plotly_figure = go.Figure(data=[edge_trace, node_trace])
            title = f"Plot I [Merge Tree]:"
            xaxis = dict(title="Nodes")
            yaxis = dict(title="Height")

        else:
            label_index = self.tpc_meta["classes"][self.figure2_label]
            label_vals = self.tpc_meta[f"{self.figure2_label}_labels"]
            self.second_scatter_random_colors = [ii for ii in range(256)]
            random.shuffle(self.second_scatter_random_colors)
            self.second_scatter_colors = {
                val: Magma256[int(self.second_scatter_random_colors[ii] % 256)]
                for ii, val in enumerate(label_vals.values())
            }
            for key, val in label_vals.items():
                if self.figure2_plot_option == "Truth":
                    mask = self.figure2_event_classes[:, label_index] == key
                else:
                    if (
                        self.figure2_label
                        not in self.available_prediction_labels
                    ):
                        continue
                    labels = np.argmax(
                        self.event_predictions[self.figure2_label], axis=1
                    )
                    mask = labels == key
                if np.sum(mask) == 0:
                    continue
                if self.figure2_adc_slider_option_bool == True:
                    self.second_scatter[str(val)] = go.Scatter3d(
                        x=self.figure2_event_features[:, 0][mask],
                        y=self.figure2_event_features[:, 1][mask],
                        z=self.figure2_event_features[:, 2][mask],
                        mode="markers",
                        marker=dict(
                            size=abs(self.figure2_event_features[:, 2][mask]),
                            color=self.second_scatter_colors[val],
                        ),
                        name=str(val),
                    )
                    self.figure2_slider = pn.widgets.FloatSlider()

                    def update_figure2_marker_size(event):
                        for trace in self.figure2.object.data:
                            trace.marker.size = event.new
                        self.figure2.param.trigger("object")

                    self.figure2_slider.param.watch(
                        update_figure2_marker_size, "value"
                    )

                else:
                    self.second_scatter[str(val)] = go.Scatter3d(
                        x=self.figure2_event_features[:, 0][mask],
                        y=self.figure2_event_features[:, 1][mask],
                        z=self.figure2_event_features[:, 2][mask],
                        mode="markers",
                        marker=dict(size=10, color=self.second_scatter_colors[val]),
                        name=str(val),
                    )

            plotly_figure = go.Figure(data=list(self.second_scatter.values()))
            title = f"Plot II [{self.figure2_plot_type} {self.figure2_plot_option}]:"
            xaxis = dict(title="Channel [n]")
            yaxis = dict(title="TDC [10ns]")

        self.figure2 = plotly_figure.to_dict()
        self.figure2["layout"].update(
            template="presentation",
            title=title,
            xaxis=xaxis,
            yaxis=yaxis,
            showlegend=True,
        )
        self.figure2_pane.object = self.figure2
