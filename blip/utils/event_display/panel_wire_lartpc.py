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


class WireLArTPCPanelDisplay:
    """
    Tools for displaying events
    """

    def __init__(self, document=None):
        # File folder and file select.
        # These act as drop down menus which update upon
        # selecting, one for wire planes and another for
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
            "edep_id",
            "edep_process",
            "detsim",
            "random_detsim",
            "edep_detsim",
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
        self.edep_features = []
        self.view_features = []
        self.view_0_features = {0: 0}
        self.view_1_features = {0: 0}
        self.view_2_features = {0: 0}
        self.merge_features = {0: 0}
        self.view_0_classes = {0: 0}
        self.view_1_classes = {0: 0}
        self.view_2_classes = {0: 0}
        self.merge_classes = {0: 0}
        self.view_0_clusters = {0: 0}
        self.view_1_clusters = {0: 0}
        self.view_2_clusters = {0: 0}
        self.view_0_hits = {0: 0}
        self.view_1_hits = {0: 0}
        self.view_2_hits = {0: 0}
        self.classes = []
        self.clusters = []
        self.hits = []
        self.edeps = []
        self.predictions = {}

        self.plot_types = ["Wire Plane", "Wire Channel", "TPC", "Merge Tree"]
        self.plot_options = ["Truth", "Predictions"]
        self.options = ["View 0", "View 1", "View 2"]
        self.wire_channel_options = []
        self.tpc_options = []
        self.merge_tree_options = ["View 0", "View 1", "View 2"]

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
        self.available_edep_truth_labels = [
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
        self.available_edep_prediction_labels = ["None"]
        self.available_merge_tree_truth_labels = ["cluster_particle", "MergeTree"]
        self.available_merge_tree_prediction_labels = ["None"]

        self.first_figure_label = "adc"
        self.first_scatter = {}
        self.first_figure_plot_type = "Wire Plane"

        # parameters for second plot
        self.available_prediction_labels = []
        self.second_figure_label = ""
        self.second_scatter = {}
        self.second_figure_plot_type = "Wire Plane"

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
        self.simulation_wrangler_string = "Particle Information:\n"
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

    def update_first_figure_taptool(self, event):
        print(event.x, event.y)

    def update_second_figure_taptool(self):
        pass

    def construct_widgets(self, document):
        # Left hand column
        self.construct_left_hand_column()
        self.construct_first_figure_column()
        self.construct_second_figure_column()
        self.construct_layout()

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
        # Plot text information
        self.simulation_wrangler_pretext = PreText(
            text=self.simulation_wrangler_string, width=200, height=200
        )

        # First plot column
    def construct_first_figure_column(self):
        self.first_figure_event_features = []
        self.first_figure_event_classes = []
        self.first_figure_event_clusters = []
        self.first_figure_event_hits = []
        self.first_figure = go.Figure(
            layout=dict(
                template="presentation",
                title="Plot I [Wire Plane Truth]",
                xaxis=dict(title="x []"),
                yaxis=dict(title="y []"),
                coloraxis=dict(colorbar=dict(title=""), colorscale="Viridis"),
                legend=dict(
                    x=1,
                    y=1,
                    xanchor='left',
                    yanchor='top'
                )
            )
        )
        self.first_figure = self.first_figure.add_trace(go.Scatter())
        self.first_figure.data[0].on_click(self.update_first_figure_taptool)
        self.first_figure_adc_slider_option = Checkbox(
            name="Use ADC Slider", value=False
        )
        self.first_figure_adc_slider_option.param.watch(
            self.update_first_figure_adc_slider_option, "value"
        )
        self.first_figure_adc_slider_option_bool = False
        self.first_figure_slider = Slider(start=0.1, end=1, step=0.1, value=0.1)
        # Plot type radio group
        self.first_figure_plot_type = "Wire Plane"
        self.first_figure_radio_text = PreText(text="Plot I type:")
        self.first_figure_radio_group = RadioButtonGroup(options=self.plot_types)
        self.first_figure_radio_group.param.watch(
            self.update_first_figure_radio_group, "value"
        )
        # Plot type labeling options
        self.first_figure_color_select = Select(
            name="Plot I labeling:",
            value="",
            options=self.available_truth_labels,
            width_policy="fixed",
            width=150,
        )
        self.first_figure_color_select.param.watch(
            self.update_first_figure_color, "value"
        )
        # Plot options (truth/predictions)
        self.first_figure_plot_option = "Truth"
        self.first_figure_plot_option_text = PreText(text="Truth/Predictions:")
        self.first_figure_plot_options = RadioButtonGroup(
            options=self.plot_options, value=0
        )
        self.first_figure_plot_options.param.watch(
            self.update_first_figure_radio_group, "value"
        )
        # Plot type options
        self.first_figure_plot_type_options = Select(
            name="Plot I options:",
            value="",
            options=self.options,
            width_policy="fixed",
            width=150,
        )
        self.first_figure_plot_type_options.param.watch(
            self.update_first_figure_plot_type_options, "value"
        )
        # Plot button
        self.first_figure_plot_button = Button(
            name="Plot event", button_type="primary", width_policy="fixed", width=100
        )
        self.first_figure_plot_button.on_click(self.plot_first_event)
        self.first_figure_pane = pn.pane.Plotly(self.first_figure)

        # Second plot column
    def construct_second_figure_column(self):
        self.second_figure_event_features = []
        self.second_figure_event_classes = []
        self.second_figure_event_clusters = []
        self.second_figure_event_hits = []
        self.second_figure = go.Figure(
            layout=dict(
                template="presentation",
                title="Plot II [Predictions]",
                xaxis=dict(title="x []"),
                yaxis=dict(title="y []"),
                coloraxis=dict(colorbar=dict(title=""), colorscale="Viridis"),
            )
        )
        # Defining properties of color mapper
        # self.second_figure_color_mapper = LinearColorMapper(palette = "Viridis256")
        # self.second_figure_color_bar    = ColorBar(
        #     color_mapper   = self.second_figure_color_mapper,
        #     label_standoff = 12,
        #     location       = (0,0),
        #     title          = ''
        # )
        self.second_figure = self.second_figure.add_trace(go.Scatter())
        self.second_figure.data[0].on_click(self.update_second_figure_taptool)

        self.second_figure_adc_slider_option = Checkbox(
            name="Use ADC Slider", value=False
        )
        self.second_figure_adc_slider_option.param.watch(
            self.update_second_figure_adc_slider_option, "value"
        )
        self.second_figure_adc_slider_option_bool = False
        self.second_figure_slider = Slider(start=0.1, end=1, step=0.1, value=0.1)
        # self.second_figure.legend.click_policy="hide"
        # Plot II type
        self.second_figure_plot_type = "Wire Plane"
        self.second_figure_radio_text = PreText(text="Plot II type:")
        self.second_figure_radio_group = RadioButtonGroup(
            options=self.plot_types, value=1
        )
        self.second_figure_radio_group.param.watch(
            self.update_second_figure_radio_group, "value"
        )
        # Plot II labeling
        self.second_figure_color_select = Select(
            name="Plot II labeling:",
            value="",
            options=self.available_prediction_labels,
            width_policy="fixed",
            width=150,
        )
        self.second_figure_color_select.param.watch(
            self.update_second_figure_color, "value"
        )
        # Plot options (truth/predictions)
        self.second_figure_plot_option = "Truth"
        self.second_figure_plot_option_text = PreText(text="Truth/Predictions:")
        self.second_figure_plot_options = RadioButtonGroup(
            options=self.plot_options, value=0
        )
        self.second_figure_plot_options.param.watch(
            self.update_second_figure_radio_group, "value"
        )
        # Plot type options
        self.second_figure_plot_type_options = Select(
            name="Plot I options:",
            value="",
            options=self.options,
            width_policy="fixed",
            width=150,
        )
        self.second_figure_plot_type_options.param.watch(
            self.update_second_figure_plot_type_options, "value"
        )
        self.second_figure_plot_button = Button(
            name="Plot event", button_type="primary", width_policy="fixed", width=100
        )
        self.second_figure_plot_button = Button(
            name="Plot event", button_type="primary", width_policy="fixed", width=100
        )
        self.second_figure_plot_button.on_click(self.plot_second_event)
        self.second_figure_pane = pn.pane.Plotly(self.second_figure)

    def construct_layout(self):
        # construct the wire plane layout
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
                self.first_figure_pane,
                Row(
                    self.first_figure_adc_slider_option,
                    self.first_figure_slider,
                    width_policy="fixed",
                    width=600,
                ),
                Row(
                    Column(
                        self.first_figure_radio_text,
                        self.first_figure_radio_group,
                        self.first_figure_plot_option_text,
                        self.first_figure_plot_options,
                        self.first_figure_color_select,
                        self.first_figure_plot_type_options,
                        self.first_figure_plot_button,
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
                self.second_figure_pane,
                Row(
                    self.second_figure_adc_slider_option,
                    self.second_figure_slider,
                    width_policy="fixed",
                    width=600,
                ),
                Row(
                    Column(
                        self.second_figure_radio_text,
                        self.second_figure_radio_group,
                        self.second_figure_plot_option_text,
                        self.second_figure_plot_options,
                        self.second_figure_color_select,
                        self.second_figure_plot_type_options,
                        self.second_figure_plot_button,
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
    functions here are for updating the Wire Plane display left panel.
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
        if event.new == None:
            self.event = 0
        else:
            self.event = int(event.new)

    def update_link_axes(self, new):
        if self.link_axes_toggle.value:
            self.second_figure.x_range = self.first_figure.x_range
            self.second_figure.y_range = self.first_figure.y_range

    """
    functions here are for updating the Wire Plane display plots.
    """

    def update_first_figure_adc_slider_option(self, new):
        if self.first_figure_adc_slider_option.value == True:
            self.first_figure_adc_slider_option_bool = True
        else:
            self.first_figure_adc_slider_option_bool = False

    def update_first_figure_radio_group(self, figure):
        if self.first_figure_radio_group.value == "Wire Plane":
            self.first_figure_plot_type = "Wire Plane"
            self.first_figure_plot_type_options.options = self.options
            if self.first_figure_plot_options.value == "Truth":
                self.first_figure_plot_option = "Truth"
                self.first_figure_color_select.options = (
                    self.available_truth_labels
                )
                self.first_figure_color_select.value = (
                    self.available_truth_labels[0]
                )
                self.first_figure_label = self.available_truth_labels[0]
            elif self.first_figure_plot_options.value == "Predictions":
                self.first_figure_plot_option = "Predictions"
                self.first_figure_color_select.options = (
                    self.available_prediction_labels
                )
                if len(self.available_prediction_labels) > 0:
                    self.first_figure_color_select.value = (
                        self.available_prediction_labels[0]
                    )
                    self.first_figure_label = (
                        self.available_prediction_labels[0]
                    )

        elif self.first_figure_radio_group.value == "Wire Channel":
            self.first_figure_plot_type = "Wire Channel"
            self.first_figure_plot_type_options.options = self.wire_channel_options
            if self.first_figure_plot_options.value == "Truth":
                self.first_figure_plot_option = "Truth"
                self.first_figure_color_select.options = (
                    self.available_wire_channel_truth_labels
                )
                self.first_figure_color_select.value = (
                    self.available_wire_channel_truth_labels[0]
                )
                self.first_figure_label = self.available_wire_channel_truth_labels[0]
            elif self.first_figure_plot_options.value == "Predictions":
                self.first_figure_plot_option = "Predictions"
                self.first_figure_color_select.options = (
                    self.available_wire_channel_prediction_labels
                )
                if len(self.available_wire_channel_prediction_labels) > 0:
                    self.first_figure_color_select.value = (
                        self.available_wire_channel_prediction_labels[0]
                    )
                    self.first_figure_label = (
                        self.available_wire_channel_prediction_labels[0]
                    )

        elif self.first_figure_radio_group.value == "TPC":
            self.first_figure_plot_type = "TPC"
            self.first_figure_plot_type_options.options = self.tpc_options
            if self.first_figure_plot_options.value == "Truth":
                self.first_figure_plot_option = "Truth"
                self.first_figure_color_select.options = (
                    self.available_edep_truth_labels
                )
                self.first_figure_color_select.value = self.available_edep_truth_labels[
                    0
                ]
                self.first_figure_label = self.available_edep_truth_labels[0]
            elif self.first_figure_plot_options.value == "Predictions":
                self.first_figure_plot_option = "Predictions"
                self.first_figure_color_select.options = (
                    self.available_edep_prediction_labels
                )
                if len(self.available_edep_prediction_labels) > 0:
                    self.first_figure_color_select.value = (
                        self.available_edep_prediction_labels[0]
                    )
                    self.first_figure_label = self.available_edep_prediction_labels[0]

        elif self.first_figure_radio_group.value == "Merge Tree":
            self.first_figure_plot_type = "Merge Tree"
            self.first_figure_plot_type_options.options = self.merge_tree_options
            if self.first_figure_plot_options.value == "Truth":
                self.first_figure_plot_option = "Truth"
                self.first_figure_color_select.options = (
                    self.available_merge_tree_truth_labels
                )
                self.first_figure_color_select.value = (
                    self.available_merge_tree_truth_labels[0]
                )
                self.first_figure_label = self.available_merge_tree_truth_labels[0]
            elif self.first_figure_plot_options.value == "Predictions":
                self.first_figure_plot_option = "Predictions"
                self.first_figure_color_select.options = (
                    self.available_merge_tree_prediction_labels
                )
                if len(self.available_merge_tree_prediction_labels) > 0:
                    self.first_figure_color_select.value = (
                        self.available_merge_tree_prediction_labels[0]
                    )
                    self.first_figure_label = (
                        self.available_merge_tree_prediction_labels[0]
                    )

        self.first_figure["layout"].update(
            template="presentation",
            title=f"Plot I [{self.first_figure_plot_type} {self.first_figure_plot_option}]:",
        )

    def update_first_figure_color(self, new):
        self.first_figure_label = self.first_figure_color_select.value

    def update_first_figure_plot_type_options(self, new):
        self.first_figure_plot_type_option = self.first_figure_plot_type_options.value

        if (self.first_figure_plot_type == "Wire Plane") or (
            self.first_figure_plot_type == "Merge Tree"
            and self.first_figure_label == "cluster_particle"
        ):
            self.first_figure_plot_type == "Wire Plane"
            if self.first_figure_plot_type_option == "View 0":
                self.first_figure_event_features = self.view_0_features[self.event]
                self.first_figure_event_classes = self.view_0_classes[self.event]
                self.first_figure_event_clusters = self.view_0_clusters[self.event]
                self.first_figure_event_hits = self.view_0_hits[self.event]
            elif self.first_figure_plot_type_option == "View 1":
                self.first_figure_event_features = self.view_1_features[self.event]
                self.first_figure_event_classes = self.view_1_classes[self.event]
                self.first_figure_event_clusters = self.view_1_clusters[self.event]
                self.first_figure_event_hits = self.view_1_hits[self.event]
            elif self.first_figure_plot_type_option == "View 2":
                self.first_figure_event_features = self.view_2_features[self.event]
                self.first_figure_event_classes = self.view_2_classes[self.event]
                self.first_figure_event_clusters = self.view_2_clusters[self.event]
                self.first_figure_event_hits = self.view_2_hits[self.event]

        elif self.first_figure_plot_type == "TPC":
            self.first_figure_event_features = self.edep_features[self.event]
            self.first_figure_event_classes = self.edep_classes[self.event]
            self.first_figure_event_clusters = self.edep_clusters[self.event]
            self.first_figure_event_hits = []

        elif self.first_figure_plot_type == "Merge Tree":
            self.first_figure_event_features = []
            self.first_figure_event_classes = []
            self.first_figure_event_clusters = []
            self.first_figure_event_hits = []

    def update_second_figure_adc_slider_option(self, new):
        if self.second_figure_adc_slider_option.value == True:
            self.second_figure_adc_slider_option_bool = True
        else:
            self.second_figure_adc_slider_option_bool = False

    def update_second_figure_radio_group(self, figure):
        if self.second_figure_radio_group.value == "Wire Plane":
            self.second_figure_plot_type = "Wire Plane"
            self.second_figure_plot_type_options.options = self.options
            if self.second_figure_plot_options.value == "Truth":
                self.second_figure_plot_option = "Truth"
                self.second_figure_color_select.options = (
                    self.available_truth_labels
                )
                self.second_figure_color_select.value = (
                    self.available_truth_labels[0]
                )
                self.second_figure_label = self.available_truth_labels[0]
            elif self.second_figure_plot_options.value == "Predictions":
                self.second_figure_plot_option = "Predictions"
                self.second_figure_color_select.options = (
                    self.available_prediction_labels
                )
                if len(self.available_prediction_labels) > 0:
                    self.second_figure_color_select.value = (
                        self.available_prediction_labels[0]
                    )
                    self.second_figure_label = (
                        self.available_prediction_labels[0]
                    )
        if self.second_figure_radio_group.value == "Wire Channel":
            self.second_figure_plot_type = "Wire Channel"
            self.second_figure_plot_type_options.options = self.wire_channel_options
            if self.second_figure_plot_options.value == "Truth":
                self.second_figure_plot_option = "Truth"
                self.second_figure_color_select.options = (
                    self.available_wire_channel_truth_labels
                )
                self.second_figure_color_select.value = (
                    self.available_wire_channel_truth_labels[0]
                )
                self.second_figure_label = self.available_wire_channel_truth_labels[0]
            elif self.second_figure_plot_options.value == "Predictions":
                self.second_figure_plot_option = "Predictions"
                self.second_figure_color_select.options = (
                    self.available_wire_channel_prediction_labels
                )
                if len(self.available_wire_channel_prediction_labels) > 0:
                    self.second_figure_color_select.value = (
                        self.available_wire_channel_prediction_labels[0]
                    )
                    self.second_figure_label = (
                        self.available_wire_channel_prediction_labels[0]
                    )

        if self.second_figure_radio_group.value == "TPC":
            self.second_figure_plot_type = "TPC"
            self.second_figure_plot_type_options.options = self.tpc_options
            if self.second_figure_plot_options.value == "Truth":
                self.second_figure_plot_option = "Truth"
                self.second_figure_color_select.options = (
                    self.available_edep_truth_labels
                )
                self.second_figure_color_select.value = (
                    self.available_edep_truth_labels[0]
                )
                self.second_figure_label = self.available_edep_truth_labels[0]
            elif self.second_figure_plot_options.value == "Predictions":
                self.second_figure_plot_option = "Predictions"
                self.second_figure_color_select.options = (
                    self.available_edep_prediction_labels
                )
                if len(self.available_edep_prediction_labels) > 0:
                    self.second_figure_color_select.value = (
                        self.available_edep_prediction_labels[0]
                    )
                    self.second_figure_label = self.available_edep_prediction_labels[0]

        if self.second_figure_radio_group.value == "Merge Tree":
            self.second_figure_plot_type = "Merge Tree"
            self.second_figure_plot_type_options.options = self.merge_tree_options
            if self.second_figure_plot_options.value == "Truth":
                self.second_figure_plot_option = "Truth"
                self.second_figure_color_select.options = (
                    self.available_merge_tree_truth_labels
                )
                self.second_figure_color_select.value = (
                    self.available_merge_tree_truth_labels[0]
                )
                self.second_figure_label = self.available_merge_tree_truth_labels[0]
            elif self.second_figure_plot_options.value == "Predictions":
                self.second_figure_plot_option = "Predictions"
                self.second_figure_color_select.options = (
                    self.available_merge_tree_prediction_labels
                )
                if len(self.available_merge_tree_prediction_labels) > 0:
                    self.second_figure_color_select.value = (
                        self.available_merge_tree_prediction_labels[0]
                    )
                    self.second_figure_label = (
                        self.available_merge_tree_prediction_labels[0]
                    )

        self.first_figure["layout"].update(
            template="presentation",
            title=f"Plot II [{self.second_figure_plot_type} {self.second_figure_plot_option}]:",
        )

    def update_second_figure_color(self, new):
        self.second_figure_label = self.second_figure_color_select.value

    def update_second_figure_plot_type_options(self, new):
        self.second_figure_plot_type_option = self.second_figure_plot_type_options.value

        if (self.second_figure_plot_type == "Wire Plane") or (
            self.second_figure_plot_type == "Merge Tree"
            and self.second_figure_label == "cluster_particle"
        ):
            self.second_figure_plot_type == "Wire Plane"
            if self.second_figure_plot_type_option == "View 0":
                self.second_figure_event_features = self.view_0_features[self.event]
                self.second_figure_event_classes = self.view_0_classes[self.event]
                self.second_figure_event_clusters = self.view_0_clusters[self.event]
                self.second_figure_event_hits = self.view_0_hits[self.event]
            elif self.second_figure_plot_type_option == "View 1":
                self.second_figure_event_features = self.view_1_features[self.event]
                self.second_figure_event_classes = self.view_1_classes[self.event]
                self.second_figure_event_clusters = self.view_1_clusters[self.event]
                self.second_figure_event_hits = self.view_1_hits[self.event]
            elif self.second_figure_plot_type_option == "View 2":
                self.second_figure_event_features = self.view_2_features[self.event]
                self.second_figure_event_classes = self.view_2_classes[self.event]
                self.second_figure_event_clusters = self.view_2_clusters[self.event]
                self.second_figure_event_hits = self.view_2_hits[self.event]

        elif self.second_figure_plot_type == "TPC":
            self.second_figure_event_features = self.edep_features[self.event]
            self.second_figure_event_classes = self.edep_classes[self.event]
            self.second_figure_event_clusters = self.edep_clusters[self.event]
            self.second_figure_event_hits = []

        elif self.second_figure_plot_type == "Merge Tree":
            self.second_figure_event_features = []
            self.second_figure_event_classes = []
            self.second_figure_event_clusters = []
            self.second_figure_event_hits = []

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
        if "edep_features" in input_file.files:
            self.edep_features = input_file["edep_features"]
        if "edep_classes" in input_file.files:
            self.edep_classes = input_file["edep_classes"]
        if "edep_clusters" in input_file.files:
            self.edep_clusters = input_file["edep_clusters"]
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
        if self.first_figure_plot_type == "Predictions":
            self.first_figure_color_select.options = (
                self.available_prediction_labels
            )
            if len(self.available_prediction_labels) > 0:
                self.first_figure_color_select.value = (
                    self.available_prediction_labels[0]
                )
                self.first_figure_label = self.available_prediction_labels[0]
        if self.second_figure_plot_type == "Predictions":
            self.second_figure_color_select.options = (
                self.available_prediction_labels
            )
            if len(self.available_prediction_labels) > 0:
                self.second_figure_color_select.value = (
                    self.available_prediction_labels[0]
                )
                self.second_figure_label = self.available_prediction_labels[
                    0
                ]

    def load_root_file(self):
        pass

    def load_event(self, event):
        if str(self.event) in self.available_events:
            self.first_figure_event_features = self.view_0_features[self.event]
            self.first_figure_event_classes = self.view_0_classes[self.event]
            self.first_figure_event_clusters = self.view_0_clusters[self.event]
            self.first_figure_event_hits = self.view_0_hits[self.event]
            self.second_figure_event_features = self.view_0_features[self.event]
            self.second_figure_event_classes = self.view_0_classes[self.event]
            self.second_figure_event_clusters = self.view_0_clusters[self.event]
            self.second_figure_event_hits = self.view_0_hits[self.event]
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
        if self.first_figure_label == "adc":
            print("Plotting adc (in progress)")
            pass
        if self.first_figure_label == "MergeTree":
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
            if "cluster" in self.first_figure_label:
                label_index = self.tpc_meta["clusters"][
                    self.first_figure_label.replace("cluster_", "")
                ]
                label_vals = np.unique(self.first_figure_event_clusters[:, label_index])
                self.first_scatter = {}
                self.first_scatter_random_colors = [ii for ii in range(256)]
                random.shuffle(self.first_scatter_random_colors)
                self.first_scatter_colors = {
                    val: Magma256[int(self.first_scatter_random_colors[ii] % 256)]
                    for ii, val in enumerate(label_vals)
                }
                for val in label_vals:
                    if self.first_figure_plot_option == "Truth":
                        print("Plotting truth values")
                        mask = self.first_figure_event_clusters[:, label_index] == val
                    else:
                        print("Plotting predictions")
                        if (
                            self.first_figure_label
                            not in self.available_prediction_labels
                        ):
                            continue
                        labels = np.argmax(
                            self.event_predictions[self.first_figure_label], axis=1
                        )
                        mask = labels == val
                    if np.sum(mask) == 0:
                        continue
                    if self.first_figure_adc_slider_option_bool == True:
                        print("Plotting with ADC slider")
                        self.first_scatter[str(val)] = go.Scatter(
                            x=self.first_figure_event_features[:, 0][mask],
                            y=self.first_figure_event_features[:, 1][mask],
                            mode="markers",
                            marker=dict(
                                size=self.first_figure_event_features[:, 2][mask],
                                color=self.first_scatter_colors[val],
                            ),
                            name=str(val),
                        )
                        self.first_figure_slider = pn.widgets.FloatSlider()

                        def update_first_figure_marker_size(event):
                            for trace in self.first_figure.object.data:
                                trace.marker.size = event.new
                            self.first_figure.param.trigger("object")

                        self.first_figure_slider.param.watch(
                            update_first_figure_marker_size, "value"
                        )
                    else:
                        print("Plotting without ADC slider")
                        self.first_scatter[str(val)] = go.Scatter(
                            x=self.first_figure_event_features[:, 0][mask],
                            y=self.first_figure_event_features[:, 1][mask],
                            mode="markers",
                            marker=dict(size=10, color=self.first_scatter_colors[val]),
                            name=str(val),
                        )
            elif "hit" in self.first_figure_label:
                mask = self.first_figure_event_hits[:, 0] == -1
                self.first_scatter["hit"] = go.Scatter(
                    x=self.first_figure_event_features[:, 0][mask],
                    y=self.first_figure_event_features[:, 1][mask],
                    mode="markers",
                    marker=dict(size=10, color="#DD4968"),
                    name="induction",
                )

                mask = self.first_figure_event_hits[:, 0] != -1
                self.first_scatter["hit"] = go.Scatter(
                    x=self.first_figure_event_features[:, 0][mask],
                    y=self.first_figure_event_features[:, 1][mask],
                    mode="markers",
                    marker=dict(size=10, color="#3B0F6F"),
                    name="hits",
                )
            else:
                label_index = self.tpc_meta["classes"][self.first_figure_label]
                label_vals = self.tpc_meta[f"{self.first_figure_label}_labels"]

                self.first_scatter_random_colors = [ii for ii in range(256)]
                random.shuffle(self.first_scatter_random_colors)
                self.first_scatter_colors = {
                    val: Magma256[int(self.first_scatter_random_colors[ii] % 256)]
                    for ii, val in enumerate(label_vals.values())
                }
                for key, val in label_vals.items():
                    if self.first_figure_plot_option == "Truth":
                        mask = self.first_figure_event_classes[:, label_index] == key
                    else:
                        if (
                            self.first_figure_label
                            not in self.available_prediction_labels
                        ):
                            continue
                        labels = np.argmax(
                            self.event_predictions[self.first_figure_label], axis=1
                        )
                        mask = labels == key
                    if np.sum(mask) == 0:
                        continue
                    if self.first_figure_adc_slider_option_bool is True:
                        self.first_scatter[str(val)] = go.Scatter(
                            x=self.first_figure_event_features[:, 0][mask],
                            y=self.first_figure_event_features[:, 1][mask],
                            mode="markers",
                            marker=dict(
                                size=abs(self.first_figure_event_features[:, 2][mask])
                                * 1000,
                                color=self.first_scatter_colors[val],
                            ),
                            name=str(val),
                        )
                        self.first_figure_slider = pn.widgets.FloatSlider()

                        def update_first_figure_marker_size(event):
                            for trace in self.first_figure.object.data:
                                trace.marker.size = event.new * 1000
                            self.first_figure.param.trigger("object")

                        self.first_figure_slider.param.watch(
                            update_first_figure_marker_size, "value"
                        )
                    else:
                        self.first_scatter[str(val)] = go.Scatter(
                            x=self.first_figure_event_features[:, 0][mask],
                            y=self.first_figure_event_features[:, 1][mask],
                            mode="markers",
                            marker=dict(size=10, color=self.first_scatter_colors[val]),
                            name=str(val),
                        )

            plotly_figure = go.Figure(data=list(self.first_scatter.values()))
            title = f"Plot I [{self.first_figure_plot_type} {self.first_figure_plot_option}]:"
            xaxis = dict(title="Channel [n]")
            yaxis = dict(title="TDC [10ns]")

        self.first_figure = plotly_figure.to_dict()
        self.first_figure["layout"].update(
            template="presentation",
            title=title,
            xaxis=xaxis,
            yaxis=yaxis,
            showlegend=True,
        )
        self.first_figure_pane.object = self.first_figure

    def plot_second_event(self, event):
        if self.second_figure_label == "adc":
            pass
        if self.second_figure_label == "MergeTree":
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
            if "cluster" in self.second_figure_label:
                label_index = self.tpc_meta["clusters"][
                    self.second_figure_label.replace("cluster_", "")
                ]
                label_vals = np.unique(
                    self.second_figure_event_clusters[:, label_index]
                )
                self.second_scatter = {}
                self.second_scatter_random_colors = [ii for ii in range(256)]
                random.shuffle(self.second_scatter_random_colors)
                self.second_scatter_colors = {
                    val: Magma256[int(self.second_scatter_random_colors[ii] % 256)]
                    for ii, val in enumerate(label_vals)
                }
                for val in label_vals:
                    if self.second_figure_plot_option == "Truth":
                        mask = self.second_figure_event_clusters[:, label_index] == val
                    else:
                        if (
                            self.second_figure_label
                            not in self.available_prediction_labels
                        ):
                            continue
                        labels = np.argmax(
                            self.event_predictions[self.second_figure_label], axis=1
                        )
                        mask = labels == val
                    if np.sum(mask) == 0:
                        continue
                    if self.second_figure_adc_slider_option_bool == True:
                        self.second_scatter[str(val)] = go.Scatter(
                            x=self.second_figure_event_features[:, 0][mask],
                            y=self.second_figure_event_features[:, 1][mask],
                            mode="markers",
                            marker=dict(
                                size=abs(self.second_figure_event_features[:, 2][mask]),
                                color=self.second_scatter_colors[val],
                            ),
                            name=str(val),
                        )
                        self.second_figure_slider = pn.widgets.FloatSlider()

                        def update_second_figure_marker_size(event):
                            for trace in self.second_figure.object.data:
                                trace.marker.size = event.new
                            self.second_figure.param.trigger("object")

                        self.second_figure_slider.param.watch(
                            update_second_figure_marker_size, "value"
                        )

                    else:
                        self.second_scatter[str(val)] = go.Scatter(
                            x=self.second_figure_event_features[:, 0][mask],
                            y=self.second_figure_event_features[:, 1][mask],
                            mode="markers",
                            marker=dict(size=10, color=self.second_scatter_colors[val]),
                            name=str(val),
                        )
            elif "hit" in self.second_figure_label:
                mask = self.second_figure_event_hits[:, 0] == -1
                self.second_scatter["hit"] = go.Scatter(
                    x=self.second_figure_event_features[:, 0][mask],
                    y=self.second_figure_event_features[:, 1][mask],
                    mode="markers",
                    marker=dict(size=10, color="#DD4968"),
                    name="induction",
                )
                mask = self.second_figure_event_hits[:, 0] != -1
                self.second_scatter["hit"] = go.Scatter(
                    x=self.second_figure_event_features[:, 0][mask],
                    y=self.second_figure_event_features[:, 1][mask],
                    mode="markers",
                    marker=dict(size=10, color="#3B0F6F"),
                    name="hits",
                )
            else:
                label_index = self.tpc_meta["classes"][self.second_figure_label]
                label_vals = self.tpc_meta[f"{self.second_figure_label}_labels"]
                self.second_scatter_random_colors = [ii for ii in range(256)]
                random.shuffle(self.second_scatter_random_colors)
                self.second_scatter_colors = {
                    val: Magma256[int(self.second_scatter_random_colors[ii] % 256)]
                    for ii, val in enumerate(label_vals.values())
                }
                for key, val in label_vals.items():
                    if self.second_figure_plot_option == "Truth":
                        mask = self.second_figure_event_classes[:, label_index] == key
                    else:
                        if (
                            self.second_figure_label
                            not in self.available_prediction_labels
                        ):
                            continue
                        labels = np.argmax(
                            self.event_predictions[self.second_figure_label], axis=1
                        )
                        mask = labels == key
                    if np.sum(mask) == 0:
                        continue
                    if self.second_figure_adc_slider_option_bool == True:
                        self.second_scatter[str(val)] = go.Scatter(
                            x=self.second_figure_event_features[:, 0][mask],
                            y=self.second_figure_event_features[:, 1][mask],
                            mode="markers",
                            marker=dict(
                                size=abs(self.second_figure_event_features[:, 2][mask]),
                                color=self.second_scatter_colors[val],
                            ),
                            name=str(val),
                        )
                        self.second_figure_slider = pn.widgets.FloatSlider()

                        def update_second_figure_marker_size(event):
                            for trace in self.second_figure.object.data:
                                trace.marker.size = event.new
                            self.second_figure.param.trigger("object")

                        self.second_figure_slider.param.watch(
                            update_second_figure_marker_size, "value"
                        )

                    else:
                        self.second_scatter[str(val)] = go.Scatter(
                            x=self.second_figure_event_features[:, 0][mask],
                            y=self.second_figure_event_features[:, 1][mask],
                            mode="markers",
                            marker=dict(size=10, color=self.second_scatter_colors[val]),
                            name=str(val),
                        )

            plotly_figure = go.Figure(data=list(self.second_scatter.values()))
            title = f"Plot II [{self.second_figure_plot_type} {self.second_figure_plot_option}]:"
            xaxis = dict(title="Channel [n]")
            yaxis = dict(title="TDC [10ns]")

        self.second_figure = plotly_figure.to_dict()
        self.second_figure["layout"].update(
            template="presentation",
            title=title,
            xaxis=xaxis,
            yaxis=yaxis,
            showlegend=True,
        )
        self.second_figure_pane.object = self.second_figure
