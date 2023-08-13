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
from blip.utils.event_display.tpc import TPCDisplay

class BlipDisplay:
    """
    """
    def __init__(self,
        document = None
    ):
        if document == None:
            self.document = curdoc()
        else:
            self.document = document

        self.construct_widgets(self.document)

    def construct_widgets(self,
        document
    ):
        self.construct_header_widgets(document)
        self.construct_blip_widgets(document)
        self.construct_tpc_display(document)
        self.construct_semantic_widgets(document)
        self.construct_point_net_embedding_widgets(document)

        # enumerate the different tabs
        self.header_tab = TabPanel(
            child=self.header_layout, title="Blip Display"
        )
        self.blip_tab = TabPanel(
            child=self.blip_layout, title="Blip Runner"
        )
        self.wire_plane_display_tab = TabPanel(
            child=self.tpc_display.wire_plane_layout, title="LArTPC Display"
        )
        self.semantic_model_tab = TabPanel(
            child=self.semantic_model_layout, title="Semantic Model"
        )
        self.point_net_embedding_tab = TabPanel(
            child=self.point_net_embedding_layout, title="PointNet Embedding"
        )
        self.tab_layout = Tabs(tabs=[
            self.header_tab,
            self.blip_tab,
            self.wire_plane_display_tab,
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

    def construct_tpc_display(self,
        document
    ):
        self.tpc_display = TPCDisplay(document)
    
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