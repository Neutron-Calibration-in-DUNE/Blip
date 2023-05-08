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
from bokeh.models import CheckboxButtonGroup, CustomJS
from bokeh.models import ColumnDataSource
from bokeh.palettes import Turbo256
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
        
        if document == None:
            self.document = curdoc()
        else:
            self.document = document

        self.construct_widgets(self.document)

    def update_available_folders(self):
        self.available_folders = ['.', '..'] + [
            f.parts[-1] for f in Path(self.file_folder).iterdir() if f.is_dir()
        ]

    def update_available_files(self):
        self.available_files = [
            f.parts[-1] for f in Path(self.file_folder).iterdir() if f.is_file()
        ]

    def construct_widgets(self,
        document
    ):
        self.input_figure = figure()
        self.output_figure = figure()

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
            title="Blip file", value="", 
            options=self.available_files,
            width_policy='fixed', width=350
        )
        if len(self.available_files) > 0:
            self.file_select.value = self.available_files[0]
            self.input_file = self.file_select.value
        self.file_select.on_change(
            "value", self.update_input_file
        )
        self.button = Button(
            label="Load file", 
            button_type="success",
            width_policy='fixed', width=100
        )
        self.button.on_click(
            self.load_input_file
        )
        # construct the layout
        self.layout = row(
            column(
                self.file_folder_select,
                self.file_select,
                self.button,
                width_policy = 'fixed', width=400
            ),
            column(self.input_figure),
            column(self.output_figure)
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
    
    def load_input_file(self):
        print(self.input_file)