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

# import holoviews as hv
# from holoviews import opts
# from holoviews.operation.datashader import datashade, shade, dynspread, spread
# from holoviews.operation.datashader import rasterize, ResamplingOperation
# from holoviews.operation import decimate
# hv.extension('bokeh')
import pandas as pd

# import panel as pn
import os
import imageio

from blip.utils.logger import Logger

class BlipDisplay:
    """
    """
    def __init__(self,
        document = None
    ):
        self.file_folder = '.'
        self.available_files = []
        self.input_file = ''
        if document == None:
            self.document = curdoc()
        else:
            self.document = document

        self.construct_widgets(self.document)

    def construct_widgets(self,
        document
    ):
        self.input_figure = figure()
        self.output_figure = figure()

        # first column
        # self.file_input = FileInput(accept='.txt')
        # self.file_input.on_change('filename', self.update_folder)
        # self.file_list = MultiSelect(title='Files in folder', height=300, width=200)
        # self.file_list.on_change('value', self.select_file)
        # self.update_file_list(self.file_folder)

        self.file_folder_input = TextInput(value="data/", title="File folder location:")
        self.file_folder_input.on_change("value", self.update_file_folder)
        self.file_multi_select = Select(title="Blip file", value="", options=self.available_files)
        self.file_multi_select.on_change("value", self.update_input_file)

        self.button = Button(label="Foo", button_type="success")
        # self.button.on_click()
        self.layout = row(
            column(
                self.file_folder_input,
                self.file_multi_select
                # self.file_input,
                # self.file_list
            ),
            column(self.input_figure),
            column(self.output_figure, self.button)
        )

        document.add_root(self.layout)
        document.title = "Blip Display"

    def update_folder(self, attr, old, new):
        self.folder = os.path.dirname(new)
        self.update_file_list(self.folder)

    def update_file_list(self, folder):
        self.file_list.items = ['..'] + sorted(os.listdir(folder))

    def select_file(self, attr, old, new):
        selected_file = self.file_list.value
        if selected_file == '..':
            parent_folder = os.path.abspath(os.path.join(self.folder, os.pardir))
            self.update_file_list(parent_folder)
            self.folder = parent_folder
        elif os.path.isfile(os.path.join(self.folder, selected_file)):
            print(f'Selected file: {selected_file}')
        elif os.path.isdir(os.path.join(self.folder, selected_file)):
            self.folder = os.path.join(self.folder, selected_file)
            self.update_file_list(self.folder)
    
    def update_file_folder(self, attr, old, new):
        if os.path.isdir(new):
            self.file_folder = new
            self.available_files = os.listdir(self.file_folder)
            self.file_multi_select.options = self.available_files
        else:
            self.file_folder_input.title = f"File folder location: ({new} not valid!)"
    
    def update_input_file(self, attr, old, new):
        self.input_file = new