"""
To run the bokeh server from a terminal + browser,
run the command:
    > bokeh server --show blip_http_server.py
"""
from bokeh.io import output_notebook, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler

from blip.utils.event_display import BlipDisplay
from bokeh.io import curdoc

blip_display = BlipDisplay()
curdoc().add_root(blip_display.layout)