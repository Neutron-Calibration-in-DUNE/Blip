"""
Tools for displaying events
"""
import os,imageio
import numpy  as np
import panel  as pn
import pandas as pd
from matplotlib import pyplot as plt
from pathlib    import Path

from bokeh.io                            import curdoc, output_notebook, show
from bokeh.application                   import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.layouts                       import row, column, layout
from bokeh.plotting                      import figure, show
from bokeh.models                        import TabPanel, Tabs, TapTool
from bokeh.models                        import Div, RangeSlider, Spinner
from bokeh.models                        import Select, MultiSelect, FileInput
from bokeh.models                        import Button, CheckboxGroup, TextInput
from bokeh.models                        import CategoricalColorMapper, Toggle
from bokeh.models                        import CheckboxButtonGroup, CustomJS
from bokeh.models                        import Paragraph, PreText, Dropdown
from bokeh.models                        import ColumnDataSource, RadioGroup
from bokeh.events                        import Tap
from bokeh.palettes                      import Turbo256, Category20, Category20b, TolRainbow, Magma256
from bokeh.transform                     import linear_cmap
from bokeh.transform                     import factor_cmap, factor_mark
from bokeh.server.server                 import Server
from bokeh.command.util                  import build_single_handler_applications
from bokeh.document                      import Document

from blip.utils.logger            import Logger
from blip.utils.event_display.tpc import TPCDisplay

class PanelDisplay:

    def __init__(self):
        # File folder and file select.
        # These act as drop down menus which update upon 
        # selecting, one for wire planes and another for 
        # edeps.

        self.construct_widgets()
        self.template = pn.template.MaterialTemplate(
            title='Blip Display',
            sidebar=pn.WidgetBox(
                self.sidebar_markdown,
                self.file_browser,
                max_width=350,
                sizing_mode='stretch_width'
            ),
            main=pn.Tabs(
                ("Wire Plane Display",None),
                ("Energy Deposit Display",None),
            ),
        ).servable()

    def construct_widgets(self):
        self.siderbar_text = """
        #  Blip Display

        This application ...
        """
        self.sidebar_markdown = pn.pane.Markdown(
            self.siderbar_text,
            margin=(0,10)
        )
        self.file_browser = pn.widgets.FileInput(sizing_mode='stretch_width')
   

display = PanelDisplay()