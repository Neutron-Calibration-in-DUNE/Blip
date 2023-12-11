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
from bokeh.models                        import Slider
from bokeh.models                        import Select, MultiSelect, FileInput
from bokeh.models                        import Button, CheckboxGroup, TextInput
from bokeh.models                        import CategoricalColorMapper, Toggle
from bokeh.models                        import CheckboxButtonGroup, CustomJS
from bokeh.models                        import Paragraph, PreText, Dropdown
from bokeh.models                        import ColumnDataSource, RadioGroup
from bokeh.models                        import ColorBar, LinearColorMapper
from bokeh.events                        import Tap
from bokeh.palettes                      import Turbo256, Category20, Category20b, TolRainbow, Magma256
from bokeh.transform                     import linear_cmap
from bokeh.transform                     import factor_cmap, factor_mark
from bokeh.server.server                 import Server
from bokeh.command.util                  import build_single_handler_applications
from bokeh.document                      import Document

from blip.utils.logger                   import Logger
from blip.utils.event_display.panel_tpc  import TPCPanelDisplay
# from blip.utils.event_display.panel_merge  import MergePanelDisplay #work on progress

class PanelDisplay:

    def __init__(self,
            document = None
        ):
        # File folder and file select.
        # These act as drop down menus which update upon 
        # selecting, one for wire planes and another for 
        # edeps.

        if document == None: self.document = curdoc()
        else:                self.document = document

        self.construct_widgets(self.document)

    def construct_widgets(self,document):
        self.construct_blip_widgets(document)
        self.construct_tpc_display(document)
        self.construct_semantic_widgets(document)
        self.construct_point_net_embedding_widgets(document)
        # self.construct_merge_tree_widgets(document)

        self.logo_img = "data/neutrino.png"
        self.tabs     = pn.Tabs(
            ("Blip Runner",        pn.Row(pn.pane.Markdown("Content for Blip Runner"),        self.blip_layout)),
            ("LArTPC Display",     pn.Row(pn.pane.Markdown("Content for LArTPC Display"),     self.tpc_display.wire_plane_layout)),
            ("Semantic Model",     pn.Row(pn.pane.Markdown("Content for Semantic Model"),     self.semantic_model_layout)),
            ("PointNet Embedding", pn.Row(pn.pane.Markdown("Content for PointNet Embedding"), self.point_net_embedding_layout)),
            # ("PointNet Embedding", pn.Row(pn.pane.Markdown("Content for Merge Tree Display"), self.merge_display.merge_tree_layout)),
            active = 1,
        )
        self.tabs.param.trigger('active')
        

        self.template = pn.template.MaterialTemplate(
            title   = 'BLIP DISPLAY',
            logo    = self.logo_img,  # Add the image to the logo
            main    = self.tabs,
        ).servable()

    def construct_blip_widgets(self, document):
        self.blip_layout = row()

    def construct_edep_widgets(self, document):
        self.edep_layout = row()

    def construct_tpc_display(self, document):
        self.tpc_display = TPCPanelDisplay(document)
    
    def construct_wire_plane_channel_widgets(self, document):
        self.wire_plane_channel_layout = row()
    
    def construct_semantic_widgets(self, document):
        self.semantic_model_layout = row()

    def construct_point_net_embedding_widgets(self, document):
        self.point_net_embedding_layout = row()

    # def construct_merge_tree_widgets(self, document):
    #     self.merge_display = MergePanelDisplay(document)

display = PanelDisplay()