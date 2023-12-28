"""
Tools for displaying events
"""
import panel as pn
from bokeh.io import curdoc
from bokeh.layouts import row

from blip.utils.logger import Logger
from blip.utils.event_display.panel_wire_lartpc import WireLArTPCPanelDisplay
from blip.utils.event_display.panel_pixel_lartpc import PixelLArTPCPanelDisplay
from panel.widgets import Select

class PanelDisplay:

    def __init__(
        self,
        document=None
    ):
        """
        Initialize the PanelDisplay object
        """

        if document is None:
            self.document = curdoc()
        else:
            self.document = document
        
        self.tabs = pn.Tabs()
        self.nplots_select = Select(
            name="Number of plots:",
            value="2",
            options=["1","2"],
            width_policy="fixed",
            width=50,
        )
        self.nplots_select.param.watch(self.update_nplots, "value")
        self.construct_widgets(self.document)

    def construct_widgets(
        self,
        document
    ):
        self.construct_blip_widgets(document)
        self.construct_wire_lartpc_display(document,self.nplots_select.value)
        self.construct_pixel_lartpc_display(document)
        self.construct_semantic_widgets(document)
        self.construct_point_net_embedding_widgets(document)

        self.logo_img = "/workspace/Blip/data/neutrino.png"
        self.tabs.clear()
        self.tabs.extend([
            ("Blip Runner", pn.Row(
                pn.pane.Markdown("Blip Runner"), 
                self.blip_layout)),
            ("Wire LArTPC Display", pn.Row(
                pn.Column(
                    pn.pane.Markdown("Wire LArTPC Display"),
                    self.nplots_select,
                    width=150,
                ),
                self.wire_lartpc_display.layout
            )),
            ("Pixel LArTPC Display", pn.Row(
                pn.pane.Markdown("Pixel LArTPC Display"),
                self.pixel_lartpc_display.layout
            )),
            ("BlipSegmentation", pn.Row(
                pn.pane.Markdown("BlipSegmentation"), 
                self.semantic_model_layout)),
            ("BlipGraph", pn.Row(
                pn.pane.Markdown("BlipGraph"),
                self.point_net_embedding_layout
            )),
        ])
        self.tabs.active = 1
        self.tabs.param.trigger('active')

        self.template = pn.template.MaterialTemplate(
            title='BLIP DISPLAY',
            logo=self.logo_img,
            main=self.tabs,
        ).servable()


    def update_nplots(self,event):
        self.nplots_select.value = event.new
        self.construct_widgets(self.document)

    def construct_blip_widgets(self,document):
        self.blip_layout = row()

    def construct_edep_widgets(self,document):
        self.edep_layout = row()

    def construct_wire_lartpc_display(self,document,nplots):
        if hasattr(self, 'wire_lartpc_display'):
            # Update the existing WireLArTPCPanelDisplay object
            self.wire_lartpc_display.document = document

        self.wire_lartpc_display = WireLArTPCPanelDisplay(self.document,self.nplots_select.value)
    
    def construct_pixel_lartpc_display(self,document):
        self.pixel_lartpc_display = PixelLArTPCPanelDisplay(document)

    def construct_wire_plane_channel_widgets(self,document):
        self.wire_plane_channel_layout = row()

    def construct_semantic_widgets(self,document):
        self.semantic_model_layout = row()

    def construct_point_net_embedding_widgets(self,document):
        self.point_net_embedding_layout = row()

display = PanelDisplay()
