"""
Tools for displaying events
"""
import panel as pn
from bokeh.io import curdoc
from bokeh.layouts import row

from blip.utils.logger import Logger
from blip.utils.event_display.panel_wire_lartpc import WireLArTPCPanelDisplay
from blip.utils.event_display.panel_pixel_lartpc import PixelLArTPCPanelDisplay


class PanelDisplay:

    def __init__(
        self,
        document=None
    ):
        # File folder and file select.
        # These act as drop down menus which update upon
        # selecting, one for wire planes and another for
        # edeps.

        if document is None:
            self.document = curdoc()
        else:
            self.document = document

        self.construct_widgets(self.document)

    def construct_widgets(
        self,
        document
    ):
        self.construct_blip_widgets(document)
        self.construct_wire_lartpc_display(document)
        self.construct_pixel_lartpc_display(document)
        self.construct_semantic_widgets(document)
        self.construct_point_net_embedding_widgets(document)

        self.logo_img = "/workspace/Blip/data/neutrino.png"
        self.tabs = pn.Tabs(
            ("Blip Runner", pn.Row(pn.pane.Markdown("Blip Runner"), self.blip_layout)),
            ("Wire LArTPC Display", pn.Row(
                pn.pane.Markdown("Wire LArTPC Display"),
                self.wire_lartpc_display.layout
            )),
            ("Pixel LArTPC Display", pn.Row(
                pn.pane.Markdown("Pixel LArTPC Display"),
                self.pixel_lartpc_display.layout
            )),
            ("BlipSegmentation", pn.Row(pn.pane.Markdown("BlipSegmentation"), self.semantic_model_layout)),
            ("BlipGraph", pn.Row(
                pn.pane.Markdown("BlipGraph"),
                self.point_net_embedding_layout
            )),
            active=1,
        )
        self.tabs.param.trigger('active')

        self.template = pn.template.MaterialTemplate(
            title='BLIP DISPLAY',
            logo=self.logo_img,
            main=self.tabs,
        ).servable()

    def construct_blip_widgets(self, document):
        self.blip_layout = row()

    def construct_edep_widgets(self, document):
        self.edep_layout = row()

    def construct_wire_lartpc_display(self, document):
        self.wire_lartpc_display = WireLArTPCPanelDisplay(document)
        
    def construct_pixel_lartpc_display(self, document):
        self.pixel_lartpc_display = PixelLArTPCPanelDisplay(document)

    def construct_wire_plane_channel_widgets(self, document):
        self.wire_plane_channel_layout = row()

    def construct_semantic_widgets(self, document):
        self.semantic_model_layout = row()

    def construct_point_net_embedding_widgets(self, document):
        self.point_net_embedding_layout = row()

display = PanelDisplay()
