"""
To run the bokeh server from a terminal + browser,
run the command:
    > bokeh server --show blip_http_server.py
"""
from bokeh.io                            import curdoc
from blip.utils.event_display import BlipDisplay, PanelDisplay

# OLD BLIP DISPLAY BOKEH BASED:
# blip_display = BlipDisplay()
# curdoc().add_root(blip_display.layout)

# NEW BLIP DISPLAY PLOTLY BASED:
blip_display = PanelDisplay()