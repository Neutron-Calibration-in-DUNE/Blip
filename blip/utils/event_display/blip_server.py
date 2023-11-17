"""
To run the bokeh server in a jupyter notebook, 
import the create_server function from this
script and run it in the notebook.

To run the bokeh server from a terminal + browser,
run the command:
    > bokeh server --show blip_http_server.py
"""
from bokeh.io                            import curdoc,output_notebook, show
from bokeh.layouts                       import column
from bokeh.models                        import ColumnDataSource
from bokeh.plotting                      import figure
from bokeh.server.server                 import Server
from bokeh.application                   import Application
from bokeh.application.handlers.function import FunctionHandler

from blip.utils.event_display import BlipDisplay

def blip_app(doc):
    blip_display = BlipDisplay(doc)
    doc.add_root(blip_display.layout)


def create_server():
    app = Application(FunctionHandler(blip_app))
    blip_server = Server(
        {'/': app},
        port=0,
        allow_websocket_origin=["localhost:8888"]
    )
    blip_server.start()
    output_notebook()
    show(app, notebook_url="localhost:8888")
    return blip_server