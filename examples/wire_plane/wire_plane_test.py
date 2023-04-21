"""
Training loop for a BLIP model   
"""
# blip imports
from blip.module import Module
from blip.dataset.wire_plane import WirePlanePointCloud


if __name__ == "__main__":

    blip_module = Module("blip", "config/test2.yaml")