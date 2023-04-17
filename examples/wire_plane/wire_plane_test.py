"""
Training loop for a BLIP model   
"""
# blip imports
from blip.dataset.wire_plane import WirePlanePointCloud


if __name__ == "__main__":

    prepare_data = True
    if prepare_data:
        wire_plane_dataset = WirePlanePointCloud(
            "wire_plane_test",
            "../../data/arrakis_output_10.root"
        )
        wire_plane_dataset.generate_training_data()