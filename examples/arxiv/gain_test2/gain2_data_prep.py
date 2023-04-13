"""
Training loop for a BLIP model   
"""
# blip imports
from blip.dataset.arrakis import Arrakis


if __name__ == "__main__":

    prepare_data = True
    if prepare_data:
        arrakis_dataset = Arrakis(
            "../../../ArrakisEventDisplay/data/multiple_neutron_arrakis4.root"
        )
        arrakis_dataset.generate_training_data()