"""
Training loop for a BLIP model   
"""
import argparse

# blip imports
from blip.module import Module

parser = argparse.ArgumentParser(
    prog='BLIP Module Runner',
    description='This program constructs a BLIP module '+
        'from a config file, and then runs training/inference.',
    epilog='...'
)
parser.add_argument(
    'config_file', metavar='<str>.yml', type=str,
    help='config file specification for a BLIP module.'
)
parser.add_argument(
    '-n', dest='name', default='blip',
    help='name for this run (default "blip").'
)

if __name__ == "__main__":
    
    args = parser.parse_args()
    blip_module = Module(
        args.name, 
        args.config_file
    )