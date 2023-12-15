"""
Script for generating hyperparameter config files from a base config
"""
import os
import shutil
import argparse
import random
import csv

from blip.utils.logger import Logger
from blip.utils.config import ConfigParser
from blip.utils.utils import traverse_nested_dictionary, update_nested_dictionary_value
from blip.utils.utils import generate_random_dictionaries


def run():
    parser = argparse.ArgumentParser(
        prog='BLIP ML Hyper-parameter Config Generator',
        description='This program constructs a BLIP ML...',
        epilog='...'
    )
    parser.add_argument(
        'config_file', metavar='<str>.yml', type=str,
        help='config file specification for a BLIP module.'
    )
    parser.add_argument(
        '-hyper_parameter_location', dest='hyper_parameter_location', default='/local_scratch',
        help='location for the local scratch directory.'
    )

    logger = Logger('hyper_parameter_generator', output="both", file_mode="w")

    args = parser.parse_args()
    config_file = args.config_file
    hyper_parameter_location = args.hyper_parameter_location

    config_parser = ConfigParser(config_file)
    config = config_parser.data

    if "hyper_parameters" not in config.keys():
        logger.error('hyper_parameters not specified in config!')
    if "iterations" not in config["hyper_parameters"].keys():
        logger.error("iterations not in hyper_parameters section of config!")
    if "model" not in config.keys():
        logger.error("model section not specified in config!")
    if "model_parameters" not in config['hyper_parameters'].keys():
        logger.error("model_parameters not in hyper_parameters section of config!")

    model_name = list(config['hyper_parameters']['model_parameters'].keys())[0]

    if model_name not in config["model"].keys():
        logger.error(f'model_name: {model_name} not in model section of config!')

    hyper_parameter_dict = config['hyper_parameters']['model_parameters'][model_name]
    model_dict = config['model'][model_name]

    iterations = config["hyper_parameters"]["iterations"]

    model_dicts = generate_random_dictionaries(
        model_dict,
        num_configs=iterations,
        sample_dictionary=hyper_parameter_dict
    )

    if len(model_dicts) < iterations:
        logger.warn(
            f'number of possible model configurations ({len(model_dicts)}) ' +
            f'is less than specified iterations ({iterations})!'
        )
    logger.info(f'setting up {len(model_dicts)} configurations')
    hyper_parameter_folders = []

    for ii in range(len(model_dicts)):
        random_config = config.copy()
        random_config['model'][model_name] = model_dicts[ii]
        if not os.path.isdir(f'{hyper_parameter_location}/hyper_parameter_iteration_{ii}'):
            os.makedirs(f'{hyper_parameter_location}/hyper_parameter_iteration_{ii}')
        hyper_parameter_folders.append(f'{hyper_parameter_location}/hyper_parameter_iteration_{ii}')
        config_parser.save_config(
            random_config, f'{hyper_parameter_location}/hyper_parameter_iteration_{ii}/hyper_parameter_config.yaml'
        )
    with open(f'{hyper_parameter_location}/hyper_parameter_data.csv', "w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerows([
            hyper_parameter_folders
        ])


if __name__ == "__main__":
    run()
