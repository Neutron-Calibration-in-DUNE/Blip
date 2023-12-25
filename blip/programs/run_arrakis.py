"""
Blip main program
"""
import torch
import os
import argparse

from blip.utils.logger import Logger
from blip.utils.config import ConfigParser

from blip.dataset.arrakis import Arrakis


def run():
    """
    Arrkis main program.
    """
    parser = argparse.ArgumentParser(
        prog='Arrakis Module Runner',
        description='This program runs the Arrakis module ' +
                    'from a config file.',
        epilog='...'
    )
    parser.add_argument(
        'config_file', metavar='<str>.yml', type=str,
        help='config file specification for a BLIP module.'
    )
    parser.add_argument(
        '-n', dest='name', default='arrakis',
        help='name for this run (default "arrakis").'
    )
    parser.add_argument(
        '-scratch', dest='local_scratch', default='/local_scratch',
        help='location for the local scratch directory.'
    )
    parser.add_argument(
        '-blip', dest='local_blip', default='/local_blip',
        help='location for the local blip directory.'
    )
    parser.add_argument(
        '-data', dest='local_data', default='/local_data',
        help='location for the local data directory.'
    )
    args = parser.parse_args()
    # Setup config file.
    name = args.name
    if not os.path.isdir(args.local_scratch):
        args.local_scratch = './'
    if not os.path.isdir(args.local_blip):
        args.local_blip = './'
    local_blip_files = [
        args.local_blip + '/' + file
        for file in os.listdir(path=os.path.dirname(args.local_blip))
    ]
    if not os.path.isdir(args.local_data):
        args.local_data = './'
    local_data_files = [
        args.local_data + '/' + file
        for file in os.listdir(path=os.path.dirname(args.local_data))
    ]
    os.environ['LOCAL_SCRATCH'] = args.local_scratch
    os.environ['LOCAL_BLIP'] = args.local_blip
    os.environ['LOCAL_DATA'] = args.local_data

    config = ConfigParser(args.config_file).data
    logger = Logger(name, output="both", file_mode="w")
    logger.info("configuring arrakis...")

    if "module" not in config.keys():
        logger.error('"module" section not specified in config!')
    if "dataset" not in config.keys():
        logger.error('"dataset" section not specified in config!')
    system_info = logger.get_system_info()
    for key, value in system_info.items():
        logger.info(f"system_info - {key}: {value}")

    meta = {
        'config_file':  args.config_file,
        'local_scratch':    args.local_scratch,
        'local_blip':       args.local_blip,
        'local_data':       args.local_data,
        'local_blip_files': local_blip_files,
        'local_data_files': local_data_files
    }
    logger.info(f'"local_scratch" directory set to: {args.local_scratch}.')
    logger.info(f'"local_blip" directory set to: {args.local_blip}.')
    logger.info(f'"local_data" directory set to: {args.local_data}.')

    if "verbose" in config["module"]:
        if not isinstance(config["module"]["verbose"], bool):
            logger.error(f'"module:verbose" must be of type bool, but got {type(config["module"]["verbose"])}!')
        meta["verbose"] = config["module"]["verbose"]
    else:
        meta["verbose"] = False

    # Eventually we will want to check that the order of the modules makes sense,
    # and that the data products are compatible and available for the different modes.

    # check for devices
    if "gpu" not in config["module"].keys():
        logger.warn('"module:gpu" not specified in config!')
        gpu = None
    else:
        gpu = config["module"]["gpu"]
    if "gpu_device" not in config["module"].keys():
        logger.warn('"module:gpu_device" not specified in config!')
        gpu_device = None
    else:
        gpu_device = config["module"]["gpu_device"]

    if torch.cuda.is_available():
        logger.info("CUDA is available with devices:")
        for ii in range(torch.cuda.device_count()):
            device_properties = torch.cuda.get_device_properties(ii)
            cuda_stats = f"name: {device_properties.name}, "
            cuda_stats += f"compute: {device_properties.major}.{device_properties.minor}, "
            cuda_stats += f"memory: {device_properties.total_memory}"
            logger.info(f" -- device: {ii} - " + cuda_stats)

    # set gpu settings
    if gpu:
        if torch.cuda.is_available():
            if gpu_device >= torch.cuda.device_count() or gpu_device < 0:
                logger.warn(f"desired gpu_device '{gpu_device}' not available, using device '0'")
                gpu_device = 0
            meta['device'] = torch.device(f"cuda:{gpu_device}")
            logger.info(
                f"CUDA is available, using device {gpu_device}" +
                f": {torch.cuda.get_device_name(gpu_device)}"
            )
        else:
            gpu = False
            logger.warn("CUDA not available! Using the cpu")
            meta['device'] = torch.device("cpu")
    else:
        logger.info("using cpu as device")
        meta['device'] = torch.device("cpu")
    meta['gpu'] = gpu

    # Configure the dataset
    logger.info("configuring dataset.")
    dataset_config = config['dataset']
    dataset_config["device"] = meta['device']
    dataset_config["verbose"] = meta["verbose"]

    # default to what's in the configuration file. May decide to deprecate in the future
    if ("simulation_folder" in dataset_config):
        simulation_folder = dataset_config["simulation_folder"]
        logger.info(
            "Set simulation file folder from configuration. " +
            f" simulation_folder : {simulation_folder}"
        )
    elif ('BLIP_SIMULATION_PATH' in os.environ):
        logger.debug('Found BLIP_SIMULATION_PATH in environment')
        simulation_folder = os.environ['BLIP_SIMULATION_PATH']
        logger.info(
            "Setting simulation path from Enviroment." +
            f" BLIP_SIMULATION_PATH = {simulation_folder}"
        )
    else:
        logger.error('No dataset_folder specified in environment or configuration file!')

    # check for processing simulation files
    if "simulation_files" in dataset_config and dataset_config["process_simulation"]:
        if 'simulation_type' not in dataset_config.keys():
            logger.error('simulation_type not specified in dataset config!')
        _ = Arrakis(
            name,
            dataset_config,
            meta
        )


if __name__ == "__main__":
    run()
