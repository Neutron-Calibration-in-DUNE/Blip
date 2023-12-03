from blip.utils.logger import Logger
from blip.utils.config import ConfigParser
from blip.utils.utils import get_datetime

from blip.dataset.arrakis import Arrakis
from blip.dataset.arrakis_nd import ArrakisND
from blip.dataset.mssm import MSSM
from blip.dataset.blip import BlipDataset
from blip.dataset.vanilla import VanillaDataset
from blip.utils.loader import Loader
from blip.module import ModuleHandler
from blip.module.common import module_types

import torch
import os
import shutil
os.environ["TQDM_NOTEBOOK"] = "false"


def wrangle_data(
    config_file:    str,
    run_name:       str = None,
    local_scratch:  str = './',
    local_blip:     str = './',
    local_data:     str = './',
    anomaly:        bool = False
):
    # set up directories
    if not os.path.isdir(local_scratch):
        local_scratch = './'
    if not os.path.isdir(local_blip):
        local_blip = './'
    if not os.path.isdir(local_data):
        local_data = './'
    local_blip_files = [
        local_blip + '/' + file
        for file in os.listdir(path=os.path.dirname(local_blip))
    ]
    local_data_files = [
        local_data + '/' + file
        for file in os.listdir(path=os.path.dirname(local_data))
    ]
    os.environ['LOCAL_SCRATCH'] = local_scratch
    os.environ['LOCAL_BLIP'] = local_blip
    os.environ['LOCAL_DATA'] = local_data

    logger = Logger('data_wrangler', output="both", file_mode="w")
    logger.info("configuring data...")

    # begin parsing configuration file
    if config_file is None:
        logger.error('no config_file specified in parameters!')

    config = ConfigParser(config_file).data

    if anomaly:
        logger.info(f'setting anomaly detection to {anomaly}')
        torch.autograd.set_detect_anomaly(bool(anomaly))

    if "module" not in config.keys():
        logger.error('"module" section not specified in config!')
    if "dataset" not in config.keys():
        logger.error('"dataset" section not specified in config!')
    if "loader" not in config.keys():
        logger.error('"loader" section not specified in config!')
    system_info = logger.get_system_info()
    for key, value in system_info.items():
        logger.info(f"system_info - {key}: {value}")

    # get run_name
    if run_name is None:
        run_name = config['module']['module_name']
    # add unique datetime
    now = get_datetime()
    run_name += f"_{now}"
    local_run = local_scratch + '/' + run_name

    meta = {
        'now':              now,
        'run_name':         run_name,
        'config_file':      config_file,
        'run_directory':    local_run,
        'local_scratch':    local_scratch,
        'local_blip':       local_blip,
        'local_data':       local_data,
        'local_blip_files': local_blip_files,
        'local_data_files': local_data_files
    }
    logger.info(f'"now" set to: {now}')
    logger.info(f'"run_name" set to: {run_name}')
    logger.info(f'"run" directory set to: {local_run}.')
    logger.info(f'"local_scratch" directory set to: {local_scratch}.')
    logger.info(f'"local_blip" directory set to: {local_blip}.')
    logger.info(f'"local_data" directory set to: {local_data}.')

    # create .tmp directory
    if not os.path.isdir(f"{local_scratch}/.backup"):
        os.makedirs(f"{local_scratch}/.backup")
    if os.path.isdir(f"{local_scratch}/.backup/.tmp"):
        shutil.rmtree(f"{local_scratch}/.backup/.tmp")
    if os.path.isdir(f"{local_scratch}/.tmp"):
        shutil.move(
            f"{local_scratch}/.tmp/",
            f"{local_scratch}/.backup/"
        )
        logger.info("copied old .tmp to .backup in local_scratch directory.")
    os.makedirs(f"{local_scratch}/.tmp")

    # create run directory
    if not os.path.isdir(local_run):
        os.makedirs(local_run)

    # set verbosity of logger
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
            " simulation_folder : {simulation_folder}"
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
        if dataset_config["simulation_type"] == "LArSoft":
            _ = Arrakis(
                run_name,
                dataset_config,
                meta
            )
        elif dataset_config["simulation_type"] == "larnd-sim":
            _ = ArrakisND(
                run_name,
                dataset_config,
                meta
            )
        elif dataset_config["simulation_type"] == "MSSM":
            _ = MSSM(
                run_name,
                dataset_config,
                meta
            )
        else:
            logger.error(f'specified "dataset:simulation_type" "{dataset_config["simulation_type"]}" not an allowed type!')

    logger.info("configuring dataset.")
    dataset = BlipDataset(
        run_name,
        dataset_config,
        meta
    )
    meta['dataset'] = dataset

    # Configure the loader
    logger.info("configuring loader.")
    loader_config = config['loader']
    loader = Loader(
        run_name,
        loader_config,
        meta
    )
    meta['loader'] = loader

    # Configure the module handler
    logger.info("configuring modules.")
    module_config = config
    module_handler = ModuleHandler(
        run_name,
        module_config,
        meta=meta
    )
    return meta, module_handler


def parse_command_line_config(
    params
):
    # set up local scratch and local blip
    if params.local_scratch is not None:
        if not os.path.isdir(params.local_scratch):
            params.local_scratch = './'
    else:
        params.local_scratch = './'
    if params.local_blip is not None:
        if not os.path.isdir(params.local_blip):
            params.local_blip = './'
    else:
        params.local_blip = './'

    # set up local data
    if params.local_data is not None:
        if not os.path.isdir(params.local_data):
            params.local_data = './'
    else:
        params.local_data = './'

    return wrangle_data(
        params.config_file,
        params.name,
        params.local_scratch,
        params.local_blip,
        params.local_data,
        params.anomaly
    )
