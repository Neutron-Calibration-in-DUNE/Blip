"""
"""
from blip.utils.logger import Logger
from blip.utils.config import ConfigParser
from blip.utils.utils import get_datetime

from blip.module import ModuleHandler

import torch
import os
import shutil
os.environ["TQDM_NOTEBOOK"] = "false"


class BlipRunner:
    """
    """
    def __init__(
        self,
        config_file:    str,
        run_name:       str = None,
        local_scratch:  str = './',
        local_blip:     str = './',
        local_data:     str = './',
        anomaly:        bool = False
    ):
        # set up directories
        self.config_file = config_file
        self.run_name = run_name
        self.local_scratch = local_scratch
        self.local_blip = local_blip
        self.local_data = local_data
        self.anomaly = anomaly

        if not os.path.isdir(self.local_scratch):
            self.local_scratch = './'
        if not os.path.isdir(self.local_blip):
            self.local_blip = './'
        if not os.path.isdir(self.local_data):
            self.local_data = './'
        self.local_blip_files = [
            self.local_blip + '/' + file
            for file in os.listdir(path=os.path.dirname(self.local_blip))
        ]
        self.local_data_files = [
            self.local_data + '/' + file
            for file in os.listdir(path=os.path.dirname(self.local_data))
        ]
        os.environ['LOCAL_SCRATCH'] = self.local_scratch
        os.environ['LOCAL_BLIP'] = self.local_blip
        os.environ['LOCAL_DATA'] = self.local_data

        self.logger = Logger('blip_runner', output="both", file_mode="w")

        # begin parsing configuration file
        if self.config_file is None:
            self.logger.error('no config_file specified in parameters!')

        self.config = ConfigParser(self.config_file).data

        if self.anomaly:
            self.logger.info(f'setting anomaly detection to {self.anomaly}')
            torch.autograd.set_detect_anomaly(bool(self.anomaly))

        if "module" not in self.config.keys():
            self.logger.error('"module" section not specified in config!')

        self.set_up_directories()
        self.set_up_meta()
        self.set_up_devices()
        self.set_up_modules()

    def set_up_directories(self):
        # create .tmp and .backup directories
        if not os.path.isdir(f"{self.local_scratch}/.backup"):
            os.makedirs(f"{self.local_scratch}/.backup")
        if os.path.isdir(f"{self.local_scratch}/.backup/.tmp"):
            shutil.rmtree(f"{self.local_scratch}/.backup/.tmp")
        if os.path.isdir(f"{self.local_scratch}/.tmp"):
            shutil.move(
                f"{self.local_scratch}/.tmp/",
                f"{self.local_scratch}/.backup/"
            )
            self.logger.info("copied old .tmp to .backup in local_scratch directory.")
        os.makedirs(f"{self.local_scratch}/.tmp")

    def set_up_meta(self):
        self.logger.info("configuring meta...")

        system_info = self.logger.get_system_info()
        for key, value in system_info.items():
            self.logger.info(f"system_info - {key}: {value}")

        # get run_name
        if self.run_name is None:
            self.run_name = self.config['module']['module_name']

        # add unique datetime
        now = get_datetime()
        self.run_name += f"_{now}"
        self.local_run = self.local_scratch + '/' + self.run_name

        # create run directory
        if not os.path.isdir(self.local_run):
            os.makedirs(self.local_run)

        self.meta = {
            'system_info':      system_info,
            'now':              now,
            'run_name':         self.run_name,
            'config_file':      self.config_file,
            'run_directory':    self.local_run,
            'local_scratch':    self.local_scratch,
            'local_blip':       self.local_blip,
            'local_data':       self.local_data,
            'local_blip_files': self.local_blip_files,
            'local_data_files': self.local_data_files
        }
        self.logger.info(f'"now" set to: {now}')
        self.logger.info(f'"run_name" set to: {self.run_name}')
        self.logger.info(f'"run_directory" set to: {self.local_run}.')
        self.logger.info(f'"local_scratch" directory set to: {self.local_scratch}.')
        self.logger.info(f'"local_blip" directory set to: {self.local_blip}.')
        self.logger.info(f'"local_data" directory set to: {self.local_data}.')

        # set verbosity of self.logger
        if "verbose" in self.config["module"]:
            if not isinstance(self.config["module"]["verbose"], bool):
                self.logger.error(f'"module:verbose" must be of type bool, but got {type(self.config["module"]["verbose"])}!')
            self.meta["verbose"] = self.config["module"]["verbose"]
        else:
            self.meta["verbose"] = False

    def set_up_devices(self):
        # check for devices
        if "gpu" not in self.config["module"].keys():
            self.logger.warn('"module:gpu" not specified in config!')
            gpu = None
        else:
            gpu = self.config["module"]["gpu"]
        if "gpu_device" not in self.config["module"].keys():
            self.logger.warn('"module:gpu_device" not specified in config!')
            gpu_device = None
        else:
            gpu_device = self.config["module"]["gpu_device"]

        if torch.cuda.is_available():
            self.logger.info("CUDA is available with devices:")
            for ii in range(torch.cuda.device_count()):
                device_properties = torch.cuda.get_device_properties(ii)
                cuda_stats = f"name: {device_properties.name}, "
                cuda_stats += f"compute: {device_properties.major}.{device_properties.minor}, "
                cuda_stats += f"memory: {device_properties.total_memory}"
                self.logger.info(f" -- device: {ii} - " + cuda_stats)

        # set gpu settings
        if gpu:
            if torch.cuda.is_available():
                if gpu_device >= torch.cuda.device_count() or gpu_device < 0:
                    self.logger.warn(f"desired gpu_device '{gpu_device}' not available, using device '0'")
                    gpu_device = 0
                self.meta['device'] = torch.device(f"cuda:{gpu_device}")
                self.logger.info(
                    f"CUDA is available, using device {gpu_device}" +
                    f": {torch.cuda.get_device_name(gpu_device)}"
                )
            else:
                gpu = False
                self.logger.warn("CUDA not available! Using the cpu")
                self.meta['device'] = torch.device("cpu")
        else:
            self.logger.info("using cpu as device")
            self.meta['device'] = torch.device("cpu")
        self.meta['gpu'] = gpu

    def set_up_modules(self):
        # Configure the module handler
        self.logger.info("configuring modules.")
        module_config = self.config
        self.module_handler = ModuleHandler(
            self.run_name,
            module_config,
            meta=self.meta
        )

    def get_products(self):
        return self.meta, self.module_handler


def parse_command_line_config(
    params
):
    # set up local scratch directory
    if params.local_scratch is not None:
        if not os.path.isdir(params.local_scratch):
            if "LOCAL_SCRATCH" in os.environ:
                params.local_scratch = os.environ["LOCAL_SCRATCH"]
            else:
                params.local_scratch = './'
    else:
        if "LOCAL_SCRATCH" in os.environ:
            params.local_scratch = os.environ["LOCAL_SCRATCH"]
        else:
            params.local_scratch = './'

    # set up local blip directory
    if params.local_blip is not None:
        if not os.path.isdir(params.local_blip):
            if "LOCAL_BLIP" in os.environ:
                params.local_blip = os.environ["LOCAL_BLIP"]
            else:
                params.local_blip = './'
    else:
        if "LOCAL_BLIP" in os.environ:
            params.local_blip = os.environ["LOCAL_BLIP"]
        else:
            params.local_blip = './'

    # set up local data directory
    if params.local_data is not None:
        if not os.path.isdir(params.local_data):
            if "LOCAL_DATA" in os.environ:
                params.local_data = os.environ["LOCAL_DATA"]
            else:
                params.local_data = './'
    else:
        if "LOCAL_DATA" in os.environ:
            params.local_data = os.environ["LOCAL_DATA"]
        else:
            params.local_data = './'

    blip_runner = BlipRunner(
        params.config_file,
        params.name,
        params.local_scratch,
        params.local_blip,
        params.local_data,
        params.anomaly
    )
    return blip_runner.get_products()
