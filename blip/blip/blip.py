"""
"""
import os
import sys
import torch
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from mpi4py import MPI
import traceback
from matplotlib import pyplot as plt
from importlib.metadata import version
import pynvml

from blip.dataset.blip_dataset import BlipDataset
from blip.dataset.blip_loader import BlipLoader
from blip.models.blip_model import BlipModel
from blip.models.common import init_ddp_model_and_reduction_hooks
from blip.optimizers.blip_optimizer import BlipOptimizer
from blip.losses.blip_loss import BlipLoss
from blip.metrics.blip_metric import BlipMetric
from blip.trainer.blip_trainer import BlipTrainer
from blip.utils.logger import Logger
from blip.utils.utils import (
    profiler,
    timing_manager,
    memory_manager,
    fig_to_array
)
from blip.utils.common import sync_params
from blip.utils import comm


class Blip:
    """
    Main Blip class for running jobs. Blip is designed
    to work with multi-gpus and H5.
    """
    @profiler
    def __init__(
        self,
        config: dict = {},
        meta:   dict = {},
    ):
        """_summary_

        Args:
            config (dict): config file for running Blip.
            meta (dict): dictionary of meta information to
            be shared across all nodes.
        """

        """Get mpi communication parameters"""
        self.config = config
        self.meta = meta

        try:
            self.comm = MPI.COMM_WORLD
        except Exception as exception:
            raise RuntimeError(f"unable to obtain MPI parameters: {exception}")

        """Set input parameters and set up loggers"""
        try:
            self.logger = Logger(meta=self.meta)
        except Exception as exception:
            raise RuntimeError(f"unable to set up logging system: {exception}")

        """Setting error status for this node"""
        self.error_status = None
        self.exc_type = None
        self.exc_value = None
        self.exc_traceback = None
        self.line_number = None
        self.file_name = None
        self.tb_str = None
        self.traceback_details = None
        self.event_errors = []
        self.plugin_errors = []
        self.event_plugin_errors = []
        self.event_plugin_exc_types = []
        self.event_plugin_exc_values = []
        self.event_plugin_exc_tracebacks = []
        self.event_plugin_line_numbers = []
        self.event_plugin_file_names = []
        self.event_plugin_tb_strs = []
        self.event_plugin_traceback_details = []

        time = datetime.now()
        self.now = f"{time.hour}:{time.minute}:{time.second} [{time.day}/{time.month}/{time.year}]"

        """Parse config"""
        try:
            self.parse_config()
        except Exception as exception:
            self.report_error(
                exception=exception,
                message='error parsing config'
            )

    @profiler
    def parse_config(
        self,
    ):
        """Startup main Blip program"""
        if self.meta["local_rank"] == 0:
            self.logger.info(
                f'############################ BLIP  v. [{version("blip")}] ############################'
            )

        """Check for main blip parameters"""
        if "blip" not in self.config.keys():
            self.report_error(
                exception=KeyError,
                message="blip section not in config!"
            )
        blip_config = self.config["blip"]
        if blip_config is None:
            blip_config = {}

        """Try to grab system info and display to the logger"""
        if self.meta["local_rank"] == 0:
            system_info = self.logger.get_system_info()
            self.logger.info(f'system_info - local time: {self.now}')
            for key, value in system_info.items():
                self.logger.info(f"system_info - {key}: {value}")

        """Set the verbosity of the Blip program"""
        if "verbose" in blip_config:
            if not isinstance(blip_config["verbose"], bool):
                self.logger.error(
                    f'"blip:verbose" must be of type bool, but got {type(blip_config["verbose"])}!'
                )
            self.meta["verbose"] = blip_config["verbose"]
        else:
            self.meta["verbose"] = False

        """See if any CUDA devices are available on the system"""
        if self.meta["local_rank"] == 0:
            if torch.cuda.is_available():
                self.logger.info("CUDA is available with devices:")
                for ii in range(torch.cuda.device_count()):
                    device_properties = torch.cuda.get_device_properties(ii)
                    self.logger.info(f" -- device: {ii}")
                    self.logger.info(f" {' ':<{5}} name: {device_properties.name}")
                    self.logger.info(f" {' ':<{5}} compute: {device_properties.major}.{device_properties.minor}")
                    self.logger.info(f" {' ':<{5}} memory: {round(device_properties.total_memory / (1024**3))} GB")

    @profiler
    def report_error(
        self,
        exception: Exception = None,
        message: str = None,
    ):
        exc_type, exc_value, exc_traceback = sys.exc_info()
        """Extracting the line number from the traceback"""
        line_number = exc_traceback.tb_lineno
        file_name = exc_traceback.tb_frame.f_code.co_filename
        """Optionally, use traceback to format a string of the entire traceback"""
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        traceback_details = "".join(tb_str)
        if message:
            self.logger.critical(message)
        else:
            self.logger.critical("error encountered in blip")
        self.logger.critical(f"exception:   {exception}")
        self.logger.critical(f"exc_type:    {exc_type}")
        self.logger.critical(f"line_number: {line_number}")
        self.logger.critical(f"file_name:   {file_name}")
        self.logger.critical(f"traceback:   {traceback_details}")
        exit(1)

    @profiler
    def get_cuda_usage(
        self,
    ):
        try:
            if self.meta["world_rank"] == 0:
                all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(
                    self.meta["nvml_handle"]
                ).used / (1024. * 1024. * 1024.)
        except Exception as exception:
            self.report_error(
                exception=exception,
                message="error collecting memory information"
            )
        return all_mem_gb

    @profiler
    def run_end_of_blip(self):
        """
        Set of functions to be run at the end of the entire
        Blip job.  Some default operations are to create
        profiling plots.
        """
        self.generate_timing_and_memory_plots()
        try:
            self.logger.info("Blip program ran successfully. Closing out.")
        except Exception as exception:
            self.report_error(exception=exception, message="error at run_end_of_file")

    @profiler
    def generate_timing_and_memory_plots(self):
        """Generate timing and memory plots"""
        try:
            timings = timing_manager.timings
            memory = memory_manager.memory

            timing_averages = {}
            timing_stds = {}
            for item in timings.keys():
                timing_averages[item] = np.mean(timings[item])
                timing_stds[item] = np.std(timings[item])

            """Plot timings from blip timings"""
            blip_fig, blip_axs = plt.subplots(figsize=(15, 10))
            blip_box_values = []
            blip_labels = []
            for item in timings.keys():
                blip_box_values.append(timings[item])
                blip_axs.plot(
                    [],
                    [],
                    marker="",
                    linestyle="-",
                    label=f'{item}\n({timing_averages[item]:.2f} +/- {timing_stds[item]:.2f})',
                )
                blip_labels.append(item)
            blip_axs.boxplot(blip_box_values, vert=True, patch_artist=True, labels=blip_labels)
            blip_axs.set_ylabel(r"$\langle\Delta t\rangle$ (ms)")
            blip_axs.set_xticklabels(blip_labels, rotation=45, ha="right")
            blip_axs.set_yscale("log")
            plt.title(r"blip $\langle\Delta t\rangle$ (ms) vs. function")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
            plt.tight_layout()
            plt.savefig(f"/{os.environ['LOCAL_SCRATCH']}/blip_timing_avg.png")
            fig_array = fig_to_array(blip_fig)
            self.meta['tensorboard'].add_image(
                'blip_timing',
                fig_array,
                self.meta['num_epochs'],
                dataformats='HWC'
            )

            """Plot memory from blip memory"""
            memory_averages = {}
            memory_stds = {}
            for item in memory.keys():
                memory_averages[item] = np.mean(memory[item])
                memory_stds[item] = np.std(memory[item])

            blip_fig, blip_axs = plt.subplots(figsize=(15, 10))
            blip_box_values = []
            blip_labels = []
            for item in memory.keys():
                blip_box_values.append(memory[item])
                blip_axs.plot(
                    [],
                    [],
                    marker="",
                    linestyle="-",
                    label=f'{item}\n({memory_averages[item]:.2f} +/- {memory_stds[item]:.2f})',
                )
                blip_labels.append(item)
            blip_axs.boxplot(blip_box_values, vert=True, patch_artist=True, labels=blip_labels)
            blip_axs.set_ylabel(r"$\langle\Delta m\rangle$ (Mb)")
            blip_axs.set_xticklabels(blip_labels, rotation=45, ha="right")
            blip_axs.set_yscale("log")
            plt.title(r"blip $\langle\Delta m\rangle$ (Mb) vs. function")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
            plt.tight_layout()
            plt.savefig(f"/{os.environ['LOCAL_SCRATCH']}/blip_memory_avg.png")
            fig_array = fig_to_array(blip_fig)
            self.meta['tensorboard'].add_image(
                'blip_memory',
                fig_array,
                self.meta['num_epochs'],
                dataformats='HWC'
            )
        except Exception as exception:
            self.report_error(exception=exception, message="error generating timing/memory plots")

    @profiler
    def set_up_device(
        self,
    ):
        if self.meta["world_rank"] == 0:
            self.logger.info("setting up devices")
        try:
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(self.meta["local_rank"])

            """Set gpu_device to rank"""
            self.meta["device"] = torch.device(f'cuda:{self.meta["local_rank"]}')

            """Initialize pynvml"""
            pynvml.nvmlInit()
            self.meta["nvml_handle"] = pynvml.nvmlDeviceGetHandleByIndex(self.meta["device"].index)
        except Exception as exception:
            self.report_error(
                exception=exception,
                message='error in setting up devices'
            )
        if self.meta["world_rank"] == 0:
            mem_usage = self.get_cuda_usage()
            self.logger.info(f'device allocated {mem_usage:.2f} Gb')

    @profiler
    def set_up_dataset(
        self,
    ):
        if self.meta["world_rank"] == 0:
            self.logger.info("setting up dataset")
        if "dataset" not in self.config.keys():
            self.report_error(
                exception=KeyError,
                message="'dataset' not specified in config!"
            )
        try:
            self.meta['dataset'] = BlipDataset(
                self.config['dataset'],
                self.meta
            )
        except Exception as exception:
            self.report_error(
                exception=exception,
                message="error creating BlipDataset"
            )

    @profiler
    def set_up_loader(
        self,
    ):
        if self.meta["world_rank"] == 0:
            self.logger.info("setting up loader")
        if "loader" not in self.config.keys():
            self.report_error(
                exception=KeyError,
                message="'loader' not specified in config!"
            )
        try:
            self.meta['loader'] = BlipLoader(
                self.config['loader'],
                self.meta
            )
        except Exception as exception:
            self.report_error(
                exception=exception,
                message="error creating DataLoader"
            )

    @profiler
    def set_up_model(
        self,
    ):
        if self.meta["world_rank"] == 0:
            self.logger.info("setting up model")
            starting_mem = self.get_cuda_usage()
        if "model" not in self.config.keys():
            self.report_error(
                exception=KeyError,
                message="'model' not specified in config!"
            )
        try:
            self.meta['model'] = BlipModel(
                self.config['model'],
                self.meta
            )
        except Exception as exception:
            self.report_error(
                exception=exception,
                message="error creating BlipModel"
            )
        try:
            self.meta['model'].model.to(self.meta["device"])
        except Exception as exception:
            self.report_error(
                exception=exception,
                message="error setting model to device"
            )

        """Compile model if jit is enabled"""
        try:
            if self.meta["enable_jit"]:
                self.meta['model'].model = torch.compile(self.meta['model'].model)
        except Exception as exception:
            self.report_error(
                exception=exception,
                message="error compiling model with jit"
            )

        """Sync parameters across model if its split on gpus"""
        try:
            if comm.get_size("model") > 1:
                sync_params(self.meta['model'].model)
        except Exception as exception:
            self.report_error(
                exception=exception,
                message="error running sync_params on model"
            )

        """Set up DistributedDataParallel if enabled"""
        try:
            if self.meta["distributed"]:
                self.meta['model'].model = init_ddp_model_and_reduction_hooks(
                    self.meta['model'].model,
                    device_ids=[self.meta["local_rank"]],
                    output_device=[self.meta["local_rank"]],
                    bucket_cap_mb=self.meta["bucket_cap_mb"]
                )
        except Exception as exception:
            self.report_error(
                exception=exception,
                message="error setting up distributed data parallel with model"
            )

        """Report model memory usage"""
        if self.meta["world_rank"] == 0:
            ending_mem = self.get_cuda_usage()
            self.logger.info(f'model allocated {(ending_mem - starting_mem):.2f} Gb')

    @profiler
    def set_up_scaler(
        self,
    ):
        if self.meta["world_rank"] == 0:
            self.logger.info("setting up scaler")
        try:
            if self.meta["amp_dtype"] == torch.float16:
                self.meta["scaler"] = GradScaler()
            else:
                self.meta["scaler"] = None
        except Exception as exception:
            self.report_error(
                exception=exception,
                message="failed to construct scaler"
            )

    @profiler
    def set_up_optimizer(
        self,
    ):
        if self.meta["world_rank"] == 0:
            self.logger.info("setting up optimizer")
        if "optimizer" not in self.config.keys():
            self.report_error(
                exception=KeyError,
                message="'optimizer' not specified in config!"
            )
        try:
            self.meta['optimizer'] = BlipOptimizer(
                self.config['optimizer'],
                self.meta
            )
        except Exception as exception:
            self.report_error(
                exception=exception,
                message="failed to construct optimizer"
            )

    @profiler
    def set_up_scheduler(
        self,
    ):
        if self.meta["world_rank"] == 0:
            self.logger.info("setting up scheduler")
        if "scheduler" not in self.config.keys():
            self.meta['scheduler'] = None
        else:
            if 'lr_schedule' not in self.config['scheduler']:
                self.meta['scheduler'] = None
            else:
                try:
                    if self.config['scheduler']['lr_schedule'] == 'cosine':
                        if "warmup" not in self.config['scheduler']:
                            self.config['scheduler']["warmup"] = 1
                        if self.config['scheduler']['warmup'] > 0:
                            lr_scale = lambda x: min(
                                (x + 1) / self.config['scheduler']['warmup'],
                                0.5 * (1 + np.cos(np.pi * x / self.meta['num_iterations']))
                            )
                            self.meta['scheduler'] = torch.optim.lr_scheduler.LambdaLR(
                                self.meta['optimizer'].optimizer,
                                lr_scale
                            )
                        else:
                            self.meta['scheduler'] = torch.optim.lr_scheduler.CosineAnnealingLR(
                                self.meta['optimizer'].optimizer,
                                self.meta['num_iterations']
                            )
                    else:
                        self.meta['scheduler'] = None
                except Exception as exception:
                    self.report_error(
                        exception=exception,
                        message="failed to construct scheduler"
                    )

    @profiler
    def set_up_criterion(
        self,
    ):
        if self.meta["world_rank"] == 0:
            self.logger.info("setting up criterion")
        if "criterion" not in self.config.keys():
            self.report_error(
                exception=KeyError,
                message="'criterion' not specified in config!"
            )
        try:
            self.meta['criterion'] = BlipLoss(
                self.config['criterion'],
                self.meta
            )
        except Exception as exception:
            self.report_error(
                exception=exception,
                message="failed to construct BlipLoss"
            )

    @profiler
    def set_up_metrics(
        self,
    ):
        if self.meta["world_rank"] == 0:
            self.logger.info("setting up metrics")
        if "metrics" not in self.config.keys():
            self.report_error(
                exception=KeyError,
                message="'metrics' not specified in config!"
            )
        try:
            self.meta['metrics'] = BlipMetric(
                self.config['metrics'],
                self.meta
            )
        except Exception as exception:
            self.report_error(
                exception=exception,
                message="failed to construct BlipLoss"
            )

    @profiler
    def set_up_trainer(
        self,
    ):
        if self.meta["world_rank"] == 0:
            self.logger.info("setting up trainer")
        if "trainer" not in self.config.keys():
            self.report_error(
                exception=KeyError,
                message="'trainer' not specified in config!"
            )
        try:
            self.meta['trainer'] = BlipTrainer(
                self.config['trainer'],
                self.meta
            )
        except Exception as exception:
            self.report_error(
                exception=exception,
                message="failed to construct trainer"
            )

    @profiler
    def set_up_tensorboard(
        self,
    ):
        if self.meta["world_rank"] == 0:
            self.logger.info("setting up tensorboard")
        self.meta['tensorboard'] = SummaryWriter(
            log_dir=os.path.join(
                self.meta["experiment_directory"], "logs/", self.now
            )
        )

    @profiler
    def run_training(
        self,
    ):
        if self.meta["world_rank"] == 0:
            self.logger.info("running training")
        self.meta['trainer'].train()

    @profiler
    def run_blip(self):
        """
        Main Blip program loop.
        """
        """Set device and benchmarks"""
        try:
            self.set_up_device()
        except Exception as exception:
            self.report_error(exception=exception)

        """Set up dataset"""
        try:
            self.set_up_dataset()
        except Exception as exception:
            self.report_error(exception=exception)

        """Set up loader"""
        try:
            self.set_up_loader()
        except Exception as exception:
            self.report_error(exception=exception)

        """Set up model"""
        try:
            self.set_up_model()
        except Exception as exception:
            self.report_error(exception=exception)

        """Set up scaler"""
        try:
            self.set_up_scaler()
        except Exception as exception:
            self.report_error(exception=exception)

        """Set up optimizer"""
        try:
            self.set_up_optimizer()
        except Exception as exception:
            self.report_error(exception=exception)

        """Set up scheduler"""
        try:
            self.set_up_scheduler()
        except Exception as exception:
            self.report_error(exception=exception)

        """Set up criterion"""
        try:
            self.set_up_criterion()
        except Exception as exception:
            self.report_error(exception=exception)

        """Set up metrics"""
        try:
            self.set_up_metrics()
        except Exception as exception:
            self.report_error(exception=exception)

        """Set up trainer"""
        try:
            self.set_up_trainer()
        except Exception as exception:
            self.report_error(exception=exception)

        """Set up tensorboard"""
        try:
            self.set_up_tensorboard()
        except Exception as exception:
            self.report_error(exception=exception)

        """Run training"""
        self.run_training()

        """Run end of blip"""
        self.run_end_of_blip()
