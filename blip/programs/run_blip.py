"""
Blip main program
"""
import argparse
import os
import sys
import subprocess
from mpi4py import MPI
import torch

from blip.utils import comm
from blip.utils.config import ConfigParser
from blip.blip.blip import Blip


def run():
    """
    This program runs the Blip module from a config file.
    It utilizes H5 + multi-gpu support to distribute the
    training of neural networks over multiple gpus which
    greatly speeds up runtime.  When used in conjunction
    with the "create_hyperparameter_runs" program,
    """

    """Check if the script is run interactively or as an sbatch job"""
    is_sbatch_job = 'SLURM_JOB_ID' in os.environ
    
    """Check if MPI is initialized"""
    mpi_initialized = MPI.Is_initialized()

    """If not running under MPI and not an sbatch job, re-execute with mpirun"""
    if not is_sbatch_job and not mpi_initialized:
        command = ['mpirun', '-np', '1', sys.executable] + sys.argv
        result = subprocess.run(command)
        sys.exit(result.returncode)

    """
    We do a preliminary check to ensure that MPI is available
    and that the number of processes is at least 1.  Assuming
    that Blip is being run on a system with more than one GPU.
    """
    try:
        _ = MPI.COMM_WORLD
    except Exception as exception:
        raise RuntimeError(f"error occurred with gathering MPI: {exception}")

    """Set up command line arguments"""
    parser = argparse.ArgumentParser(
        prog="Blip Module Runner",
        description="This program runs the Blip module from a config file. \
                    It utilizes H5 + multi-gpu support to distribute the \
                    training of neural networks over multiple gpus which \
                    greatly speeds up runtime.  When used in conjunction \
                    with the 'create_hyperparameter_runs' program,",
        epilog="...",
    )
    parser.add_argument(
        "config_file",
        metavar="<config_file>.yml",
        type=str,
        help="config file specification for this Blip module.",
    )
    parser.add_argument(
        "-run_name",
        dest="run_name",
        default=None,
        help='name for this run if wanting to override config (default "blip").',
    )
    parser.add_argument(
        "-run_num",
        dest="run_num",
        default=None,
        type=str,
        help='tag for indexing the current experiment if wanting to override config (default "00")'
    )
    parser.add_argument(
        "-model_parallel_sizes",
        dest="model_parallel_sizes",
        default=None,
        type=list,
        help='list of model parallel sizes if wanting to override config (default [1])'
    )
    parser.add_argument(
        "-model_parallel_names",
        dest="model_parallel_names",
        default=None,
        type=list,
        help='list of names for parallel models if wanting to override config (default ["model"])'
    )
    parser.add_argument(
        "-global_batch_size",
        dest="global_batch_size",
        default=None,
        type=int,
        help='global batch size if wanting to override config (default 16)'
    )
    parser.add_argument(
        "-amp_mode",
        dest="amp_mode",
        default=None,
        type=str,
        help='NVIDIA amp mode if wanting to override config (default "fp32")'
    )
    parser.add_argument(
        "-enable_jit",
        dest="enable_jit",
        default=None,
        type=bool,
        help='whether to enable JIT for model and loss if wanting to override config (default false)'
    )
    parser.add_argument(
        "-bucket_cap_mb",
        dest="bucket_cap_mb",
        default=None,
        type=int,
        help='bucket cap in Mb if wanting to override config (default 25)'
    )
    parser.add_argument(
        "-num_iterations",
        dest="num_iterations",
        default='10000',
        type=int,
        help='number of training batches if wanting to override config (default "10000")'
    )
    parser.add_argument(
        "-n",
        dest="number_of_files",
        default=None,
        help='number of files to process (default None).',
    )
    parser.add_argument(
        "-local_scratch",
        dest="local_scratch",
        default="/local_scratch",
        help='local scratch directory'
    )

    """Parse command line arguments"""
    try:
        args = parser.parse_args()
        config_file = args.config_file
        run_name = args.run_name
        run_num = args.run_num
        model_parallel_sizes = args.model_parallel_sizes
        model_parallel_names = args.model_parallel_names
        global_batch_size = args.global_batch_size
        amp_mode = args.amp_mode
        enable_jit = args.enable_jit
        bucket_cap_mb = args.bucket_cap_mb
        number_of_files = args.number_of_files
        local_scratch = args.local_scratch
        num_iterations = args.num_iterations
    except Exception as exception:
        raise RuntimeError(f"failed to parse command line arguments: {exception}")

    """Parse the config file"""
    try:
        config = ConfigParser(config_file).data
    except Exception as exception:
        raise RuntimeError(f"failed to parse config: {exception}")

    """Determine run_name and run_num"""
    if run_name is None:
        if "run_name" not in config["blip"]:
            run_name = "blip"
            config["blip"]["run_name"] = "blip"
        else:
            run_name = config["blip"]["run_name"]
    if run_num is None:
        if "run_num" not in config["blip"]:
            run_num = "00"
            config["blip"]["run_num"] = "00"
        else:
            run_num = config["blip"]["run_num"]

    """Determine model parallel attributes"""
    if model_parallel_sizes is None:
        if "model_parallel_sizes" not in config["blip"]:
            model_parallel_sizes = [1]
            config["blip"]["model_parallel_sizes"] = [1]
        else:
            model_parallel_sizes = config["blip"]["model_parallel_sizes"]
    if model_parallel_names is None:
        if "model_parallel_names" not in config["blip"]:
            model_parallel_names = ["model"]
            config["blip"]["model_parallel_names"] = ["model"]
        else:
            model_parallel_names = config["blip"]["model_parallel_names"]

    """Determine global batch size"""
    if global_batch_size is None:
        if "global_batch_size" not in config["blip"]:
            global_batch_size = 16
            config["blip"]["global_batch_size"] = 16
        else:
            global_batch_size = config["blip"]["global_batch_size"]

    """Determine NVIDIA amp mode"""
    if amp_mode is None:
        if "amp_mode" not in config["blip"]:
            amp_mode = "fp32"
            config["blip"]["amp_mode"] = "fp32"
        else:
            amp_mode = config["blip"]["amp_mode"]

    """Determine whether to use NVIDIA Amp with mixed precision"""
    amp_dtype = torch.float32
    if amp_mode in ["fp16", "bf16"]:
        amp_enabled = True
        if amp_mode == "fp16":
            amp_dtype = torch.float16
        elif amp_mode == "bf16":
            amp_dtype = torch.bfloat16
    else:
        amp_enabled = False

    """Determine enable JIT"""
    if enable_jit is None:
        if "enable_jit" not in config["blip"]:
            enable_jit = False
            config["blip"]["enable_jit"] = False
        else:
            enable_jit = config["blip"]["enable_jit"]

    """Determine global batch size"""
    if bucket_cap_mb is None:
        if "bucket_cap_mb" not in config["blip"]:
            bucket_cap_mb = 25
            config["blip"]["bucket_cap_mb"] = 25
        else:
            bucket_cap_mb = config["blip"]["bucket_cap_mb"]

    """Determine number of training batches"""
    if "num_iterations" not in config["blip"]:
        config["blip"]["num_iterations"] = num_iterations
    else:
        num_iterations = config["blip"]["num_iterations"]

    """Set up local scratch directory"""
    try:
        if not os.path.isdir(local_scratch):
            """Assuming now that we are running on NERSC"""
            new_local_scratch = f"/pscratch/sd/{os.environ['USER'][0]}/{os.environ['USER']}/"
            os.environ['LOCAL_SCRATCH'] = new_local_scratch
        else:
            if not os.path.isdir(local_scratch):
                raise RuntimeError(f'specified "local_scratch": "{local_scratch}" is not a directory!')
            if local_scratch[-1] != "/":
                local_scratch += "/"
            os.environ['LOCAL_SCRATCH'] = local_scratch
    except Exception as exception:
        raise RuntimeError(f"failed to set up local_scratch: {exception}")

    """Determine whether to only pick a subset of files"""
    if number_of_files is not None:
        try:
            number_of_files = int(number_of_files)
        except Exception as exception:
            raise RuntimeError(
                f"unable to convert number_of_files from {type(number_of_files)} to int: {exception}"
            )

        if isinstance(number_of_files, int):
            if number_of_files > 0:
                config["blip"]["number_of_files"] = number_of_files

    """Set up MPI variables"""
    try:
        config["blip"]["model_parallel_size"] = comm.init(config["blip"], verbose=True)
    except Exception as exception:
        raise RuntimeError(f"failed to run comm.init: {exception}")

    """Get info from MPI"""
    try:
        world_size = comm.get_world_size()
        world_rank = comm.get_world_rank()
        local_rank = comm.get_local_rank()
        distributed = (world_size > 1)
    except Exception as exception:
        raise RuntimeError(
            f"failed to get world_size, world_rank and local_rank variables: {exception}"
        )

    """Assert global batch size is divisible by number of GPUs"""
    if global_batch_size % comm.get_size("data") != 0:
        raise RuntimeError(f'cannot evenly distribute {global_batch_size} across {comm.get_size("data")} GPU(s)')

    """Determine the local data shards"""
    try:
        num_data_shards = comm.get_size("data")
        data_shard_id = comm.get_rank("data")
    except Exception as exception:
        raise RuntimeError(f"failed to get data shard information from utils.comm: {exception}")

    """Set up experiment directory"""
    try:
        experiment_directory = os.path.join(
            os.environ['LOCAL_SCRATCH'],
            f'{run_name}/{run_num}/'
        )
        if world_rank == 0:
            if not os.path.isdir(experiment_directory):
                os.makedirs(experiment_directory)
    except Exception as exception:
        raise RuntimeError(f"failed to construct experiment directory: {exception}")

    """Construct meta dictionary"""
    meta = {
        "run_name": run_name,
        "run_num": run_num,
        "num_iterations": num_iterations,
        "world_size": world_size,
        "world_rank": world_rank,
        "local_rank": local_rank,
        "distributed": distributed,
        "bucket_cap_mb": bucket_cap_mb,
        "enable_jit": enable_jit,
        "global_batch_size": global_batch_size,
        "amp_enabled": amp_enabled,
        "amp_dtype": amp_dtype,
        "num_data_shards": num_data_shards,
        "data_shard_id": data_shard_id,
        "config_file": config_file,
        'experiment_directory': os.path.abspath(experiment_directory),
        'local_scratch': os.environ['LOCAL_SCRATCH'],
    }

    """Create the Blip instance"""
    try:
        blip = Blip(config, meta)
    except Exception as exception:
        raise RuntimeError(f"failed to construct Blip object: {exception}")

    """Run Blip"""
    blip.run_blip()

    if distributed:
        torch.distributed.barrier()


if __name__ == "__main__":
    run()
