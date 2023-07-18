#!/bin/sh
# Setup script for installing/running on the Wilson cluster.
#---------------------Directory---------------------#
# this handy piece of code determines the relative
# directory that this script is in.
SOURCE="${BASH_SOURCE[0]}"
# resolve $SOURCE until the file is no longer a symlink
while [ -h "$SOURCE" ]; do 
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  # if $SOURCE was a relative symlink, we need to resolve it relative 
  # to the path where the symlink file was located
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" 
done
LOCAL_BLIP_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )/../"

# You'll need to specify some directories for the anaconda 
# environment, and where you want the BLIP environment stored.
export ANACONDA_ENV_DIR=/wclustre/dune/$USER
export ANACONDA_PKG_DIR=/wclustre/dune/$USER
export BLIP_ENV_DIR=/wclustre/dune/$USER/blip

# Currently, four modules are required for installation,
# gnu, openblas, cuda and condaforge.  The versions which
# work are listed below

GNU_VERSION=gnu8/8.3.0
OPENBLAS_VERSION=openblas/0.3.7
CUDA_VERSION=cuda11/11.8.0
CONDAFORGE_VERSION=condaforge/py39

# First, load the modules
module load $GNU_VERSION
module load $OPENBLAS_VERSION
module load $CUDA_VERSION
module load $CONDAFORGE_VERSION

# For installing MinkowskiEngine, we need to set
# the cuda arch list, otherwise it will fail.
export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Set up conda directories
conda config --add pkgs_dirs $ANACONDA_PKG_DIR
conda config --add envs_dirs $ANACONDA_ENV_DIR

# Install BLIP
conda env create --prefix $BLIP_ENV_DIR -f $LOCAL_BLIP_DIR/environment_blip.yml
conda activate $BLIP_ENV_DIR
conda install openblas
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas" --install-option="--force_cuda"
