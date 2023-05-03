# Setup script for installing/running on the Wilson cluster.
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
export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0+PTX"
