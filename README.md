# Blip

Blip is a collection of machine learning tools for reconstructing, classifying and analyzing low energy (< MeV) interactions in liquid argon time projection chambers (LArTPCs).  These interactions leave small point like signals (commonly referred to as "blips", hence the name). Blip is a python package which can be installed locally, or on the Wilson cluster, by following the directions below (eventually Blip will be available on the Wilson cluster without the need to install).

### Usage
Blip can be used in three different ways, 
   - I.   By running a set of pre-defined programs with a config file.
   - II.  With the event display through a browser or jupyter notebook.
   - III. Within your own code by importing/using blip modules.

#### Modules
There are several programs that will run different tasks such as; training a neural network, running a TDA or clustering algorithm, performing some analysis, etc.  Each of these tasks are specified by a *module_type* and a corresponding *module_mode*.  For example, to train a neural network one would set in the configuration file:
```yaml
# example module section
module:
  module_name:  'training_test'
  module_type:  'ml'            # ml, clustering, tda, analysis, ...
  module_mode:  'training'      # training, inference, parameter_scan, ...
  gpu:          True
  gpu_device:   0
```
#### Command line


##### Creating and using your own code
Many of the classes in Blip are built from an abstract class with the prefix 'Generic'.  Any user can inherit from these classes and making sure to override the required functions.  These custom classes can then be loaded to Blip at runtime by specifying the python files in their appropriate config section.


#### Event Display

### Installation

<!-- #### Conda/Pip
Assuming you have CUDA >= 11.8 installed, the easiest way to start from scratch is to use anaconda together with pip.  First, create a new anaconda environment using python version 3.9
```bash
conda create -n blip python=3.10
```
I've used the name *blip* for the anaconda environment.  Once this is set up, activate the environment,
```bash
conda activate blip
```
and then install torch.
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
Another large pytorch library we will need is pytorch geometric, which can be installed similarly with
```bash
conda install pyg -c pyg
```
Finally, we install several other dependecies,
```bash
conda install matplotlib pyyaml pandas seaborn
conda install -c nvidia cuda
pip install uproot
``` -->

#### Environment YAML
The easiet way to install is to create a conda environment dedicated to the API using the packages defined in ``environment_blip.yml``:
```bash
conda env create -f environment_blip.yml
conda activate blip
```
You can optionally add the flag ``-n <name>`` to specify a name for the environment.

#### MinkowskiEngine
With the libopenblas dependency, we can install MinkowskiEngine via the following
```bash
sudo apt-get install libopenblas-dev
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
```

You may need to switch to a different version of GCC in order to install CUDA.  To do this, switch to the older version with:
```bash
sudo apt -y install gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
```
You'll then need to select the alternative version
```bash
$ sudo update-alternatives --config gcc

There are 2 choices for the alternative gcc (providing /usr/bin/gcc).

  Selection    Path             Priority   Status
------------------------------------------------------------
* 0            /usr/bin/gcc-12   12        auto mode
  1            /usr/bin/gcc-11   11        manual mode
  2            /usr/bin/gcc-12   12        manual mode

Press <enter> to keep the current choice[*], or type selection number: 1
```

#### BLIP
From the main folder of Blip you can run:
```bash
pip install .
```
which should install the API for you.

#### Wilson Cluster
To install BLIP on the Wilson cluter at FNAL, we first need to set up our conda environment.  Due to the limited size of the home directory, we want to tell anaconda to download packages and install blip in a different directory.  Once logged in to the Wilson cluster, do the following to activate gnu8, openblas, cuda and condaforge
```bash
module load gnu8/8.3.0
module load openblas/0.3.7
module load cuda11/11.8.0
module load condaforge/py39
```

The default anaconda location for packages and environments is usually the home directory, which has limited space.  To check which directories are set, run
```bash
conda config --show
```
which should give an output like the following:
```bash
[<user_name>@wc:~:]$ conda config --show

...

envs_dirs:
  - /nashome/<first_letter>/<user_name>/.conda/envs

...

pkgs_dirs:
  - <old_package_directory>
...
```

Then, tell anaconda to use a different directory for downloading packages:
```bash
conda config --remove pkgs_dirs <old_package_directory>
conda config --remove envs_dirs <old_env_directory>
conda config --add pkgs_dirs <package_directory>
conda config --add envs_dirs <env_directory>
```
I've used */wclustre/dune/<username>* as the package/env directory.  Then, install blip using the YAML file:
```bash
conda env create --prefix <blip_install_directory> -f environment_blip.yml
```

I've also used */wclustre/dune/<username>/blip* as the install directory.  Once installed, blip can be activated with
```bash
conda activate <blip_install_directory>
```
In order to install MinkowskiEngine with CUDA, we need to set an environment variable which specifies the number of architectures that the current version of cuda will work with.  On a generic linux system, this can be achieved with a small script:
```bash
CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
if [[ ${CUDA_VERSION} == 9.0* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;7.0+PTX"
elif [[ ${CUDA_VERSION} == 9.2* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0+PTX"
elif [[ ${CUDA_VERSION} == 10.* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5+PTX"
elif [[ ${CUDA_VERSION} == 11.0* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0+PTX"
elif [[ ${CUDA_VERSION} == 11.* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
else
    echo "unsupported cuda version."
    exit 1
fi
```

For our purposes however, we are choosing cuda 11.8.0, so we can just run the command
```bash
export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
```

Then, install MinkowskiEngine
```bash
conda install openblas
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas" --install-option="--force_cuda"
```

### Basic Usage
---------------


### Datasets
------------

  <!-- One immediate way of addressing this is to come up with a common format for expressing datasets.  We can do this by imposing a set of constraints on how a dataset should be expressed in memory, which for now is done by creating a compressed numpy file (.npz) with the following minimal set of arrays:

```python

events = np.random.normal(0,1,1000) # random array of values
classes = np.ones((1000,1))
weights = np.ones((1000,1))
class_weights = np.ones((1000,1))

# dictionary containing meta data
event_meta = {
    "who_created":  "none",
    "when_created": "end_of_time",
    "where_created":"the_void",
    "num_events":   len(events), 
    "features":     {"x": 0},
    "classes":      {"y": 0},
    "sample_weights":{"w": 0},
    "class_weights":{"c": 0},
}

np.savez(
    "compressed_file.npz",
    meta=event_meta,
    event_features=features,
    event_classes=classes,
    event_sample_weights=weights,
    event_class_weights=class_weights,
)
```
Here we have a set of arrays containing **features** (events), **classes** (classes), **sample_weights** (weights) and **class_weights** (class_weights), as well as a dictionary **meta** which contains information about who/when/where the dataset was created, as well as the number of events and a set of dictionaries describing the various items in the arrays.  Everything but the **sample_weights** and **class_weights** items are required in the meta dictionary, which will be checked whenever a dataset is loaded from an .npz file. -->
