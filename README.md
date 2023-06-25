# Blip

[![Join the chat at https://gitter.im/Neutron-Calibration-in-DUNE/Blip](https://badges.gitter.im/Join%20Chat.svg)]([https://gitter.im/NESTCollaboration/nestpy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge](https://matrix.to/#/#neutron-calibration-in-dune.blip:gitter.im))
[![CodeFactor](https://www.codefactor.io/repository/github/neutron-calibration-in-dune/blip/badge)](https://www.codefactor.io/repository/github/neutron-calibration-in-dune/blip)
<!-- [![Pytest](https://github.com/NESTCollaboration/nestpy/actions/workflows/test_package.yml/badge.svg?branch=master)](https://github.com/NESTCollaboration/nestpy/actions/workflows/test_package.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1345595.svg)](https://doi.org/10.5281/zenodo.1345595) -->
[![PyPi version](https://pypip.in/v/duneblip/badge.png)](https://pypi.org/project/duneblip/)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
<!-- [![Python Versions](https://img.shields.io/pypi/pyversions/nestpy.svg)](https://pypi.python.org/pypi/nestpy)
[![PyPI downloads](https://img.shields.io/pypi/dm/nestpy.svg)](https://pypistats.org/packages/nestpy) -->

Blip is a collection of machine learning tools for reconstructing, classifying and analyzing low energy (< MeV) interactions in liquid argon time projection chambers (LArTPCs).  These interactions leave small point like signals (commonly referred to as "blips", hence the name). Blip is a python package which can be installed locally, or on the Wilson cluster, by following the directions below (eventually Blip will be available on the Wilson cluster without the need to install).

### Table of Contents

1. [ Getting the Repository ](#get)
2. [ Building Blip ](#build)
	* [ Environment YAML ](#yaml)
	* [ MinkowskiEngine ](#minkowski)
	* [ Blip ](#blip)
	* [ Installing on the Wilson Cluster ](#wilson)
3. [ Usage ](#usage)
	* [ Modules ](#modules)
	* [ Event Display ](#eventdisplay)
	* [ Using custom code with Blip ](#customcode)
4. [ Configuration Files ](#config)
5. [ Running Modules ](#runningmodules)
	* [ Machine Learning ](#ml)
	* [ Clustering/Manifold Learning ](#cluster)
	* [ Topological Data Analysis ](#tda)
	* [ Analysis ](#analysis)
6. [ Event Display ](#usingeventdisplay)
7. [ Versioning ](#versions)
8. [ Contact (Authors) ](#contact)
9. [ Citation ](#citation)
10. [ License ](#license)
11. [ Support ](#support)

<a name="get"></a>
## Getting the Repository

In the terminal, one can clone this repository by typing the command:

`git clone https://personal_username@github.com/Neutron-Calibration-in-DUNE/Blip.git`

This uses the HTTPS protocol. For environments (e.g. computing clusters) where one has to use the SSH protocol:

`git clone git@github.com:Neutron-Calibration-in-DUNE/Blip.git`

Anyone in the "Neutron-Calibration-in-DUNE" organization should be able to develop (push changes to the remote repository).

Please contact Nicholas Carrara or David Rivera about becoming involved in development before merging with the master branch. 

<a name="build"></a>
## Building Blip

<a name="yaml"></a>
### Environment YAML
The easiet way to install is to create a conda environment dedicated to the API using the packages defined in ``environment_blip.yml``:
```bash
conda env create -f environment_blip.yml
conda activate blip
```
You can optionally add the flag ``-n <name>`` to specify a name for the environment.

<a name="minkowski"></a>
### MinkowskiEngine
Due to the nature of the large datasets generated from LArTPC data, parts of Blip make use of SparseTensors in order to be more memory efficient, and to speed up overall performance.  SpraseTensors are handled through the *MinkowskiEngine* package, which interfaces with pytorch.
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

<a name="blip"></a>
### BLIP
From the main folder of Blip you can run:
```bash
pip install .
```
which should install the API for you.


<a name="wilson"></a>
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
<a name="usage"></a>
## Usage
Blip can be used in three different ways, 
   - I.   By running a set of pre-defined programs with a config file.
   - II.  With the event display through a browser or jupyter notebook.
   - III. Within your own code by importing/using blip modules.

<a name="modules"></a>
### Modules
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

<a name="eventdisplay"></a>
### Event Display

<a name="customcode"></a>
### Using Custom Code with Blip
Many of the classes in Blip are built from an abstract class with the prefix 'Generic'.  Any user can inherit from these classes and making sure to override the required functions.  These custom classes can then be loaded to Blip at runtime by specifying the python files in their appropriate config section.

<a name="config"></a>
## Configuration Files
   
<a name="runningmodules"></a>
## Running Modules

<a name="ml"></a>
### Machine Learning

<a name="cluster"></a>
### Clustering and Manifold Learning

<a name="tda"></a>
### Topological Data Analysis
   
<a name="analysis"></a>
### Analysis
   
<a name="usingeventdisplay"></a>
## Using the Event Display
   
<a name="versions"></a>
## Versioning
For the versions available, see the [tags on this repository](https://github.com/Neutron-Calibration-in-DUNE/Blip/tags). 
   
<a name="contact"></a>
## Contact (Authors)
If you have questions, please contact Nicholas Carrara, nmcarrara@ucdavis.edu.

See also the list of [contributors](https://github.com/orgs/Neutron-Calibration-in-DUNE/people) who participate in this project.

See AUTHORS.md for information on the developers.

<a name="support"></a>
## Support

* Bugs: Please report bugs to the [issue tracker on Github](https://github.com/Neutron-Calibration-in-DUNE/Blip/issues) such that we can keep track of them and eventually fix them.  Please explain how to reproduce the issue (including code) and which system you are running on.
* Help: Help can be provided also via the issue tracker by tagging your issue with 'question'
* Contributing:  Please fork this repository then make a pull request.  In this pull request, explain the details of your change and include tests.
   
<a name="citation"></a>
## Citation

When you use `blip`, please say so in your slides or publications (for publications, see Zenodo link above).  This is important for us being able to get funding to support this project.
