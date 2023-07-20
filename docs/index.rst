.. Blip documentation master file, created by
   sphinx-quickstart on Thu Jul 20 10:40:50 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================================
Welcome to Blip's documentation!
================================


Contents
==================
.. toctree::
   :maxdepth: 2

   Introduction
   Usage
   Examples

Summary (local installation)
------------------------------

For a quick summary or just as a reminder follow the next steps:

1.- Clone the repository:

.. code-block:: bash

   git clone https://github.com/Neutron-Calibration-in-DUNE/Blip.git 

2.- Create a conda enviroment:

.. code-block:: bash

   conda env create -f environment_blip.yml
   conda activate blip

3.- Install MinkoskiEngine:

.. code-block:: bash

   sudo apt-get install libopenblas-dev
   pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

4.- You are done! 🎉 (Now from the main ``Blip`` folder you need to run):

.. code-block:: bash

   pip install .

Summary (wilson cluster installation)
--------------------------------------

1.- Clone the repository and load the modules:

.. code-block:: bash
   
   git clone https://github.com/Neutron-Calibration-in-DUNE/Blip.git 
   
   module load gnu8/8.3.0
   module load openblas/0.3.7
   module load cuda11/11.8.0
   module load condaforge/py39

2.- Create a conda enviroment:

Configure the paths:

.. code-block:: bash

   conda config --show #check the variables envs_dirs + pkgs_dirs
   conda config --add pkgs_dirs <package_directory>
   conda config --add envs_dirs <enviroment_directory>

If you have mistaken your path you can: ``conda config --remove envs_dirs <old_env_directory>``

Create the enviroment:

.. code-block:: bash
   
   cd Blip/
   conda env create --prefix /wclustre/davis_nc/USER/ -f environment_blip.yml
   conda activate /wclustre/davis_nc/USER/

3.- Install MinkoskiEngine:

.. code-block:: bash

   export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
   conda install openblas
   pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas" --install-option="--force_cuda"



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
