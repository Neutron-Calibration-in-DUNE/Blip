.. Blip documentation master file, created by
   sphinx-quickstart on Thu Jul 20 10:40:50 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================================
WELCOME to BLIP !
================================

Blip is a collection of machine learning tools for reconstructing, classifying and analyzing low energy (< MeV) interactions in liquid argon time projection chambers (LArTPCs).  
These interactions leave small point like signals (commonly referred to as "blips", hence the name). 
Blip is a python package which can be installed locally, or on the Wilson Cluster (WC), or on Perlmutter at NERSC, by following the directions below (eventually Blip will be available on the Wilson cluster without the need to install).

---------------------------------------------------------------------------------------------------------------------------------------------

**CONTENTS**
============
.. toctree::
   :maxdepth: 3

   0.0.GettingStarted
   1.0.Blip
   2.0.Dataset
   3.0.Arrakis
   4.0.ArrakisND
   5.0.Examples

---------------------------------------------------------------------------------------------------------------------------------------------

**SUMMARY**
------------
Independently of the installation method you choose, you will need to clone the repository:

0.- Clone the repository:

.. code-block:: bash

   git clone https://github.com/Neutron-Calibration-in-DUNE/Blip.git 
 

Then, depending on the installation method, follow the next steps:

.. admonition:: **Local installation**

   1.- Create a conda enviroment:

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

.. admonition:: **Wilson Cluster installation**

   1.- Load the modules:

   .. code-block:: bash
      
      module load gnu8/8.3.0
      module load openblas/0.3.7
      module load cuda11/11.8.0
      module load condaforge/py39

   2.- Create a conda enviroment:

   * Configure the paths:

   .. code-block:: bash

      conda config --show #check the variables envs_dirs + pkgs_dirs
      conda config --add pkgs_dirs <package_directory>
      conda config --add envs_dirs <enviroment_directory>

   If you have mistaken your path you can: ``conda config --remove envs_dirs <old_env_directory>``

   * Create the enviroment:

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
