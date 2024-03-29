# Use use previous versions, modify these variables
# ARG PYTORCH="1.9.0"
# ARG CUDA="11.1"

ARG PYTORCH="2.0.0"
ARG CUDA="11.7"
ARG CUDNN="8"

# grab pytorch and anaconda3
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

##############################################
# You should modify this to match your GPU compute capability
# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install dependencies
RUN apt-get update
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# For faster build, use more jobs.
ENV MAX_JOBS=4
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
RUN cd MinkowskiEngine; python setup.py install --force_cuda --blas=openblas

RUN pip install torch_geometric matplotlib scipy scikit-learn uproot pandas seaborn bokeh imageio pytorch-metric-learning torchmetrics ripser torch-cluster torch-scatter h5py

RUN cd ..
RUN git clone --recursive "https://github.com/Neutron-Calibration-in-DUNE/Blip"
RUN cd Blip; pip install .

RUN cd ..
RUN git clone --recursive "https://github.com/Neutron-Calibration-in-DUNE/BlipModels"
