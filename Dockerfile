FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

# Dependencies for Pymesh
RUN apt update -qq && apt install -q -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget

# Cmake 3.15 (Pymesh dependency)
RUN apt update && \
    apt install -y libncurses5-dev libncursesw5-dev libssl1.0.0 libssl-dev && \
    cd /workspace && \
    wget https://github.com/Kitware/CMake/releases/download/v3.15.2/cmake-3.15.2.tar.gz && \
    tar -xf cmake-3.15.2.tar.gz                     && \
    cd cmake-3.15.2/                                && \
    ./bootstrap                                     && \
    make -j                                         && \
    make install                                    && \
    cd ..

# Pymesh
RUN apt update && \
    apt install -y libgmp-dev libmpfr-dev libboost-dev libboost-thread-dev && \
    cd /workspace                                   && \
    git clone https://github.com/PyMesh/PyMesh.git  && \
    cd PyMesh                                       && \
    git submodule update --init                     && \
    ./setup.py build                                && \
    ./setup.py install

RUN conda install torchvision==0.3 -c pytorch

# Softras
RUN conda install scikit-image==0.15.0
RUN cd /workspace                                   && \
    git clone https://github.com/ShichenLiu/SoftRas.git  && \
    cd SoftRas                                       && \
    python setup.py install

# NMR
## for CUDA9, `pip install neural_renderer_pytorch` 
## for CUDA 10, install from source as follows
RUN pip install cupy-cuda100 && \
    cd /workspace                                   && \
    git clone https://github.com/daniilidis-group/neural_renderer/ && \
    cd neural_renderer && \
    python setup.py build && \
    python setup.py install

# Python dependencies
RUN pip install absl-py tensorflow==2.0.0 tensorboard==2.0.1 tensorboardx==1.9 \
        opencv-python==4.1.0.25 dotmap scipy visdom dominate meshzoo==0.4.3 moviepy==1.0.1 chainer ipdb

RUN apt update -qq \
    && apt install -qy libglib2.0-0 \
    && apt install -y openssh-server
