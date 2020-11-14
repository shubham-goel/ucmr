# Installation
You can either use the dockerfile [here](../Dockerfile) or follow the instructions here.

First, recurively clone the git repo and it's linked dependencies:
```
git clone --recursive git@github.com:shubham-goel/ucmr.git
```

Next, create a conda environment and install python dependencies
```
conda create -n ucmr python=3.7 anaconda pytorch=1.1 torchvision=0.3 cudatoolkit=10.0 -c pytorch
conda activate ucmr
pip install absl-py tensorflow tensorboard tensorboardX opencv-python==4.1.0.25 dotmap dominate meshzoo==0.4.3 moviepy visdom chainer
```

## Other Dependencies
The following additional dependencies need to be installed: [PyMesh](https://github.com/PyMesh/PyMesh), [SoftRas](https://github.com/ShichenLiu/SoftRas) and [NMR](https://github.com/daniilidis-group/neural_renderer). For convenience, here's how you can install them:

### PyMesh
Note that building pymesh requires [CMake1.15](https://github.com/Kitware/CMake)
```bash
apt install -y libgmp-dev libmpfr-dev libboost-dev libboost-thread-dev
git clone --recursive https://github.com/PyMesh/PyMesh.git
cd PyMesh
./setup.py build
./setup.py install
```

### SoftRas
```bash
git clone https://github.com/ShichenLiu/SoftRas.git
cd SoftRas
python setup.py install
```

### NMR
```bash
# Install using pip if CUDA9
pip install neural_renderer_pytorch

# Install from source if CUDA10
pip install cupy-cuda100
git clone https://github.com/daniilidis-group/neural_renderer/
cd neural_renderer
python setup.py build
python setup.py install
```
