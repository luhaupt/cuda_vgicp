# CUDA_VGICP

## Getting started

1. Follow the instructions for installing the [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
2. Add necessary libraries and executables to global path (i.e. append ~/.bashrc), where `<v>` is the current CUDA version.
```bash
export PATH=/usr/local/cuda-<v>/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-<v>/lib64:${LD_LIBRARY_PATH}
```
3. Clone and compile this project. `-arch=sm_<a>` needs a _compute capability_ where `<a>` can be taken from [this list](https://developer.nvidia.com/cuda-gpus) accordingly.
```bash
# The flag "-arch=sm_89" is for Ada Lovelace GPUs corresponing to the RTX 4000 series.
nvcc -O3 -arch=sm_89 main.cu -o target
```
