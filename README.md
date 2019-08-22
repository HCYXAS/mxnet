<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

<div align="center">
  <a href="https://mxnet.incubator.apache.org/"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mxnet_logo_2.png"></a><br>
</div>

Apache MXNet (incubating) for Deep Learning
=====
| Master         | Docs          | License  |
| :-------------:|:-------------:|:--------:|
| [![Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/incubator-mxnet/job/master/badge/icon)](http://jenkins.mxnet-ci.amazon-ml.com/job/incubator-mxnet/job/master/)  | [![Documentation Status](http://jenkins.mxnet-ci.amazon-ml.com/job/restricted-website-build/badge/icon)](https://mxnet.incubator.apache.org/) | [![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE) |

![banner](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/banner.png)

Apache MXNet (incubating) is a deep learning framework designed for both *efficiency* and *flexibility*.
It allows you to ***mix*** [symbolic and imperative programming](https://mxnet.incubator.apache.org/architecture/index.html#deep-learning-system-design-concepts)
to ***maximize*** efficiency and productivity.
At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly.
A graph optimization layer on top of that makes symbolic execution fast and memory efficient.
MXNet is portable and lightweight, scaling effectively to multiple GPUs and multiple machines.

MXNet is more than a deep learning project. It is a collection of
[blue prints and guidelines](https://mxnet.incubator.apache.org/architecture/index.html#deep-learning-system-design-concepts) for building
deep learning systems, and interesting insights of DL systems for hackers.


## Installation Guide

### Prerequisites

[GCC 4.8](https://gcc.gnu.org/gcc-4.8/) or later to compile C++ 11. [GNU Make](https://www.gnu.org/software/make/)

### Install Dependencies to build mxnet for HIP/ROCm(AMD) : 

* Install ROCm following AMD ROCm's [Installation guide](https://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#installing-from-amd-rocm-repositories) to setup MXNet with GPU support

* Install ROCm Libraries

      sudo apt install -y rocm-device-libs rocm-libs rocblas hipblas rocrand rocfft

* Install ROCm opencl
       
       sudo apt install -y rocm-opencl rocm-opencl-dev

* Install MIOpen For acceleration

      sudo apt install -y miopengemm miopen-hip
      
* Install rocthrust , rocprim , hipcub Libraries
      
       sudo apt install -y rocthrust rocprim hipcub

  
### Install Dependencies to build mxnet for HIP/CUDA(Nvidia) :

* Install CUDA following the NVIDIA’s [installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/) to setup MXNet with GPU support

* Make sure to add CUDA install path to LD_LIBRARY_PATH
* Example - export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

* Install the dependencies [hipblas](https://github.com/ROCmSoftwarePlatform/hipBLAS/wiki/Build), [rocrand](https://github.com/ROCmSoftwarePlatform/rocRAND) from source.

## Build the MXNet library

* Step 1: Install build tools.
```bash
      sudo apt-get update
      sudo apt-get install -y build-essential
```

* Step 2: Install OpenBLAS. 
MXNet uses BLAS and LAPACK libraries for accelerated numerical computations on CPU machine. There are several flavors of  BLAS/LAPACK libraries - OpenBLAS, ATLAS and MKL. In this step we install OpenBLAS. You can choose to install ATLAS or MKL.
```bash
      sudo apt-get install -y libopenblas-dev liblapack-dev libomp-dev libatlas-dev libatlas-base-dev
```

* Step 3: Install OpenCV.
  Install [OpenCV](<https://opencv.org/>) here. MXNet uses OpenCV for efficient image loading and augmentation operations.
```bash
      sudo apt-get install -y libopencv-dev
```

* Step 4: Download MXNet sources and build MXNet core shared library.
```bash
      git clone --recursive https://github.com/ROCmSoftwarePlatform/mxnet.git
      cd mxnet
      export PATH=/opt/rocm/bin:$PATH
 ```
 
* Step 5:
    To compile on HCC PLATFORM(HIP/ROCm):
```bash
      export HIP_PLATFORM=hcc
```

   To compile on NVCC PLATFORM(HIP/CUDA):
```bash
      export HIP_PLATFORM=nvcc
```
* Step 6 :

    If building on CPU:
```bash
        make -j$nproc 
 ```
   If building on GPU:
```bash
       make -j$nproc USE_GPU=1 
```
  For MIOpen acceleration : 
  ```bash
       make -j$nproc USE_GPU=1 USE_CUDNN=1 
```
On succesfull compilation a library called libmxnet.so is created in mxnet/lib path.

**NOTE:**  USE_GPU (To build on GPU) , USE_CUDNN(for accelearation) flags can be changed in make/config.mk.
To compile on HIP/CUDA make sure to set USE_CUDA_PATH to right CUDA installation path in make/config.mk. In most cases it is - /usr/local/cuda.

### Install the MXNet Python binding

* Step 1: Install prerequisites - python, setup-tools, python-pip and numpy.
```bash
      sudo apt-get install -y python-dev python-setuptools python-numpy python-pip python-scipy
      sudo apt-get install python-tk
      sudo apt install -y fftw3 fftw3-dev pkg-config
```
* Step 2: Install the MXNet Python binding.
```bash
      cd python
      sudo python setup.py install
```    
* Step 3: Execute sample example
```bash
       cd example/
       cd bayesian-methods/
```
To run on gpu change mx.cpu() to mx.gpu() in python script (Example- bdk_demo.py)
```bash
       $ python bdk_demo.py
```

Ask Questions
-------------
* Please use [mxnet/issues](https://github.com/ROCmSoftwarePlatform/mxnet/issues) for reporting bugs.


examples
---------
* [Code Examples](https://github.com/ROCmSoftwarePlatform/mxnet/blob/master/example)

License
-------
Licensed under an [Apache-2.0](https://github.com/apache/incubator-mxnet/blob/master/LICENSE) license.

