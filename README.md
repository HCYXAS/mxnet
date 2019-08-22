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

* Install MIOpen For acceleation

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

* Step 6: To enable MIOpen for higher acceleration :
 
      USE_CUDNN=1

* Step 7: 

    If building on CPU:
```bash
        make -jn(n=number of cores) USE_GPU=0 (For Ubuntu 16.04)
        make -jn(n=number of cores)  CXX=g++-6 USE_GPU=0 (For Ubuntu 18.04)
```

   If building on GPU:
```bash
       make -jn(n=number of cores) USE_GPU=1 (For Ubuntu 16.04)
       make -jn(n=number of cores)  CXX=g++-6 USE_GPU=1 (For Ubuntu 18.04)
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


Contents
--------
* [Documentation](https://mxnet.incubator.apache.org/) and  [Tutorials](https://mxnet.incubator.apache.org/tutorials/)
* [Design Notes](https://mxnet.incubator.apache.org/architecture/index.html)
* [Code Examples](https://github.com/apache/incubator-mxnet/tree/master/example)
* [Installation](https://mxnet.incubator.apache.org/install/index.html)
* [Pretrained Models](http://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html)

Features
--------
* Design notes providing useful insights that can re-used by other DL projects
* Flexible configuration for arbitrary computation graph
* Mix and match imperative and symbolic programming to maximize flexibility and efficiency
* Lightweight, memory efficient and portable to smart devices
* Scales up to multi GPUs and distributed setting with auto parallelism
* Support for [Python](https://github.com/apache/incubator-mxnet/tree/master/python), [Scala](https://github.com/apache/incubator-mxnet/tree/master/scala-package), [C++](https://github.com/apache/incubator-mxnet/tree/master/cpp-package), [Java](https://github.com/apache/incubator-mxnet/tree/master/scala-package), [Clojure](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package), [R](https://github.com/apache/incubator-mxnet/tree/master/R-package), [Go](https://github.com/jdeng/gomxnet/), [Javascript](https://github.com/dmlc/mxnet.js/), [Perl](https://github.com/apache/incubator-mxnet/tree/master/perl-package), [Matlab](https://github.com/apache/incubator-mxnet/tree/master/matlab), and [Julia](https://github.com/apache/incubator-mxnet/tree/master/julia)
* Cloud-friendly and directly compatible with S3, HDFS, and Azure

License
-------
Licensed under an [Apache-2.0](https://github.com/apache/incubator-mxnet/blob/master/LICENSE) license.

Reference Paper
---------------

Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao,
Bing Xu, Chiyuan Zhang, and Zheng Zhang.
[MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems](https://github.com/dmlc/web-data/raw/master/mxnet/paper/mxnet-learningsys.pdf).
In Neural Information Processing Systems, Workshop on Machine Learning Systems, 2015

History
-------
MXNet emerged from a collaboration by the authors of [cxxnet](https://github.com/dmlc/cxxnet), [minerva](https://github.com/dmlc/minerva), and [purine2](https://github.com/purine/purine2). The project reflects what we have learned from the past projects. MXNet combines aspects of each of these projects to achieve flexibility, speed, and memory efficiency.
