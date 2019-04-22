<img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/mxnet2.png width=135/> *for Deep Learning*
=====

[![Build Status](https://travis-ci.org/dmlc/mxnet.svg?branch=master)](https://travis-ci.org/dmlc/mxnet)
[![Documentation Status](https://readthedocs.org/projects/mxnet/badge/?version=latest)](http://mxnet.io/)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

![banner](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/banner.png)

MXNet is a deep learning framework designed for both *efficiency* and *flexibility*.
It allows you to ***mix*** [symbolic and imperative programming](http://mxnet.io/architecture/index.html#deep-learning-system-design-concepts)
to ***maximize*** efficiency and productivity.
At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly.
A graph optimization layer on top of that makes symbolic execution fast and memory efficient.
MXNet is portable and lightweight, scaling effectively to multiple GPUs and multiple machines.

MXNet is also more than a deep learning project. It is also a collection of
[blue prints and guidelines](http://mxnet.io/architecture/index.html#deep-learning-system-design-concepts) for building
deep learning systems, and interesting insights of DL systems for hackers.

[![Join the chat at https://gitter.im/dmlc/mxnet](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dmlc/mxnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

What's New
----------
* [Version 0.9.3 Release](./docs/architecture/release_note_0_9.md) - First 0.9 official release.
* [Version 0.9.1 Release (NNVM refactor)](./docs/architecture/release_note_0_9.md) - NNVM branch is merged into master now. An official release will be made soon.
* [Version 0.8.0 Release](https://github.com/dmlc/mxnet/releases/tag/v0.8.0)
* [Updated Image Classification with new Pre-trained Models](./example/image-classification)
* [Python Notebooks for How to Use MXNet](https://github.com/dmlc/mxnet-notebooks)
* [MKLDNN for Faster CPU Performance](./MKL_README.md)
* [MXNet Memory Monger, Training Deeper Nets with Sublinear Memory Cost](https://github.com/dmlc/mxnet-memonger)
* [Tutorial for NVidia GTC 2016](https://github.com/dmlc/mxnet-gtc-tutorial)
* [Embedding Torch layers and functions in MXNet](http://mxnet.io/how_to/torch.html)
* [MXNet.js: Javascript Package for Deep Learning in Browser (without server)
](https://github.com/dmlc/mxnet.js/)
* [Design Note: Design Efficient Deep Learning Data Loading Module](http://mxnet.io/architecture/note_data_loading.html)
* [MXNet on Mobile Device](http://mxnet.io/how_to/smart_device.html)
* [Distributed Training](http://mxnet.io/how_to/multi_devices.html)
* [Guide to Creating New Operators (Layers)](http://mxnet.io/how_to/new_op.html)
* [Go binding for inference](https://github.com/songtianyi/go-mxnet-predictor)
* [Amalgamation and Go Binding for Predictors](https://github.com/jdeng/gomxnet/) - Outdated
* [Training Deep Net on 14 Million Images on A Single Machine](http://mxnet.io/tutorials/computer_vision/imagenet_full.html)

Contents
--------
* [Documentation and Tutorials](http://mxnet.io/)
* [Design Notes](http://mxnet.io/architecture/index.html)
* [Code Examples](example)
* [Installation](http://mxnet.io/get_started/setup.html)
* [Pretrained Models](https://github.com/dmlc/mxnet-model-gallery)
* [Contribute to MXNet](http://mxnet.io/community/contribute.html)
* [Frequent Asked Questions](http://mxnet.io/how_to/faq.html)

Features
--------
* Design notes providing useful insights that can re-used by other DL projects
* Flexible configuration for arbitrary computation graph
* Mix and match imperative and symbolic programming to maximize flexibility and efficiency
* Lightweight, memory efficient and portable to smart devices
* Scales up to multi GPUs and distributed setting with auto parallelism
* Support for Python, R, C++ and Julia
* Cloud-friendly and directly compatible with S3, HDFS, and Azure


Installation Guide for HIP Port
-------------------------------

***Generic Installation Steps:***

Install the system requirement following the [ROCm’s installation guide](http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html)

***Installation Steps on HCC and NVCC PLATFORM***

***Prerequisites*** 

Install CUDA 8.0 following the NVIDIA’s [installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/) to setup MXNet with GPU support 

***Note:*** Make sure to add CUDA install path to LD_LIBRARY_PATH.
Example - export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

***Building MXNet from source is a 2 step process.***

1. Build the MXNet core shared library, libmxnet.so, from the C++ sources.
2. Build the language specific bindings. Example - Python bindings, Scala bindings.

***Minimum Requirements***

1.  [GCC 4.8](https://gcc.gnu.org/gcc-4.8/) or later to compile C++ 11.
2.  [GNU Make](https://www.gnu.org/software/make/)

***ROCm installation***

***Step 1:*** Add the ROCm apt repository
For Debian based systems, like Ubuntu, configure the Debian ROCm repository as follows:
```
$ wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
$ sudo sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'
```
***Step 2:*** Install or Update
Next, update the apt-get repository list and install/update the rocm package:
Warning: Before proceeding, make sure to completely [uninstall any previous ROCm package](https://github.com/RadeonOpenCompute/ROCm#removing-pre-release-packages)

```
$ sudo apt update
$ sudo apt install -y rocm-dkms
```

***Step 3:*** Install dependent libraries
```
$ sudo apt-get install rocm-device-libs rocblas rocm-libs rocblas hipblas rocrand rocfft
$ sudo apt install -y hip-thrust 
$ sudo apt install -y rocm-opencl rocm-opencl-dev rocm-utils
$ sudo apt install -y miopengemm miopen-hip
```
For detailed installation steps refer the given [installation link](https://github.com/RadeonOpenCompute/ROCm)

***Step 4 :*** Install and build rocPRIM.
```
$ git clone https://github.com/ROCmSoftwarePlatform/rocPRIM.git
$ cd rocPRIM; mkdir build; cd build
$ CXX=/opt/rocm/bin/hcc cmake -DBUILD_BENCHMARK=OFF -DBUILD_TEST=OFF ../
$ make -j2
```

***Build the MXNet core shared library***

***Step 1 :*** Install build tools and git.
```
$ sudo apt-get update
$ sudo apt-get install -y build-essential git
```

***Step 2 :*** Install [OpenCV](https://opencv.org/)

MXNet uses OpenCV for efficient image loading and augmentation operations.
```
$ sudo apt-get install -y libopencv-dev
```
***Step 3 :*** Install OpenBLAS
```
$sudo apt-get install -y libopenblas-dev liblapack-dev libomp-dev libatlas-dev libatlas-base-dev
```
***Step 4 :*** Download MXNet sources and build MXNet core shared library.
```
$ git clone --recursive https://github.com/ROCmSoftwarePlatform/mxnet
$ cd mxnet
```
To compile on HCC PLATFORM:	
```
$ export HIP_PLATFORM=hcc
$ make -jn (n = no of cores)
```
To compile on NVCC PLATFORM:	
```
$ export HIP_PLATFORM=nvcc
$ make -jn (n = no of cores) 
```
***Note:*** 

1. USE_OPENCV, USE_BLAS, USE_CUDA, USE_CUDA_PATH are make file flags to set compilation options to use OpenCV, CUDA libraries. You can explore and use more compilation options in make/config.mk. Make sure to set USE_CUDA_PATH to right CUDA installation path. In most cases it is - /usr/local/cuda.
2. MXNet uses rocBLAS, hcFFT, hcRNG  and lapack libraries for accelerated numerical computations. cuDNN is not enabled as it is being migrated to Miopen.

***Install the MXNet Python binding***

***Step 1 :*** Install prerequisites - python, setup-tools, python-pip and numpy.
```
$ sudo apt-get install -y python-dev python-setuptools python-numpy python-pip
$ sudo apt install -y fftw3 fftw3-dev pkg-config
```
***Step 2 :*** Install the MXNet Python binding.
```
$ cd python
$ sudo python setup.py install 
```

Ask Questions
-------------
* Please use [mxnet/issues](https://github.com/dmlc/mxnet/issues) for how to use mxnet and reporting bugs

License
-------
© Contributors, 2015-2017. Licensed under an [Apache-2.0](https://github.com/dmlc/mxnet/blob/master/LICENSE) license.

Reference Paper
---------------

Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao,
Bing Xu, Chiyuan Zhang, and Zheng Zhang.
[MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems](https://github.com/dmlc/web-data/raw/master/mxnet/paper/mxnet-learningsys.pdf).
In Neural Information Processing Systems, Workshop on Machine Learning Systems, 2015

History
-------
MXNet emerged from a collaboration by the authors of [cxxnet](https://github.com/dmlc/cxxnet), [minerva](https://github.com/dmlc/minerva), and [purine2](https://github.com/purine/purine2). The project reflects what we have learned from the past projects. MXNet combines aspects of each of these projects to achieve flexibility, speed, and memory efficiency.
