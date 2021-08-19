# Low Precision Heterogenous DNN training
 
 This repository contains a report, TensorRT(https://github.com/YonasGK/Low_precision_CPU_Accelerator_Training/blob/main/tensorrt_pytorch_per_layer_tests/conv_CPU_Acc_TensorRT.py)  and Pytorch implementation(https://github.com/YonasGK/Low_precision_CPU_Accelerator_Training/blob/main/tensorrt_pytorch_per_layer_tests/conv_CPU_Acc_pytorch.py) of a custom conv2d layer for hetergenous training, and sample test codes.
 
   1. Setting up a working envoronment
    
   1.1 Jetson Xavier
   When flashing the Jeston Xavier, TensorRT and all the necassary cuda packages are installed. The versions are given as below:
   
   To check the cuda version: 
   
       $ nvcc --version
       Cuda compilation tools, release 10.2, V10.2.89
   
   To check TensorRT version:
        
       $ python3
       >>> import tensorrt as trt
       >>> trt.__version__
       '7.1.3.0'
       
   Installing Pytorch: A complete guide on how to install Pytorch for Jetson Xavier is shown in this link: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-9-0-now-available/72048. In order to do further study on avoiding data copy in shared platforms, it is recommended to build pytorch from source. The version installed for our study was '1.8.0'.
   
   After installing the above packages, we can proceed to cloning this repository and running the tests as shown in section 2 below.
   
   1.2 Desktop
    
   Nvidia provides a Docker Image, which has TensorRT and all the necassary cuda packages installed. We have compiled a summerized setup  procedure below.
   
   Pull a docker image
   
        $ docker pull nvcr.io/nvidia/tensorrt:21.07-py3
        
   Run the docker image
   
        $ sudo docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/tensorrt:21.07-py3
   
   Install python dependencies
   
        $ /opt/tensorrt/python/python_setup.sh
   
   Uninstall the Pytorch Package and reinstall it with the version we want
   
        $ pip3 uninstall torch
        $ pip3 install torch==1.8.0
        
   Change your directory to your working space and proceed to cloning and running the test codes in this repo.
   
   You can find the detailed steps on how to set up the docker environment in the link: https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt.
   
   
   2. Cloning and running tests in this repository
   
   Clone repo
   
        $ 
        
        
        
        
   
   
   
   
   After checking the above requirments
 
 ** You can find tests in the following directories**
 1. Per layer test tensorrt and pytorch: /tensorrt_per_layer_test

 2. mnist test tensorrt and pytorch: /mnist

 3. resnet test tensorrt and pytorch: /resnet
 
 Note
 
 The above tests are all done with pytorch v1.8.0 (https://pytorch.org/tutorials/) and TenorRT v7.1.3.0(https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html), both on Jetson Xavier and PC.

    To do

- Detailed OS and Kernel level analysis of data flow using(Nvidia Nsight Systems). https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Device a method to alleviate data copy bottleneck in systems with shared memory architecture for the full pytorch implementation by building pytorch from source.
- Do further analysis(function level: https://docs.python.org/3/library/profile.html#module-cProfile and Kernel Level: Nsight System)  to stem out bottlenecks once step 2 is done and propose architecture level modifications that address the issue.
- Expand implementation to include fully connected layers.
- Used quantization to further optimize the data movement latency and computational latency


