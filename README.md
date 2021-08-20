# Low Precision Heterogenous DNN training
 
 This repository contains:
 
 Project Report (https://github.com/YonasGK/Low_precision_CPU_Accelerator_Training/blob/3f5441b156fd572cadad9df04450deef1673f1e0/Report.md), 
 
 TensorRT Implementation (https://github.com/YonasGK/Low_precision_CPU_Accelerator_Training/blob/main/tensorrt_pytorch_per_layer_tests/conv_CPU_Acc_TensorRT.py)  and 
 
 Pytorch Implementation (https://github.com/YonasGK/Low_precision_CPU_Accelerator_Training/blob/main/tensorrt_pytorch_per_layer_tests/conv_CPU_Acc_pytorch.py) 

 of a custom conv2d layer for hetergenous training, and sample test codes.
 
   ## 1. Setting up a working envoronment
    
   ### 1.1 Jetson Xavier
   
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
   
  ### 1.2 Desktop
    
   Nvidia provides a Docker Image, which has TensorRT and all the necassary cuda packages installed. We have compiled a summerized setup  procedure below.
   
   Pull a docker image
   
        $ docker pull nvcr.io/nvidia/tensorrt:21.06-py3
        
   Run the docker image
   
        $ sudo docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/tensorrt:21.06-py3
   
   Install python dependencies
   
        $ /opt/tensorrt/python/python_setup.sh
   
   Uninstall the Pytorch Package and reinstall it with the version we want
   
        $ pip3 uninstall torch
        $ pip3 install torch==1.8.0
        
   Change your directory to your working space and proceed to cloning and running the test codes in this repo.
   
   You can find the detailed steps on how to set up the docker environment in the link: https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt.
   
   
  ##  2. Cloning and running tests in this repository
   
   Clone the repository
   
        $  git clone https://github.com/YonasGK/Low_precision_CPU_Accelerator_Training.git
        $  cd Low_precision_CPU_Accelerator_Training
        
   To run per layer tests access the /tensorrt_pytorch_per_layer_tests directory and follow the tutorial
   
       $  cd tensorrt_pytorch_per_layer_tests
       
   To run MNIST tests access the /mnist directory and follow the tutorial
   
       $  cd mnist
       
   To run Resnet tests access the /resnet directory and follow the tutorial
   
      $   cd resnet
      
  
  ## 3. Function level and Kernel level profiling
  
  ### 3.1 Function level profiling
  
  In our project the profiler used for function level analysis is CProfile which is incorporated in the python package.
  
  ##### Usage
  
      import cProfile, pstats, io
      from pstats import SortKey
      pr = cProfile.Profile()
      pr.enable()
      # ... do something ...
      pr.disable()
      pr.print_stats(sort=2)
     
  During printing, the sort parameter indicates which field to use to sort the profiled values. For instance, in the above example '2' indicates to sort using the cumulative time per each function call.
  
  For detailed documentation, please refer to the link: https://docs.python.org/3/library/profile.html#module-cProfile
  
 ### 3.2 Kernel level profiling
 
 In our project we used Nvidia Nsight Systems to do a deep Kernel level anaysis of the GPU. The given profiler is installed as part of the CUDA Toolkit and analysis could entirly be done on the shell with Command Line Interface(https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-installing), however if you want to use a GUI(highly recommended) to observe the details of the profiled application you can install it from the link: https://developer.nvidia.com/nsight-systems.
 
 ##### Usage
 
 To profile an application from command line
 
     $ nsys profile [options] <application> [application-arguments]
 
 Then you will obtain a '.qdrep' file which can be viewed from the command line as
 
 
     $ nsys stats [options] report1.qdrep
 
 One option used very often in our work is 'gputrace', which provides a detailed timeline of GPU kernel calls.

    $ nsys stats --report gputrace report1.qdrep
 
 
 You can also use the GUI by importing the '.qdrep' file to the GUI
 
 (Disclaimer: Although Nsight Systems can provide CPU profiling, we haven't used that option because CPU profiling occured to be a very heavy task which crashed the our Xavier board multiple times.)
 
 For more details on Nvidia Nsight Systems please refer to the link: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
 
  

   
  # To do

- Detailed OS and Kernel level analysis of data flow using(Nvidia Nsight Systems). https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Device a method to alleviate data copy bottleneck in systems with shared memory architecture for the full pytorch implementation by building pytorch from source.
- Do further analysis(function level: https://docs.python.org/3/library/profile.html#module-cProfile and Kernel Level: Nsight System)  to stem out bottlenecks once step 2 is done and propose architecture level modifications that address the issue.
- Expand implementation to include fully connected layers.
- Used quantization to further optimize the data movement latency and computational latency


