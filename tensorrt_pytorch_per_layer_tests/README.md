# TensorRT and Pytorch Per Layer Tests

## 1. Per Layer experiment Code

   This experiment is for per layer analysis of any type of convolutional layer. The tests are written in test_pytorch.py and test_tensorrt.py. The tests involoves a single convolution layer. We run a training loop on the single layer by feeding it input and having a ground truth that has the same dimension as the output activation. We run 150 training loops and can profile using either CProfile https://docs.python.org/3/library/profile.html#module-cProfile for a function level analysis or nvidia nsight systems https://docs.nvidia.com/nsight-systems/UserGuide/index.html for kernel level analysis.
      
## 2.  Running Test  

   To run the tests: run
                      
        $ python3 test_pytorch.py
        $ python3 test_tensorrt.py

## 3.  Initilizing the custom conv2d modules.
  
  The custom conv2d modules are initilized in the same way as a vanilla conv2d(nn.Conv2d).
     
   **FOR vanilla convolution layer**
      
      import torch.nn as nn
      self.conv1=nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
     
     
   **FOR Pytorch+TensorRT convolution layer**
          
       import conv_CPU_Acc_TensorRT as dla    
       self.conv1=dla.Conv_2d_DLA(input_name="conv", in_channel=1, output_channel=0, kernel_shape=(3,3),dtype=trt.float16, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=False)
       
   **FOR full Pytorch custom convolution layer** 
        
       import conv_CPU_Acc_pytoch as dla
       self.conv1=dla.custom_conv2d(in_channel=1, out_channel=0, kernel_shape=(3,3), stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=False)

 ## 4. Notes
  - To run per layer tests you can open the test codes and just edit the parameters in initilizing the layers and tensors.
  - The CProfiler is already written in the but is commented out, thus to use the CProfiler just uncomment the commented lines once you get introduced to CProfiler from the documentation attached on the link above.
  - Tutorials on how to use the Nvidia Nsight System is provided in the link attached above. However, if you are running this code on Xavier I strongly advise against profiling the CPU. Due to lack of resources and Xavier has crashed multiple time when trying to run CPU profiling with Nsight Systems. 
