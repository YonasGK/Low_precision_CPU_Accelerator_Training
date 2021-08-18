# MNIST TEST

1. MNIST TEST
    
    This test is used to demostrate the convergence of real models constructed from the custom conv2d modules provided. The architecture is a simple 2 convolutional layer followed by 2 fully connected layers.
2. Constructing a model
    
    A. Defining a Layer
        
      Our implementation is designed so that you can treat the custom conv2d layers as the native torch.nn.Conv2d module. Hence by replacing the native convolution layers with the custom ones we can run the test as we would with any other training model.
      
      However ,if we use the custom conv2d modules, we have to be careful about where we map the model after constructing it. We have to make sure that the model is mapped to CPU. If the model is not mapped to the CPU we would have a model which is running fully on GPU/Accelerator and it goes against our goal. So you will find below examples of how you countruct layers and run the test.
        
        
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
       
     B. Running Test
     
      You can run the test file test_mnist.py with the following options
        
            
         $ python3 test_mnist.py --help
         usage: test_mnist.py [-h] [--batch-size N] [--test-batch-size N] [--epochs N]
                     [--lr LR] [--gamma M] [--no-cuda] [--dry-run] [--seed S]
                     [--log-interval N] [--save-model]

        PyTorch MNIST Example

        optional arguments:
        -h, --help           show this help message and exit
        --batch-size N       input batch size for training (default: 64)
        --test-batch-size N  input batch size for testing (default: 1000)
        --epochs N           number of epochs to train (default: 10)
        --lr LR              learning rate (default: 1.0)
        --gamma M            Learning rate step gamma (default: 0.7)
        --no-cuda            disables CUDA training
        --dry-run            quickly check a single pass
        --seed S             random seed (default: 1)
        --log-interval N     how many batches to wait before logging training status
        --save-model         For Saving the current Model

     You can then tun a test as shown below.
                  
         $ python3 test_mnist --no-cuda --batch-size=4
         
  3. Notes
       - When running the test code(test_mnist.py) with the custom layers bw sure to use the --no-cuda option. Without that the model will be mapped to GPU/Accelerator and the test will fail.
       - Due to memory limitation on the Xavier platform when running the test codes avoid using very large batch sizes.
       - When running on a PC environment you can get a docker image from https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt which has tensorrt set up.
    
