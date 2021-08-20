# Resnet Test

## 1. Resnet test

This test is cloned from a github repo given as here: https://github.com/akamaster/pytorch_resnet_cifar10
The following models are provided:

| Name      | # layers | # params|
|-----------|---------:|--------:|
|[ResNet20](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet20-12fca82f.th)   |    20    | 0.27M   |
|[ResNet32](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet32-d509ac18.th)  |    32    | 0.46M   |
|[ResNet44](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet44-014dd654.th)   |    44    | 0.66M   |
|[ResNet56](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet56-4bfd9763.th)   |    56    | 0.85M   |
|[ResNet110](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet110-1d1ed7c2.th)  |   110    |  1.7M   |
|[ResNet1202](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet1202-f3b1deed.th) |  1202    | 19.4M   |

Our goal with this test is to show the convergence of our implementation, to observe the performance and assess bottlenecks in large models 

## 2. Constructing a model

### A. Defining a Layer
        
   Our implementation is designed so that you can treat the custom conv2d layers as the native torch.nn.Conv2d module. Hence by replacing the native convolution layers with the custom ones we can run the test as we would with any other training model.
      
   However, if we use the custom conv2d modules, we have to be careful about where we map the model after constructing it. We have to make sure that the model is mapped to CPU. If the model is not mapped to the CPU we would have a model which is running fully on GPU/Accelerator and it goes against our goal. So you will find below examples of how you countruct layers and run the test.
        
        
   The custom conv2d modules are initilized in the same way as a vanilla conv2d(nn.Conv2d) and you can edit the resnet.py file as below.
     
   **FOR vanilla convolution layer**
      
       import torch.nn as nn
       self.conv1=nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
     
        
   **FOR Pytorch+TensorRT convolution layer**
          
       import conv_CPU_Acc_TensorRT as dla    
       self.conv1=dla.Conv_2d_DLA(input_name="conv", in_channel=1, output_channel=0, kernel_shape=(3,3),dtype=trt.float16, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=False)
       
   **FOR full Pytorch custom convolution layer** 
        
       import conv_CPU_Acc_pytorch as dla
       self.conv1=dla.custom_conv2d(in_channel=1, out_channel=0, kernel_shape=(3,3), stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=False)
       
  ### B. Running Test
   You can run the test file trainer.py with the following options
                  
        python3 trainer.py --help
        ['resnet110', 'resnet1202', 'resnet20', 'resnet32', 'resnet44', 'resnet56']
        usage: trainer.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N]
                  [-b N] [--lr LR] [--momentum M] [--weight-decay W]
                  [--print-freq N] [--resume PATH] [-e] [--pretrained]
                  [--half] [--save-dir SAVE_DIR] [--save-every SAVE_EVERY]
                  [--no-cuda]

    Propert ResNets for CIFAR10 in pytorch

    optional arguments:
    -h, --help            show this help message and exit
    --arch ARCH, -a ARCH  model architecture: resnet110 | resnet1202 | resnet20
                        | resnet32 | resnet44 | resnet56 (default: resnet32)
    -j N, --workers N     number of data loading workers (default: 4)
    --epochs N            number of total epochs to run
    --start-epoch N       manual epoch number (useful on restarts)
    -b N, --batch-size N  mini-batch size (default: 128)
    --lr LR, --learning-rate LR
                        initial learning rate
    --momentum M          momentum
    --weight-decay W, --wd W
                        weight decay (default: 1e-4)
    --print-freq N, -p N  print frequency (default: 50)
    --resume PATH         path to latest checkpoint (default: none)
    -e, --evaluate        evaluate model on validation set
    --pretrained          use pre-trained model
    --half                use half-precision(16-bit)
    --save-dir SAVE_DIR   The directory used to save the trained models
    --save-every SAVE_EVERY
                        Saves checkpoints at every specified number of epochs
    --no-cuda             disables CUDA training 
 
  Here is an example:
      
       python3 trainer.py --arch resnet20 --batch-size=4 --no-cuda
      
 ## 3. Notes
 
   - When running the test code(trainer.py) with the custom layers be sure to use the --no-cuda option. Without that option the model will be mapped to GPU/Accelerator and the test will fail.  By using --no-cuda we map the network to CPU and will only secretly invoke the GPU to do conv2d operation.
   - Due to memory limitation on the Xavier platform when running the test codes avoid using very large batch sizes.
   - When running on a PC environment you can get a docker image from https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt which has tensorrt set up.
