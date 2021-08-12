import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import tensorrt as trt
import conv_CPU_Acc_TensorRT as dla


inputs=torch.randn(1,20, 12, 12)
grad_out=torch.randn(1,50 ,8, 8)
weight=torch.randn(50,20,5,5)

grad=dla.Conv2d_DLA_grad(input_name="conv", inputs=inputs, grad_out=grad_out, weight=weight,dtype=trt.float32, stride=(1,1), padding=(0,0))
w_grad=grad.grad_weight(inputs, grad_out)
#print("weight grad from tensorrt: ", w_grad)

w_grad2=torch.nn.grad.conv2d_weight(inputs, weight.shape, grad_out, stride=(1,1), padding=(0,0))
#print("weight grad from pytorch: ", w_grad2)
#print("weight grad from tensorrt: ", w_grad)
print("diff = ", (w_grad2-w_grad).sum())
