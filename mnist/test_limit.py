import conv_CPU_DLA_grad as dla
import linear_DLA as dla_fc
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt
import torch.optim as optim

inputs_fc=torch.rand(1,100)
inputs=torch.ones(1,1,5,5)
print(inputs)
output= torch.rand(1,3,3,3)
weight=nn.Parameter(torch.ones(3,1,3,3))
grad=torch.ones(3,1,3,3)
bias=torch.zeros(3)
model=dla.Conv_2d_DLA("conv", 1 , 3)
out=model(inputs)

print("1st forward on CPU: ",out,model.conv_vanilla.weight, model.conv_vanilla.bias)
loss=(out-output).sum()
loss.backward()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer.step()
out=model(inputs)
print("after 1st weight change forward on DLA: ", out, model.conv_vanilla.weight, model.conv_vanilla.bias)

loss=(out-output).sum()
loss.backward()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print(model.conv_vanilla.weight.grad*0.01)
optimizer.step()
out=model(inputs)
print("after weight change and on DLA: ", out, model.conv_vanilla.weight, model.conv_vanilla.bias)

