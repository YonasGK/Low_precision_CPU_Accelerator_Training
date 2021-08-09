import conv_CPU_DLA_grad as dla
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt



class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=dla.Conv_2d_DLA(input_shape=(1,28,28),output_channel=20, kernel_shape=(5,5), dtype=trt.float16)
        self.conv2=nn.Conv2d(20,32,3,1)
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        return x

model_v=nn.Conv2d(1,20,5,1)
model= dla.Conv_2d_DLA(input_shape=(1,28,28),output_channel=20, kernel_shape=(5,5), dtype=trt.float16)

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
inputs=torch.rand((1,1,28,28), requires_grad=True)
output=model(inputs)

ground_truth=torch.rand(1,32,22,22)
#loss=loss_fn(output, ground_truth)

result1=model_v(inputs)
model.weights=model_v.weight
model.bias=model_v.bias
result2=model(inputs)
loss=(result1-result2).sum()

print(loss)
#optimizer.zero_grad()
#loss.backward()
#optimizer.step()
"""

results_vanilla= model_vanilla(inputs)

loss=(results_vanilla-outs).sum()
model_vanilla.weight.requires_grad_(True)
#inputs.requires_grad=False
#loss.backward(retain_graph=True)
orrint(inputs.grad)
print(loss.grad_fn)
print(model_vanilla.weight.grad_fn)

results=torch.tensor(model(model_vanilla.weight, inputs)).reshape(results_vanilla.shape)
loss=(results-outs).sum()
results.requires_grad_(True)
print(model_vanilla.weight.requires_grad)
print(results.requires_grad)
print(inputs.requires_grad)
loss.backward(outs[0])
"""
