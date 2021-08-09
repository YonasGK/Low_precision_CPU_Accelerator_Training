import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import time
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class use_DLA_linear():
    def __init__(self, output_channel=0, weights=None, bias=None):
        super(use_DLA_linear, self).__init__()
        self.output_channel=output_channel
        self.weights=weights
        self.bias=bias
        self.network=None
        self.sum=0
        self.runs=0
        self.f=open("serial.txt", "wb")
        self.context=self.engine=self.inputs=self.outputs=self.bindings=self.stream=None
    def populate_network(self, network, weights, bias):
        # Configure the network layers based on the weights provided.
        input_tensor = network.add_input(name="fc", dtype=trt.float16, shape=trt.DimsCHW(1,weights.shape[1],1))
        #print(input_tensor)
        _w = weights.detach().numpy()
        _b = bias.detach().numpy()
        fc1 = network.add_fully_connected(input=input_tensor, num_outputs=self.output_channel, kernel=_w, bias=_b)

        network.mark_output(tensor=fc1.get_output(0))

    def build_engine(self,weights, bias):
        # For more information on TRT basics, refer to the introductory samples.
        with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config, builder.create_network() as network:
            builder.max_workspace_size = common.MB(100)
            builder.max_batch_size = 1
            config.default_device_type= trt.DeviceType.DLA
            config.DLA_core=0
            config.set_flag(trt.BuilderFlag.FP16)
            self.populate_network(network, weights, bias)
            self.network=network
        # Build and return an engine.
            #return builder.build_cuda_engine(network)
            return builder.build_engine(network, config)

    # Loads input from cpu/gpu
    def load_input(self, pagelocked_buffer, inputs_cpu):
        img=inputs_cpu.flatten()
        np.copyto(pagelocked_buffer,img)
    def initialize_engine(self):
        self.engine=self.build_engine(self.weights, self.bias)
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        self.context=self.engine.create_execution_context()
        print("engine initialized")
    def forward(self, weights, bias,inputs_cpu):
        self.load_input(self.inputs[0].host, inputs_cpu)
       # print("forward")
        start=time.time()
        output=None
        #with self.engine.create_execution_context() as context:
        [output] = common.do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream, batch_size=inputs_cpu.shape[0])
        #print(output)
        end=time.time()
        self.stream.synchronize()
        self.sum+=end-start
        self.runs+=1
        print("Time: ", end-start)
        print("average time: ", self.sum/self.runs)
        return output

class use_DLA_linear_autograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs_cpu, weights,bias, model):
        ctx.save_for_backward(inputs_cpu, weights)
        outs=torch.tensor(model.forward(weights, bias, inputs_cpu))
        #print(outs.shape)
        outs = outs.reshape(inputs_cpu.shape[0], int(outs.shape[0]/inputs_cpu.shape[0]))
        return outs
    @staticmethod
    def backward(ctx, grad_out):
        inputs,weights=ctx.saved_tensors
        x_grad =w_grad=None
        if ctx.needs_input_grad[0]:
            w_transpose=torch.transpose(weights, 0,1)
            x_grad=torch.tensordot(grad_out, w_transpose, dims=([1], [1]))
        if ctx.needs_input_grad[1]:
            grad_out_t=torch.transpose(grad_out, 0,1)
            w_grad=torch.matmul(grad_out_t, inputs)
        return x_grad, w_grad, None,None

class Linear_2d_DLA(nn.Module):
    def __init__(self, input_channel=0, output_channel=0):
         super(Linear_2d_DLA,self).__init__()
         self.input_channel=input_channel
         self.out_channel=output_channel
         self.weights=torch.nn.Parameter(torch.randn(output_channel, input_channel), requires_grad=True)
         self.bias=torch.rand(output_channel)
         self.model=use_DLA_linear(self.out_channel, self.weights, self.bias)
         self.model.initialize_engine()
    def forward(self, inputs_cpu):
        out = use_DLA_linear_autograd.apply(inputs_cpu, self.weights, self.bias, self.model)
        return out

