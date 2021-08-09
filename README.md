1, Per layer test tensorrt and pytorch

2, mnist test tensorrt and pytorch

3, resnet test tensorrt and pytorch

 DNN training with cpu + inference accelerator
Introduction
A lot of work has been done in designing edge inference accelerators to move inference tasks from the cloud to edge devices. The advantages of edge inference engines are privacy, latency, and energy efficiency due to the elimination of transmission of data to the cloud in a power hungry and slow communications infrastructure. Training on edge is a relatively new and rapidly growing research area that seeks to move training tasks from centralized data centers to edge devices to provide a distributed and customized training paradigm. To this end we try to introduce a minimal software/hardware co-design framework on readily available inference platforms and software stacks to train DNNs on edge devices.
background
Edge computing is a realm where computation is pushed across a large number of small devices instead of being concentrated in a centralized data center. As the computing capability of edge devices is ever growing it becomes more reasonable to do processing on these end nodes. Using edge devices to do heavy processing provides latency, energy efficiency and privacy advantages. When processing on edge the whole process is done at the edge which eliminates the need for external communication, thus edge devices can make time sensitive real-time decisions quicker. Even Though, processing heavy applications on edge devices is an energy-intensive task, the transmission of data to the cloud would cost orders of magnitude more energy, thus by processing on edge we would get a net reduction in energy usage. Lastly, processing at the edge will keep personal data near to the user, thus it will reduce the risk of privacy violation and unauthorized data personal data use.
Given the above benefits it has become common to do inference on the edge. However the possibility of training on edge has not been studied well due to multiple reasons.  Firstly, if the information learned at one node is useful for other nodes, transferring model updates requires large bandwidth and latency. Secondly,  transferring training data from cloud to edge devices is more costly than transferring trained models thus, training on edge will not provide any benefits. 
However we would like to focus on cases where data is gathered and labeled on the edge device. The pre-processed data is then used to customize an already learned model obtained from the cloud. Such a paradigm would mainly provide customized service for each user, keep user data on edge and energy benefits by using quantized models and integer operation to train models on CPU/Accelerator heterogeneous systems.
Proposed mechanism
Our proposed approach to training DNNs on edge consists of both hardware and software modifications. Since edge devices have computational capability, memory and energy constraints, our training models have to be kept as small as possible. One popular method to do that is model quantization. Quantization has shown to be a promising approach to compress inference models. Over the years many algorithms have been proposed for quantization aware training and low precision training. These algorithms have made it possible to obtain a state of the art performance, while carrying out data movement and execution in low precision  . In addition to 
Software
As mentioned above on edge inference is a ubiquitous 

