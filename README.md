# pytorch-attention-mechanism
## Table of Contents

- [Introduction](#Introduction)
- [Attention Mechanism](#Attention-Mechanism)
- [Plug and Play Module](#Plug-and-Play-Module)
- [Evaluation](#Evaluation)
- [Paper List](#Paper-List)
- [Contribute](#Contribute)

## Introduction

PyTorch实现多种计算机视觉中网络设计中用到的Attention机制，还收集了一些即插即用模块，比如最经典的SPP，希望能为大家设计新模块带来灵感，同时由于能力有限精力有限，可能很多模块并没有包括进来，可以在issue中提建议，会尽快补充上的。

## Attention Mechanism

| paper                                                        | publish   | link                                                         | impl | Main Idea                                |
| ------------------------------------------------------------ | --------- | ------------------------------------------------------------ | ---- | ---------------------------------------- |
| [Global Second-order Pooling Convolutional Networks](https://cs.jhu.edu/~alanlab/Pubs20/li2020neural.pdf) | CVPR19    | [GSoPNet](https://github.com/ZilinGao/Global-Second-order-Pooling-Convolutional-Networks) |      | 将高阶和注意力机制在网络中部地方结合起来 |
| [Neural Architecture Search for Lightweight Non-Local Networks](https://cs.jhu.edu/~alanlab/Pubs20/li2020neural.pdf) | CVPR20    | [AutoNL](https://github.com/LiYingwei/AutoNL)                |      | NAS+LightNL                              |
| [Squeeze and Excitation Network](https://arxiv.org/abs/1709.01507) | CVPR18    | [SENet](https://github.com/hujie-frank/SENet)                |      | 最经典的通道注意力                       |
| [Selective Kernel Network](https://arxiv.org/pdf/1903.06586.pdf) | CVPR19    | [SKNet](https://github.com/implus/SKNet)                     |      | SE+动态选择                              |
| [Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521.pdf) | ECCV18    | [CBAM](https://github.com/Jongchan/attention-module)         |      | 串联空间+通道注意力                      |
| [BottleNeck Attention Module](https://arxiv.org/pdf/1807.06514.pdf) | BMVC18    | [BAM](https://github.com/Jongchan/attention-module)          |      | 并联空间+通道注意力                      |
| [Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks](http://arxiv.org/pdf/1803.02579v2.pdf) | MICCAI18  | [scSE](https://github.com/ai-med/squeeze_and_excitation)     |      | 并联空间+通道注意力                      |
| [Non-local Neural Networks](https://arxiv.org/abs/1711.07971) | CVPR19    | [Non-Local(NL)](https://github.com/AlexHex7/Non-local_pytorch) |      | self-attention                           |
| [GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond](https://arxiv.org/abs/1904.11492) | ICCVW19   | [GCNet](https://github.com/xvjiarui/GCNet)                   |      | 对NL进行改进                             |
| [CCNet: Criss-Cross Attention for Semantic Segmentation](https://arxiv.org/abs/1811.11721) | ICCV19    | [CCNet](https://github.com/speedinghzl/CCNet)                |      | 对NL改进                                 |
| [SA-Net:shuffle attention for deep convolutional neural networks](https://arxiv.org/pdf/2102.00240.pdf) | ICASSP 21 | [SANet](https://github.com/wofmanaf/SA-Net)                  |      | SGE+channel shuffle                      |
| [ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/pdf/1910.03151.pdf) | CVPR20    | [ECANet](https://github.com/BangguWu/ECANet)                 |      | SE的改进                                 |
| [Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks](https://arxiv.org/abs/1905.09646) | CoRR19    | [SGENet](https://github.com/implus/PytorchInsight)           |      | Group+spatial+channel                    |
| [FcaNet: Frequency Channel Attention Networks](https://arxiv.org/pdf/2012.11879.pdf) | CoRR20    | [FcaNet](https://github.com/cfzd/FcaNet)                     |      | 频域上的SE操作                           |
| [$A^2\text{-}Nets$: Double Attention Networks](https://arxiv.org/abs/1810.11579) | NeurIPS18 | [DANet](https://github.com/nguyenvo09/Double-Attention-Network) |      | NL的思想应用到空间和通道                 |
| [Asymmetric Non-local Neural Networks for Semantic Segmentation](https://arxiv.org/pdf/1908.07678.pdf) | ICCV19    | [APNB](https://github.com/MendelXu/ANN)                      |      | spp+NL                                   |
| [Efficient Attention: Attention with Linear Complexities](https://arxiv.org/pdf/1812.01243v7.pdf) | CoRR18    | [EfficientAttention](https://github.com/cmsflash/efficient-attention) |      | NL降低计算量                             |
| [Image Restoration via Residual Non-local Attention Networks](https://arxiv.org/pdf/1903.10082.pdf) | ICLR19    | [RNAN](https://github.com/yulunzhang/RNAN)                   |      |                                          |
| [Exploring Self-attention for Image Recognition](http://vladlen.info/papers/self-attention.pdf) | CVPR20    | [SAN](https://github.com/hszhao/SAN)                         |      | 理论性很强，实现起来很简单               |
| [An Empirical Study of Spatial Attention Mechanisms in Deep Networks](https://arxiv.org/pdf/1904.05873.pdf) | ICCV19    | None                                                         |      | MSRA综述self-attention                   |
| [Object-Contextual Representations for Semantic Segmentation](https://arxiv.org/pdf/1909.11065.pdf) | ECCV20    | [OCRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR?v=2) |      | 复杂的交互机制，效果确实好               |
| [IAUnet: Global Context-Aware Feature Learning for Person Re-Identification](https://arxiv.org/pdf/2009.01035.pdf) | TTNNLS20  | [IAUNet](https://github.com/blue-blue272/ImgReID-IAnet)      |      | 引入时序信息                             |
| [ResNeSt: Split-Attention Networks](https://arxiv.org/pdf/2004.08955.pdf) | CoRR20    | [ResNeSt](https://github.com/zhanghang1989/ResNeSt)          |      | SK+ResNeXt                               |
| [Gather-Excite: Exploiting Feature Context in Convolutional Neural Networks](https://papers.nips.cc/paper/8151-gather-excite-exploiting-feature-context-in-convolutional-neural-networks.pdf) | NIPS18    | [GENet](https://github.com/hujie-frank/GENet)                |      | SE续作                                   |
| [Improving Convolutional Networks with Self-calibrated Convolutions](http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf) | CVPR20    | [SCNet](https://github.com/MCG-NKU/SCNet)                    |      | 自校正卷积                               |


- DANet: Dual Attention
- AFF
  - MC-CAM
  - AFF
  - iAFF
  
  

## Plug and Play Module

- ACBlock
- Swish、wish Activation
- ASPP Block
- DepthWise Convolution
- Fused Conv & BN
- MixedDepthwise Convolution
- PSP Module
- RFBModule
- SematicEmbbedBlock
- SSH Context Module
- Some other usefull tools such as concate feature map、flatten feature map
- WeightedFeatureFusion:EfficientDet中的FPN用到的fuse方式
- StripPooling：CVPR2020中核心代码StripPooling
- GhostModule: CVPR2020GhostNet的核心模块
- SlimConv: SlimConv3x3 
- Context Gating： video classification
- EffNetBlock: EffNet
- ECCV2020 BorderDet: Border aligment module
- CVPR2019 DANet: Dual Attention
- Object Contextual Representation for sematic segmentation: OCRModule
- FPT: 包含Self Transform、Grounding Transform、Rendering Transform
- DOConv: 阿里提出的Depthwise Over-parameterized Convolution
- PyConv: 起源人工智能研究院提出的金字塔卷积
- ULSAM：用于紧凑型CNN的超轻量级子空间注意力模块
- DGC: ECCV 2020用于加速卷积神经网络的动态分组卷积
- DCANet: ECCV 2020 学习卷积神经网络的连接注意力
- PSConv: ECCV 2020 将特征金字塔压缩到紧凑的多尺度卷积层中
- Dynamic Convolution: CVPR2020 动态滤波器卷积（非官方）
- CondConv: Conditionally Parameterized Convolutions for Efficient Inference

## Evaluation

基于CIFAR10+ResNet+待测评模块，对模块进行初步测评。测评代码来自于另外一个库：https://github.com/kuangliu/pytorch-cifar/。实验过程中，不使用预训练权重，进行随机初始化。

working on it....

| 模型         | top1 acc | time    | params(MB) |
| ------------ | -------- | ------- | ---------- |
| SENet18      | 95.20%   |         |            |
| ResNet18     | 95.61%   |         | 11,173,962 |
| ResNet50     | 95.50%   | 4:24:38 | 23,520,842 |
| ShuffleNetV2 | 91.90%   | 1:02:50 | 1,263,854  |
| GoogLeNet    | 91.90%   | 1:02:50 | 6,166,250  |
| MobileNetV2  | 92.66%   | 2:04:57 | 2,296,922  |



## Paper List

SENet 论文: https://arxiv.org/abs/1709.01507 解读：https://zhuanlan.zhihu.com/p/102035721

## Contribute

欢迎在issue中提出补充的文章paper和对应code链接。