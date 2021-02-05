# pytorch-attention-mechanism
## Table of Contents

- [Introduction](#Introduction)
- [Attention Mechanism](#Attention-Mechanism)
- [Plug and Play Module](#Plug-and-Play-Module)

- [Paper List](#Paper-List)

## Introduction

PyTorch实现多种计算机视觉中网络设计中用到的Attention机制，还收集了一些即插即用模块，比如最经典的SPP，希望能为大家设计新模块带来灵感，同时由于能力有限精力有限，可能很多模块并没有包括进来，可以在issue中提建议，会尽快补充上的。

注意力机制也是即插即用模块中的一部分，由于相关工作比较多，所以将其单独列出来了。



## Attention Mechanism

| paper                                                        | publish   | link                                                         | impl | Main Idea                                |
| ------------------------------------------------------------ | --------- | ------------------------------------------------------------ | ---- | ---------------------------------------- |
| [Global Second-order Pooling Convolutional Networks](https://cs.jhu.edu/~alanlab/Pubs20/li2020neural.pdf) | CVPR19    | [GSoPNet](https://github.com/ZilinGao/Global-Second-order-Pooling-Convolutional-Networks) |      | 将高阶和注意力机制在网络中部地方结合起来 |
| [Neural Architecture Search for Lightweight Non-Local Networks](https://cs.jhu.edu/~alanlab/Pubs20/li2020neural.pdf) | CVPR20    | [AutoNL](https://github.com/LiYingwei/AutoNL)                |      | NAS+LightNL                              |
| [Squeeze and Excitation Network](https://arxiv.org/abs/1709.01507) | CVPR18    | [SENet](https://github.com/hujie-frank/SENet)                |      | 最经典的通道注意力                       |
| [Selective Kernel Network](https://arxiv.org/pdf/1903.06586.pdf) | CVPR19    | SKNet                                                        |      | SE+动态选择                              |
| [Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521.pdf) | ECCV18    | CBAM                                                         |      | 串联空间+通道注意力                      |
| [BottleNeck Attention Module](https://arxiv.org/pdf/1807.06514.pdf) | BMVC18    | BAM                                                          |      | 并联空间+通道注意力                      |
| [Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks](http://arxiv.org/pdf/1803.02579v2.pdf) | MICCAI18  | scSE                                                         |      | 并联空间+通道注意力                      |
| [Non-local Neural Networks](https://arxiv.org/abs/1711.07971) | CVPR19    | Non-Local(NL)                                                |      | self-attention                           |
| [GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond](https://arxiv.org/abs/1904.11492) | ICCVW19   | GCNet                                                        |      | 对NL进行改进                             |
| [CCNet: Criss-Cross Attention for Semantic Segmentation](https://arxiv.org/abs/1811.11721) | ICCV19    | CCNet                                                        |      | 对NL改进                                 |
| [SA-Net:shuffle attention for deep convolutional neural networks](https://arxiv.org/pdf/2102.00240.pdf) | ICASSP 21 | SANet                                                        |      |                                          |
| [ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/pdf/1910.03151.pdf) | CVPR20    | ECANet                                                       |      | SE的改进                                 |
| [Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks](https://arxiv.org/abs/1905.09646) | CoRR19    | SGENet                                                       |      |                                          |
| [Global Second-order Pooling Convolutional Networks](http://openaccess.thecvf.com/content_CVPR_2019/papers/Gao_Global_Second-Order_Pooling_Convolutional_Networks_CVPR_2019_paper.pdf) | CVPR19    | GSoPNet                                                      |      |                                          |
|                                                              |           |                                                              |      |                                          |






- BiSeNet
- HRNet
- OCR
- ResNeSt
- DANet: Dual Attention
- AFF
  - MC-CAM
  - AFF
  - iAFF
- ShuffleAttention ICASSP 2021
- ECA Efficient Channel Attention CVPR 2020

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
- ICCV2019 CCNet: Criss Cross Attention
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

## Paper List

SENet 论文: https://arxiv.org/abs/1709.01507 解读：https://zhuanlan.zhihu.com/p/102035721


