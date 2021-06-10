# Awesome-Attention-Mechanism-in-cv [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![](doc/att.jpg)

## Table of Contents

- [Introduction](#Introduction)
- [Attention Mechanism](#Attention-Mechanism)
- [Plug and Play Module](#Plug-and-Play-Module)
- [Evaluation](#Evaluation)
- [Paper List](#Paper-List)
- [Contribute](#Contribute)

## Introduction

PyTorch实现多种计算机视觉中网络设计中用到的Attention机制，还收集了一些即插即用模块。由于能力有限精力有限，可能很多模块并没有包括进来，有任何的建议或者改进，可以提交issue或者进行PR。

## Attention Mechanism

| Paper                                                        | Publish     | Link                                                         | Main Idea                                                    | Blog                                            |
| ------------------------------------------------------------ | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------- |
| [Global Second-order Pooling Convolutional Networks](https://cs.jhu.edu/~alanlab/Pubs20/li2020neural.pdf) | CVPR19      | [GSoPNet](https://github.com/ZilinGao/Global-Second-order-Pooling-Convolutional-Networks) | 将高阶和注意力机制在网络中部地方结合起来                     |                                                 |
| [Neural Architecture Search for Lightweight Non-Local Networks](https://cs.jhu.edu/~alanlab/Pubs20/li2020neural.pdf) | CVPR20      | [AutoNL](https://github.com/LiYingwei/AutoNL)                | NAS+LightNL                                                  |                                                 |
| [Squeeze and Excitation Network](https://arxiv.org/abs/1709.01507) | CVPR18      | [SENet](https://github.com/hujie-frank/SENet)                | 最经典的通道注意力                                           | [zhihu](https://zhuanlan.zhihu.com/p/102035721) |
| [Selective Kernel Network](https://arxiv.org/pdf/1903.06586.pdf) | CVPR19      | [SKNet](https://github.com/implus/SKNet)                     | SE+动态选择                                                  | [zhihu](https://zhuanlan.zhihu.com/p/102034839) |
| [Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521.pdf) | ECCV18      | [CBAM](https://github.com/Jongchan/attention-module)         | 串联空间+通道注意力                                          | [zhihu](https://zhuanlan.zhihu.com/p/102035273) |
| [BottleNeck Attention Module](https://arxiv.org/pdf/1807.06514.pdf) | BMVC18      | [BAM](https://github.com/Jongchan/attention-module)          | 并联空间+通道注意力                                          | [zhihu](https://zhuanlan.zhihu.com/p/102033063) |
| [Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks](http://arxiv.org/pdf/1803.02579v2.pdf) | MICCAI18    | [scSE](https://github.com/ai-med/squeeze_and_excitation)     | 并联空间+通道注意力                                          | [zhihu](https://zhuanlan.zhihu.com/p/102036086) |
| [Non-local Neural Networks](https://arxiv.org/abs/1711.07971) | CVPR19      | [Non-Local(NL)](https://github.com/AlexHex7/Non-local_pytorch) | self-attention                                               | [zhihu](https://zhuanlan.zhihu.com/p/102984842) |
| [GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond](https://arxiv.org/abs/1904.11492) | ICCVW19     | [GCNet](https://github.com/xvjiarui/GCNet)                   | 对NL进行改进                                                 | [zhihu](https://zhuanlan.zhihu.com/p/102990363) |
| [CCNet: Criss-Cross Attention for Semantic Segmentation](https://arxiv.org/abs/1811.11721) | ICCV19      | [CCNet](https://github.com/speedinghzl/CCNet)                | 对NL改进                                                     |                                                 |
| [SA-Net:shuffle attention for deep convolutional neural networks](https://arxiv.org/pdf/2102.00240.pdf) | ICASSP 21   | [SANet](https://github.com/wofmanaf/SA-Net)                  | SGE+channel shuffle                                          | [zhihu](https://zhuanlan.zhihu.com/p/350912960) |
| [ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/pdf/1910.03151.pdf) | CVPR20      | [ECANet](https://github.com/BangguWu/ECANet)                 | SE的改进                                                     |                                                 |
| [Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks](https://arxiv.org/abs/1905.09646) | CoRR19      | [SGENet](https://github.com/implus/PytorchInsight)           | Group+spatial+channel                                        |                                                 |
| [FcaNet: Frequency Channel Attention Networks](https://arxiv.org/pdf/2012.11879.pdf) | CoRR20      | [FcaNet](https://github.com/cfzd/FcaNet)                     | 频域上的SE操作                                               |                                                 |
| [$A^2\text{-}Nets$: Double Attention Networks](https://arxiv.org/abs/1810.11579) | NeurIPS18   | [DANet](https://github.com/nguyenvo09/Double-Attention-Network) | NL的思想应用到空间和通道                                     |                                                 |
| [Asymmetric Non-local Neural Networks for Semantic Segmentation](https://arxiv.org/pdf/1908.07678.pdf) | ICCV19      | [APNB](https://github.com/MendelXu/ANN)                      | spp+NL                                                       |                                                 |
| [Efficient Attention: Attention with Linear Complexities](https://arxiv.org/pdf/1812.01243v7.pdf) | CoRR18      | [EfficientAttention](https://github.com/cmsflash/efficient-attention) | NL降低计算量                                                 |                                                 |
| [Image Restoration via Residual Non-local Attention Networks](https://arxiv.org/pdf/1903.10082.pdf) | ICLR19      | [RNAN](https://github.com/yulunzhang/RNAN)                   |                                                              |                                                 |
| [Exploring Self-attention for Image Recognition](http://vladlen.info/papers/self-attention.pdf) | CVPR20      | [SAN](https://github.com/hszhao/SAN)                         | 理论性很强，实现起来很简单                                   |                                                 |
| [An Empirical Study of Spatial Attention Mechanisms in Deep Networks](https://arxiv.org/pdf/1904.05873.pdf) | ICCV19      | None                                                         | MSRA综述self-attention                                       |                                                 |
| [Object-Contextual Representations for Semantic Segmentation](https://arxiv.org/pdf/1909.11065.pdf) | ECCV20      | [OCRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR?v=2) | 复杂的交互机制，效果确实好                                   |                                                 |
| [IAUnet: Global Context-Aware Feature Learning for Person Re-Identification](https://arxiv.org/pdf/2009.01035.pdf) | TTNNLS20    | [IAUNet](https://github.com/blue-blue272/ImgReID-IAnet)      | 引入时序信息                                                 |                                                 |
| [ResNeSt: Split-Attention Networks](https://arxiv.org/pdf/2004.08955.pdf) | CoRR20      | [ResNeSt](https://github.com/zhanghang1989/ResNeSt)          | SK+ResNeXt                                                   |                                                 |
| [Gather-Excite: Exploiting Feature Context in Convolutional Neural Networks](https://papers.nips.cc/paper/8151-gather-excite-exploiting-feature-context-in-convolutional-neural-networks.pdf) | NeurIPS18   | [GENet](https://github.com/hujie-frank/GENet)                | SE续作                                                       |                                                 |
| [Improving Convolutional Networks with Self-calibrated Convolutions](http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf) | CVPR20      | [SCNet](https://github.com/MCG-NKU/SCNet)                    | 自校正卷积                                                   |                                                 |
| [Rotate to Attend: Convolutional Triplet Attention Module](https://arxiv.org/pdf/2010.03045.pdf) | WACV21      | [TripletAttention](https://github.com/LandskapeAI/triplet-attention) | CHW两两互相融合                                              |                                                 |
| [Dual Attention Network for Scene Segmentation](https://arxiv.org/pdf/1809.02983.pdf) | CVPR19      | [DANet](https://github.com/junfu1115/DANet)                  | self-attention                                               |                                                 |
| [Relation-Aware Global Attention for Person Re-identification](https://arxiv.org/pdf/1904.02998v1.pdf) | CVPR20      | [RGA](https://github.com/microsoft/Relation-Aware-Global-Attention-Networks) | 用于reid                                                     |                                                 |
| [Attentional Feature Fusion](https://arxiv.org/abs/2009.14082) | WACV21      | [AFF](https://github.com/YimianDai/open-aff)                 | 特征融合的attention方法                                      |                                                 |
| [An Attentive Survey of Attention Models](https://arxiv.org/abs/1904.02874) | CoRR19      | None                                                         | 包括NLP/CV/推荐系统等方面的注意力机制                        |                                                 |
| [Stand-Alone Self-Attention in Vision Models](https://arxiv.org/pdf/1906.05909.pdf) | NeurIPS19   | [FullAttention](https://github.com/leaderj1001/Stand-Alone-Self-Attention) | 全部的卷积都替换为self-attention                             |                                                 |
| [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897) | ECCV18      | [BiSeNet](https://github.com/CoinCheung/BiSeNet)             | 类似FPN的特征融合方法                                        | [zhihu](https://zhuanlan.zhihu.com/p/105925132) |
| [DCANet: Learning Connected Attentions for Convolutional Neural Networks](https://arxiv.org/pdf/2007.05099.pdf) | CoRR20      | [DCANet](https://github.com/13952522076/DCANet)              | 增强attention之间信息流动                                    |                                                 |
| [An Empirical Study of Spatial Attention Mechanisms in Deep Networks](https://arxiv.org/abs/1904.05873) | ICCV19      | None                                                         | 对空间注意力进行针对性分析                                   |                                                 |
| [Look closer to see better: Recurrent attention convolutional neural network for fine-grained image recognition](https://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Look_Closer_to_CVPR_2017_paper.pdf) | CVPR17 Oral | [RA-CNN](https://github.com/Jianlong-Fu/Recurrent-Attention-CNN) | 细粒度识别                                                   |                                                 |
| [Guided Attention Network for Object Detection and Counting on Drones](https://arxiv.org/abs/1909.11307v1) | ACM MM20    | [GANet](https://isrc.iscas.ac.cn/gitlab/research/ganet)      | 处理目标检测问题                                             |                                                 |
| [Attention Augmented Convolutional Networks](https://arxiv.org/abs/1904.09925) | ICCV19      | [AANet](https://github.com/leaderj1001/Attention-Augmented-Conv2d) | 多头+引入额外特征映射                                        |                                                 |
| [GLOBAL SELF-ATTENTION NETWORKS FOR IMAGE RECOGNITION](https://arxiv.org/pdf/2010.03019.pdf) | ICLR21      | [GSA](https://github.com/lucidrains/global-self-attention-network) | 新的全局注意力模块                                           |                                                 |
| [Attention-Guided Hierarchical Structure Aggregation for Image Matting](https://ieeexplore.ieee.org/document/9156481) | CVPR20      | [HAttMatting](https://github.com/wukaoliu/CVPR2020-HAttMatting) | 抠图方面的应用，高层使用通道注意力机制，然后再使用空间注意力机制指导低层。 |                                                 |
| [Weight Excitation: Built-in Attention Mechanisms in Convolutional Neural Networks](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750086.pdf) | ECCV20      | None                                                         | 与SE互补的权值激活机制                                       |                                                 |
| [Expectation-Maximization Attention Networks for Semantic Segmentation](https://arxiv.org/pdf/1907.13426.pdf) | ICCV19 Oral | [EMANet](https://github.com/XiaLiPKU/EMANet)                 | EM+Attention                                                 |                                                 |
| [Dense-and-implicit attention network](https://arxiv.org/abs/1905.10671) | AAAI 20     | [DIANet](https://github.com/gbup-group/DIANet)               | LSTM+全程SE注意力                                            |                                                 |
| [Coordinate Attention for Efficient Mobile Network Design](https://arxiv.org/abs/2103.02907) | CVPR21      | [CoordAttention](https://github.com/Andrew-Qibin/CoordAttention) | 横向、竖向                                                   |                                                 |
| [Cross-channel Communication Networks](https://papers.nips.cc/paper/8411-cross-channel-communication-networks.pdf) | NIPS19      | [C3Net](https://github.com/jwyang/C3Net.pytorch)             | GNN+SE                                                       |                                                 |
| [Gated Convolutional Networks with Hybrid Connectivity for Image Classification](https://arxiv.org/pdf/1908.09699.pdf) | AAAI20      | [HCGNet](https://github.com/winycg/HCGNet)                   | 引入了LSTM的部分概念                                         |                                                 |
| [Weighted Channel Dropout for Regularization of Deep Convolutional Neural Network](http://home.ustc.edu.cn/~saihui/papers/aaai2019_weighted.pdf) | AAAI19      | None                                                         | Dropout+SE                                                   |                                                 |
| [BA^2M: A Batch Aware Attention Module for Image Classification](https://arxiv.org/pdf/2103.15099.pdf) | CVPR21      | None                                                         | Batch之间建立attention                                       |                                                 |
| [EPSANet：An Efficient Pyramid Split Attention Block on Convolutional Neural Network](https://arxiv.org/abs/2105.14447) | CoRR21      | [EPSANet](https://github.com/murufeng/EPSANet)               | 多尺度                                                       |                                                 |
| [Stand-Alone Self-Attention in Vision Models](https://arxiv.org/pdf/1906.05909.pdf) | NIPS19      | [SASA](https://github.com/leaderj1001/Stand-Alone-Self-Attention) | Non-Local变体                                                |                                                 |
| [ResT: An Efficient Transformer for Visual Recognition](https://arxiv.org/pdf/2105.13677.pdf) | CoRR21      | [ResT](https://github.com/wofmanaf/ResT)                     | self-attention变体                                           |                                                 |



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

基于CIFAR10+ResNet+待测评模块，对模块进行初步测评。测评代码来自于另外一个库：https://github.com/kuangliu/pytorch-cifar/  实验过程中，不使用预训练权重，进行随机初始化。

| 模型         | top1 acc | time    | params(MB) |
| ------------ | -------- | ------- | ---------- |
| SENet18      | 95.28%   | 1:27:50 | 11,260,354 |
| ResNet18     | 95.16%   | 1:13:03 | 11,173,962 |
| ResNet50     | 95.50%   | 4:24:38 | 23,520,842 |
| ShuffleNetV2 | 91.90%   | 1:02:50 | 1,263,854  |
| GoogLeNet    | 91.90%   | 1:02:50 | 6,166,250  |
| MobileNetV2  | 92.66%   | 2:04:57 | 2,296,922  |
| SA-ResNet50  | 89.83%   | 2:10:07 | 23,528,758 |
| SA-ResNet18  | 95.07%   | 1:39:38 | 11,171,394 |



## Paper List

SENet 论文: https://arxiv.org/abs/1709.01507 解读：https://zhuanlan.zhihu.com/p/102035721

[参考](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247505490&idx=3&sn=475974048f38df8aaa190ec5e8b4af67&chksm=ec1c3babdb6bb2bda2e54dc668e732596b8e7952940d4e4caa69cb683f5d625a0188ce361e2d&scene=21#wechat_redirect)

[ATPaper库](https://github.com/ZhengZixiang/ATPapers)

## Contribute

欢迎在issue中提出补充的文章paper和对应code链接。