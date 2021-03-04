# coding=utf-8
# https://github.com/marsggbo/sppnet-pytorch/blob/master/SPP_Layer.py

import math
import torch
import torch.nn.functional as F

# 构建SPP层(空间金字塔池化层)
# Building SPP layer


class SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        # num:样本数量 c:通道数 h:高 w:宽
        # num: the number of samples
        # c: the number of channels
        # h: height
        # w: width
        num, c, h, w = x.size()
        for i in range(self.num_levels):
            level = i+1

            '''
            The equation is explained on the following site:
            http://www.cnblogs.com/marsggbo/p/8572846.html#autoid-0-0-0
            '''
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor(
                (kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(
                    x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = F.avg_pool2d(
                    x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten

# 上面的代码在当数据大小比较小的时候可能会出现下面这种恶心的错误， 即 padding的大小需要小于kernel一半的大小，
# 所以为了解决这个问题，下面代码作了进一步修改，主要方法就是先对数据进行手动更新padding，然后再计算出此时的kernel和stride
# 经测试即使输入数据大小是(7,9), spp_level=4也是正常运行的

# The above code may cause the following nausea error when the data size is relatively small,
# that is, the padding size needs to be less than half the size of the kernel, so in order to solve this problem,
# the following code is further modified, the main method is first to padding the data
# and then the kernel and stride are calculated.
# Tested even if the input data size is (7,9), spp_level=4, the code can run successfully.


class Modified_SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        # num:样本数量 c:通道数 h:高 w:宽
        # num: the number of samples
        # c: the number of channels
        # h: height
        # w: width
        num, c, h, w = x.size()
#         print(x.size())
        for i in range(self.num_levels):
            level = i+1

            '''
            The equation is explained on the following site:
            http://www.cnblogs.com/marsggbo/p/8572846.html#autoid-0-0-0
            '''
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.floor(h / level), math.floor(w / level))
            pooling = (math.floor(
                (kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            # update input data with padding
            zero_pad = torch.nn.ZeroPad2d(
                (pooling[1], pooling[1], pooling[0], pooling[0]))
            x_new = zero_pad(x)

            # update kernel and stride
            h_new = 2*pooling[0] + h
            w_new = 2*pooling[1] + w

            kernel_size = (math.ceil(h_new / level), math.ceil(w_new / level))
            stride = (math.floor(h_new / level), math.floor(w_new / level))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                try:
                    tensor = F.max_pool2d(
                        x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
                except Exception as e:
                    print(str(e))
                    print(x.size())
                    print(level)
            else:
                tensor = F.avg_pool2d(
                    x_new, kernel_size=kernel_size, stride=stride).view(num, -1)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten
