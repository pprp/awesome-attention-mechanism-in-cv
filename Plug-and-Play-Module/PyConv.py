# arxiv: https://arxiv.org/abs/2006.11538
# source: https://github.com/iduta/pyconv/blob/master/models/pyconvresnet.py
import torch
import torch.nn as nn
import os


class PyConv2d(nn.Module):
    """PyConv2d with padding (general case). Applies a 2D PyConv over an input signal composed of several input planes.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (list): Number of channels for each pyramid level produced by the convolution
        pyconv_kernels (list): Spatial size of the kernel for each pyramid level
        pyconv_groups (list): Number of blocked connections from input channels to output channels for each pyramid level
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``
    Example::
        >>> # PyConv with two pyramid levels, kernels: 3x3, 5x5
        >>> m = PyConv2d(in_channels=64, out_channels=[32, 32], pyconv_kernels=[3, 5], pyconv_groups=[1, 4])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)
        >>> # PyConv with three pyramid levels, kernels: 3x3, 5x5, 7x7
        >>> m = PyConv2d(in_channels=64, out_channels=[16, 16, 32], pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)
    """
    def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, stride=1, dilation=1, bias=False):
        super(PyConv2d, self).__init__()

        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)

        self.pyconv_levels = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels[i] = nn.Conv2d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                                              stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                                              dilation=dilation, bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))

        return torch.cat(out, 1)