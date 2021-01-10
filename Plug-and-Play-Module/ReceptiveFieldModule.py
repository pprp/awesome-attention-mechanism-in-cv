import torch.nn as nn
import torch 


class BasicConv(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(
            out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):
    '''
    [rfb]
    filters = 128
    stride = 1 or 2
    scale = 1.0
    '''
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes,
                      2 * inter_planes,
                      kernel_size=1,
                      stride=stride),
            BasicConv(2 * inter_planes,
                      2 * inter_planes,
                      kernel_size=3,
                      stride=1,
                      padding=visual,
                      dilation=visual,
                      relu=False))
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes,
                      2 * inter_planes,
                      kernel_size=(3, 3),
                      stride=stride,
                      padding=(1, 1)),
            BasicConv(2 * inter_planes,
                      2 * inter_planes,
                      kernel_size=3,
                      stride=1,
                      padding=visual + 1,
                      dilation=visual + 1,
                      relu=False))
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            BasicConv((inter_planes // 2) * 3,
                      2 * inter_planes,
                      kernel_size=3,
                      stride=stride,
                      padding=1),
            BasicConv(2 * inter_planes,
                      2 * inter_planes,
                      kernel_size=3,
                      stride=1,
                      padding=2 * visual + 1,
                      dilation=2 * visual + 1,
                      relu=False))

        self.ConvLinear = BasicConv(6 * inter_planes,
                                    out_planes,
                                    kernel_size=1,
                                    stride=1,
                                    relu=False)
        self.shortcut = BasicConv(in_planes,
                                  out_planes,
                                  kernel_size=1,
                                  stride=stride,
                                  relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class BasicRFB_small(nn.Module):
    '''
    [rfbs]
    filters = 128
    stride=1 or 2
    scale = 1.0
    '''
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_small, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes,
                      inter_planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      relu=False))
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes,
                      inter_planes,
                      kernel_size=(3, 1),
                      stride=1,
                      padding=(1, 0)),
            BasicConv(inter_planes,
                      inter_planes,
                      kernel_size=3,
                      stride=1,
                      padding=3,
                      dilation=3,
                      relu=False))
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes,
                      inter_planes,
                      kernel_size=(1, 3),
                      stride=stride,
                      padding=(0, 1)),
            BasicConv(inter_planes,
                      inter_planes,
                      kernel_size=3,
                      stride=1,
                      padding=3,
                      dilation=3,
                      relu=False))
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1),
            BasicConv(inter_planes // 2, (inter_planes // 4) * 3,
                      kernel_size=(1, 3),
                      stride=1,
                      padding=(0, 1)),
            BasicConv((inter_planes // 4) * 3,
                      inter_planes,
                      kernel_size=(3, 1),
                      stride=stride,
                      padding=(1, 0)),
            BasicConv(inter_planes,
                      inter_planes,
                      kernel_size=3,
                      stride=1,
                      padding=5,
                      dilation=5,
                      relu=False))

        self.ConvLinear = BasicConv(4 * inter_planes,
                                    out_planes,
                                    kernel_size=1,
                                    stride=1,
                                    relu=False)
        self.shortcut = BasicConv(in_planes,
                                  out_planes,
                                  kernel_size=1,
                                  stride=stride,
                                  relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out
