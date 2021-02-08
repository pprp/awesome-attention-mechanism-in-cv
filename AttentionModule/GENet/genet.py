# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ge_resnext29_8x64d", "ge_resnext29_16x64d"]


class Downblock(nn.Module):
    def __init__(self, channels, kernel_size):
        super(Downblock, self).__init__()

        self.dwconv = nn.Conv2d(
            channels,
            channels,
            groups=channels,
            stride=1,
            kernel_size=kernel_size,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn(x)
        return x


class GEModule(nn.Module):
    def __init__(self, in_planes, out_planes, spatial):
        super(GEModule, self).__init__()
        self.downop = Downblock(out_planes, kernel_size=spatial)

        self.mlp = nn.Sequential(
            nn.Conv2d(
                out_planes, out_planes // 16, kernel_size=1, padding=0, bias=False
            ),
            nn.ReLU(),
            nn.Conv2d(
                out_planes // 16, out_planes, kernel_size=1, padding=0, bias=False
            ),
        )

    def forward(self, x):
        # Down, up, sigmoid
        out = self.downop(x)
        out = self.mlp(out)
        shape_in = out.shape[-1]
        out = F.interpolate(out, shape_in)
        out = torch.sigmoid(out)
        out = x * out
        return out


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        spatial,
        cardinality,
        base_width,
        expansion,
    ):

        super(Bottleneck, self).__init__()
        width_ratio = out_channels / (expansion * 64.0)
        D = cardinality * int(base_width * width_ratio)

        self.relu = nn.ReLU(inplace=True)
        self.ge_module = GEModule(in_channels, out_channels, spatial)

        self.conv_reduce = nn.Conv2d(
            in_channels, D, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(
            D,
            D,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(
            D, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                "shortcut_conv",
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
            )
            self.shortcut.add_module("shortcut_bn", nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.conv_reduce.forward(x)
        out = self.relu(self.bn_reduce.forward(out))
        out = self.conv_conv.forward(out)
        out = self.relu(self.bn.forward(out))
        out = self.conv_expand.forward(out)
        out = self.bn_expand.forward(out)

        residual = self.shortcut.forward(x)

        out = self.ge_module(out) + residual
        out = self.relu(out)
        return out


class GeResNeXt(nn.Module):
    def __init__(self, cardinality, depth, num_classes, base_width, expansion=4):
        super(GeResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.expansion = expansion
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [
            64,
            64 * self.expansion,
            128 * self.expansion,
            256 * self.expansion,
        ]

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block("stage_1", self.stages[0], self.stages[1], 32, 1)
        self.stage_2 = self.block("stage_2", self.stages[1], self.stages[2], 16, 2)
        self.stage_3 = self.block("stage_3", self.stages[2], self.stages[3], 8, 2)
        self.fc = nn.Linear(self.stages[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def block(self, name, in_channels, out_channels, spatial, pool_stride=2):
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = "%s_bottleneck_%d" % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(
                    name_,
                    Bottleneck(
                        in_channels,
                        out_channels,
                        pool_stride,
                        spatial,
                        self.cardinality,
                        self.base_width,
                        self.expansion,
                    ),
                )
            else:
                block.add_module(
                    name_,
                    Bottleneck(
                        out_channels,
                        out_channels,
                        1,
                        spatial,
                        self.cardinality,
                        self.base_width,
                        self.expansion,
                    ),
                )
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.fc(x)


def ge_resnext29_8x64d(num_classes):
    return GeResNeXt(cardinality=8, depth=29, num_classes=num_classes, base_width=64)


def ge_resnext29_16x64d(num_classes):
    return GeResNeXt(cardinality=16, depth=29, num_classes=num_classes, base_width=64)