import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups),
                               channel // (2 * groups))  # 128/(2x4)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w) 

        return x

    def forward(self, x):
        b, c, h, w = x.shape  # [3, 128, 64, 64]

        x = x.reshape(b * self.groups, -1, h, w)  # [3x4, 32, 64,64]

        x_0, x_1 = x.chunk(2, dim=1)  # 沿着channel切分  [3x4, 16, 64, 64]

        # channel attention using SE
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias  # 线性拟合 wx+b -> channel attention
        xn = x_0 * self.sigmoid(xn)  # element-wise product

        # spatial attention using Group Norm
        xs = self.gn(x_1)  # 12, 16, 64,64
        xs = self.sweight * xs + self.sbias  # 线性拟合 wx+b -> spatial attention
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1) # 特征融合
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2) # channel shuffle
        return out


if __name__ == '__main__':
    model = sa_layer(128, groups=4)
    in_tensor = torch.zeros([3, 128, 64, 64])
    print(model(in_tensor).shape)
