import torch.nn as nn

"""
https://zhuanlan.zhihu.com/p/76378871
arxiv: 1804.03821
ExFuse
"""

class SematicEmbbedBlock(nn.Module):
    def __init__(self, high_in_plane, low_in_plane, out_plane):
        super(SematicEmbbedBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(high_in_plane, out_plane, 3, 1, 1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1x1 = nn.Conv2d(low_in_plane, out_plane, 1)

    def forward(self, high_x, low_x):
        high_x = self.upsample(self.conv3x3(high_x))
        low_x = self.conv1x1(low_x)
        return high_x * low_x