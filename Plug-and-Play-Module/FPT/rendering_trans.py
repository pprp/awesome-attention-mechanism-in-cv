import torch
from torch import nn
from torch.nn import functional as F

class RenderTrans(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(RenderTrans, self).__init__()
        self.upsample = upsample

        self.conv3x3 = nn.Conv2d(channels_high, channels_high, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_high)

        self.conv1x1 = nn.Conv2d(channels_low, channels_high, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_high)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_low, channels_high, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_high)
        else:
            self.conv_reduction = nn.Conv2d(channels_low, channels_high, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_high)
        self.relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Conv2d(channels_high*2, channels_high, kernel_size=1, padding=0, bias=False)

    def forward(self, x_high, x_low):
        b, c, h, w = x_low.shape
        x_low_gp = nn.AvgPool2d(x_low.shape[2:])(x_low).view(len(x_low), c, 1, 1)
        x_low_gp = self.conv1x1(x_low_gp)
        x_low_gp = self.bn_low(x_low_gp)
        x_low_gp = self.relu(x_low_gp)

        x_high_mask = self.conv3x3(x_high)
        x_high_mask = self.bn_high(x_high_mask)

        x_att = x_high_mask * x_low_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(x_high)) + x_att)
                # self.conv_cat(torch.cat([self.bn_upsample(self.conv_upsample(x_high)), x_att], dim=1))
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(x_high)) + x_att)
                # # self.conv_cat(torch.cat([self.bn_reduction(self.conv_reduction(x_high)), x_att], dim=1))
        return out