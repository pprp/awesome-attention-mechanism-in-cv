from torch import nn
import torch.nn.functional as F

'''
https://blog.csdn.net/yiran103/article/details/100063021
'''

class BlazeBlock(nn.Module):
    def __init__(self, inp, oup1, oup2=None, stride=1, kernel_size=5):
        super(BlazeBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_double_block = oup2 is not None
        self.use_pooling = self.stride != 1

        if self.use_double_block:
            self.channel_pad = oup2 - inp
        else:
            self.channel_pad = oup1 - inp

        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=inp, bias=True),
            nn.BatchNorm2d(inp),
            # pw-linear
            nn.Conv2d(inp, oup1, 1, 1, 0, bias=True),
            nn.BatchNorm2d(oup1),
        )
        self.act = nn.ReLU(inplace=True)

        if self.use_double_block:
            self.conv2 = nn.Sequential(
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup1, oup1, kernel_size=kernel_size,
                          stride=1, padding=padding, groups=oup1, bias=True),
                nn.BatchNorm2d(oup1),
                # pw-linear
                nn.Conv2d(oup1, oup2, 1, 1, 0, bias=True),
                nn.BatchNorm2d(oup2),
            )

        if self.use_pooling:
            self.mp = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)

    def forward(self, x):
        h = self.conv1(x)
        if self.use_double_block:
            h = self.conv2(h)

        # skip connection
        if self.use_pooling:
            x = self.mp(x)
        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), 'constant', 0)
        return self.act(h + x)


def initialize(module):
    # original implementation is unknown
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight.data, 1)
        nn.init.constant_(module.bias.data, 0)
