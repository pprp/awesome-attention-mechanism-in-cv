import torch
import torch.nn as nn

'''
arxiv: 1708.03979
SSH: Single Stage Headless Face Detector
'''

class Conv3x3BNReLU(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv3x3BNReLU,self).__init__()
        self.conv3x3 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv3x3(x)))


class SSHContextModule(nn.Module):
    def __init__(self, in_channel):
        super(SSHContextModule, self).__init__()
        self.stem = Conv3x3BNReLU(in_channel, in_channel//2)
        self.branch1_conv3x3 = Conv3x3BNReLU(in_channel//2, in_channel//2)
        self.branch2_conv3x3_1 = Conv3x3BNReLU(in_channel//2, in_channel//2)
        self.branch2_conv3x3_2 = Conv3x3BNReLU(in_channel//2, in_channel//2)

    def forward(self, x):
        x = self.stem(x)
        # branch1
        x1 = self.branch1_conv3x3(x)
        # branch2
        x2 = self.branch2_conv3x3_1(x)
        x2 = self.branch2_conv3x3_2(x2)
        # concat
        # print(x1.shape, x2.shape)
        return torch.cat([x1, x2], dim=1)


if __name__ == "__main__":
    in_tensor = torch.zeros((6, 64, 128, 128))
    module = SSHContextModule(64)
    out_tensor = module(in_tensor)
    print(out_tensor.shape)
