import torch
import torch.nn as nn


class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels,
                                      in_channels // 2,
                                      kernel_size=1,
                                      bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2,
                                         in_channels,
                                         kernel_size=1,
                                         bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)  # shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z)  # shape: [bs, c/2, 1, 1]
        z = self.Conv_Excitation(z)  # shape: [bs, c, 1, 1]
        z = self.norm(z)
        return U * z.expand_as(U)


if __name__ == "__main__":
    bs, c, h, w = 10, 3, 64, 64
    in_tensor = torch.ones(bs, c, h, w)

    c_se = cSE(c)
    print("in shape:", in_tensor.shape)
    out_tensor = c_se(in_tensor)
    print("out shape:", out_tensor.shape)
