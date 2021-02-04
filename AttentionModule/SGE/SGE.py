
# https://github.com/implus/PytorchInsight/blob/master/classification/models/imagenet/resnet_sge.py
class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups=64):
        super(SpatialGroupEnhance, self).__init__()
        self.groups = groups # 组个数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = Parameter(torch.ones(1, groups, 1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):  # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)

        xn = x * self.avg_pool(x) # 直接乘

        # 归一化
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)

        # 再学习一组参数
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)

        # 激活
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x
