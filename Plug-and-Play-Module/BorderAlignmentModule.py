# source: https://github.com/Megvii-BaseDetection/BorderDet/blob/master/cvpods/layers/border_align.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from cvpods import _C


class _BorderAlign(Function):
    @staticmethod
    def forward(ctx, input, boxes, wh, pool_size):
        output = _C.border_align_forward(input, boxes, wh, pool_size)
        ctx.pool_size = pool_size
        ctx.save_for_backward(input, boxes, wh)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        pool_size = ctx.pool_size
        input, boxes, wh = ctx.saved_tensors
        grad_input = _C.border_align_backward(
            grad_output, input, boxes, wh, pool_size)
        return grad_input, None, None, None


border_align = _BorderAlign.apply


class BorderAlign(nn.Module):
    def __init__(self, pool_size):
        super(BorderAlign, self).__init__()
        self.pool_size = pool_size

    def forward(self, feature, boxes):
        feature = feature.contiguous()
        boxes = boxes.contiguous()
        wh = (boxes[:, :, 2:] - boxes[:, :, :2]).contiguous()
        output = border_align(feature, boxes, wh, self.pool_size)
        return output

    def __repr__(self):
        tmpstr = self.__class__.__name__
        return tmpstr