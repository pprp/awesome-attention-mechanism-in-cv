#https://github.com/ZHANGDONG-NJUST/FPT/blob/ffdbf3de67ba9e811f05c800c64e4ea855cc0dae/lib/modeling/FPT.py
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# from torch.nn import DataParallel  # or your customized DataParallel module
# from sync_batchnorm import SynchronizedBatchNorm1d, patch_replication_callback

from modeling.self_trans import SelfTrans
from modeling.rendering_trans import RenderTrans
from modeling.grounding_trans import GroundTrans
import nn as mynn
from dropblock import DropBlock2D

class FPT(nn.Module):
    def __init__(self, feature_dim, with_norm='none', upsample_method='bilinear'):
        super(FPT, self).__init__()
        self.feature_dim = feature_dim
        assert upsample_method in ['nearest', 'bilinear']
        def interpolate(input):
            return F.interpolate(input, scale_factor=2, mode=upsample_method, align_corners=False if upsample_method == 'bilinear' else None)
        self.fpn_upsample = interpolate
        assert with_norm in ['group_norm', 'batch_norm', 'none']
        if with_norm == 'batch_norm':
            norm = nn.BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(num_channels):
                return nn.GroupNorm(32, num_channels)
            norm = group_norm
        self.st_p5 = SelfTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.st_p4 = SelfTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.st_p3 = SelfTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.st_p2 = SelfTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.gt_p4_p5 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.gt_p3_p4 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.gt_p3_p5 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.gt_p2_p3 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.gt_p2_p4 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.gt_p2_p5 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.rt_p5_p4 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p5_p3 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p5_p2 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p4_p3 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p4_p2 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p3_p2 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        drop_block = DropBlock2D(block_size=3, drop_prob=0.2)
        
        if with_norm != 'none':
            self.fpn_p5_1x1 = nn.Sequential(*[nn.Conv2d(2048, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p4_1x1 = nn.Sequential(*[nn.Conv2d(1024, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p3_1x1 = nn.Sequential(*[nn.Conv2d(512, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p2_1x1 = nn.Sequential(*[nn.Conv2d(256, feature_dim, 1, bias=False), norm(feature_dim)])
            
            self.fpt_p5 = nn.Sequential(*[nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpt_p4 = nn.Sequential(*[nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpt_p3 = nn.Sequential(*[nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpt_p2 = nn.Sequential(*[nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
        else:
            self.fpn_p5_1x1 = nn.Conv2d(2048, feature_dim, 1)
            self.fpn_p4_1x1 = nn.Conv2d(1024, feature_dim, 1)
            self.fpn_p3_1x1 = nn.Conv2d(512, feature_dim, 1)
            self.fpn_p2_1x1 = nn.Conv2d(256, feature_dim, 1)
            
            self.fpt_p5 = nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1)
            self.fpt_p4 = nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1)
            self.fpt_p3 = nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1)
            self.fpt_p2 = nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, res2, res3, res4, res5):
        fpn_p5_1 = self.fpn_p5_1x1(res5)
        fpn_p4_1 = self.fpn_p4_1x1(res4)
        fpn_p3_1 = self.fpn_p3_1x1(res3)
        fpn_p2_1 = self.fpn_p2_1x1(res2)
        fpt_p5_out = torch.cat((self.st_p5(fpn_p5_1), self.rt_p5_p4(fpn_p5_1, fpn_p4_1), 
            self.rt_p5_p3(fpn_p5_1,fpn_p3_1), self.rt_p5_p2(fpn_p5_1,fpn_p2_1), fpn_p5_1), 1)
        fpt_p4_out = torch.cat((self.st_p4(fpn_p4_1), self.rt_p4_p3(fpn_p4_1, fpn_p3_1), 
            self.rt_p4_p2(fpn_p4_1,fpn_p2_1), self.gt_p4_p5(fpn_p4_1,fpn_p5_1), fpn_p4_1), 1)
        fpt_p3_out = torch.cat((self.st_p3(fpn_p3_1), self.rt_p3_p2(fpn_p3_1, fpn_p2_1), 
            self.gt_p3_p4(fpn_p3_1,fpn_p4_1), self.gt_p3_p5(fpn_p3_1,fpn_p5_1), fpn_p3_1), 1)
        fpt_p2_out = torch.cat((self.st_p2(fpn_p2_1), self.gt_p2_p3(fpn_p2_1, fpn_p3_1), 
            self.gt_p2_p4(fpn_p2_1,fpn_p4_1), self.gt_p2_p5(fpn_p2_1,fpn_p5_1), fpn_p2_1), 1)
        fpt_p5 = self.fpt_p5(fpt_p5_out)
        fpt_p4 = self.fpt_p4(fpt_p4_out)
        fpt_p3 = self.fpt_p3(fpt_p3_out)
        fpt_p2 = self.fpt_p2(fpt_p2_out)
        '''
        fpt_p5 = drop_block(self.fpt_p5(fpt_p5_out))
        fpt_p4 = drop_block(self.fpt_p4(fpt_p4_out))
        fpt_p3 = drop_block(self.fpt_p3(fpt_p3_out))
        fpt_p2 = drop_block(self.fpt_p2(fpt_p2_out))
        '''
        return fpt_p2, fpt_p3, fpt_p4, fpt_p5