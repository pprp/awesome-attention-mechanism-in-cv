import torch
from torch import nn
from torch.nn import functional as F

class SelfTrans(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='dot', dimension=2, bn_layer=True):
    # def __init__(self, in_channels, inter_channels=None, mode='dot', dimension=2, bn_layer=True, n_mix, d_k):
        super(SelfTrans, self).__init__()
        assert dimension in [1, 2, 3]
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`') 
        self.mode = mode
        self.dimension = dimension
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                ) 
    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
    
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
            # f_div_C = MixtureOfSoftMax(n_mix=n_mix, d_k=d_k)

        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) 
            f_div_C = f / N
        y = torch.matmul(f_div_C, g_x)
        
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        z = W_y + x
        return z

class MixtureOfSoftMax(nn.Module):
    def __init__(self, n_mix, d_k, attn_dropout=0.1):
        super(MixtureOfSoftMax, self).__init__()
        self.temperature = np.power(d_k, 0.5)
        self.n_mix = n_mix
        self.att_drop = attn_dropout
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.d_k = d_k
        if n_mix > 1:
            self.weight = nn.Parameter(torch.Tensor(n_mix, d_k))
            std = np.power(n_mix, -0.5)
            self.weight.data.uniform_(-std, std)

    def forward(self, qt, kt, vt):
        B, d_k, N = qt.size()
        m = self.n_mix
        assert d_k == self.d_k
        d = d_k // m
        if m > 1:
            bar_qt = torch.mean(qt, 2, True)
            pi = self.softmax1(torch.matmul(self.weight, bar_qt)).view(B*m, 1, 1)
        q = qt.view(B*m, d, N).transpose(1, 2)
        N2 = kt.size(2)
        kt = kt.view(B*m, d, N2)
        v = vt.transpose(1, 2)
        attn = torch.bmm(q, kt)
        attn = attn / self.temperature
        attn = self.softmax2(attn)
        attn = self.dropout(attn)
        if m > 1:
            attn = (attn * pi).view(B, m, N, N2).sum(1)
        output = torch.bmm(attn, v)
        return output, attn