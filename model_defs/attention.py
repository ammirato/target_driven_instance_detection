import torch
from torch import nn
from torch.nn import functional as F


class CoAttention(nn.Module):
    def __init__(self, in_channels, inter_channels=None, bn_layer=False,
                 combination_mode="sum"):
        assert combination_mode in ["sum", "stack-reduce", "attention-only"]

        super(CoAttention, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_uniform(self.g.weight, a=1)
        nn.init.constant(self.g.bias, 0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            # nn.init.constant(self.W.weight, 0)
            # nn.init.constant(self.W.bias, 0)
            nn.init.kaiming_uniform(self.W.weight, a=1)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_uniform(self.theta.weight, a=1)
        nn.init.constant(self.theta.bias, 0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_uniform(self.phi.weight, a=1)
        nn.init.constant(self.phi.bias, 0)

        self.combination_mode = combination_mode
        if combination_mode == "stack-reduce":
            self.output_reduce = conv_nd(in_channels=2*self.in_channels,
                                         out_channels=self.in_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)

    def forward(self, x, target, mode="average"):
        '''
        :param x: B x C x H x W
        :param y: B x C x h x w
        :return:
        '''

        assert mode in ["softmax", "average"]

        batch_size = x.size(0)

        # embed the object features
        g_target = self.g(target).view(batch_size, self.inter_channels, -1)
        # B x hw x C
        g_target = g_target.permute(0, 2, 1)

        # B x C x HW
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        # B x HW x C
        theta_x = theta_x.permute(0, 2, 1)
        # B x C x hw
        phi_target = self.phi(target).view(batch_size, self.inter_channels, -1)

        # B x HW x hw
        f = torch.matmul(theta_x, phi_target)
        if mode == "softmax":
            f_div_C = F.softmax(f, dim=-1)
        else:
            f_div_C = f / f.size(-1)

        # B x HW x C
        y = torch.matmul(f_div_C, g_target)

        get_norm = lambda z, dim: torch.norm(z, p=2, dim=dim)
        print("="*10)
        print("Input x norm: {}".format(
            get_norm(x, 1).mean().data.cpu().numpy()))
        print("Target norm: {}".format(
            get_norm(target, 1).mean().data.cpu().numpy()))
        print("Embedded target norm: {}/{}/{}".format(
            get_norm(g_target, 2).min().data.cpu().numpy(),
            get_norm(g_target, 2).max().data.cpu().numpy(),
            get_norm(g_target, 2).mean().data.cpu().numpy()))
        print("Min/Max/Mean abs correlation: {}/{}/{}".format(
            f.abs().min().data.cpu().numpy(),
            f.abs().max().data.cpu().numpy(),
            f.abs().mean().data.cpu().numpy()))
        print("Aggregated norm: {}/{}/{}".format(
            get_norm(y, 2).min().data.cpu().numpy(),
            get_norm(y, 2).max().data.cpu().numpy(),
            get_norm(y, 2).mean().data.cpu().numpy()))

        # B x C x HW
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        if self.combination_mode == "stack-reduce":
            z = torch.cat([W_y, x], dim=1)
            z = self.output_reduce(z)
        elif self.combination_mode == "attention-only":
            z = W_y
        else:
            z = W_y + x

        return z


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

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

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch

    sub_sample = True
    bn_layer = True

    img = Variable(torch.zeros(2, 3, 20))
    net = NONLocalBlock1D(3, sub_sample=sub_sample, bn_layer=bn_layer)
    out = net(img)
    print(out.size())

    img = Variable(torch.zeros(2, 3, 20, 20))
    net = NONLocalBlock2D(3, sub_sample=sub_sample, bn_layer=bn_layer)
    out = net(img)
    print(out.size())

    img = Variable(torch.randn(2, 3, 10, 20, 20))
    net = NONLocalBlock3D(3, sub_sample=sub_sample, bn_layer=bn_layer)
    out = net(img)
    print(out.size())

