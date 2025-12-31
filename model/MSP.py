import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class MSP(nn.Module):
    def __init__(self, opt):
        super(MSP, self).__init__()
        n_view = opt.angResolution
        self.an = opt.angResolution
        n_feats = 32

        self.head_0, self.head_90 = EnHead1(n_view, n_feats), EnHead1(n_view, n_feats)
        self.body_0, self.body_90 = EnBody(n_view, n_feats), EnBody(n_view, n_feats)
        self.tail_0, self.tail_90 = EnTail(n_view, n_feats), EnTail(n_view, n_feats)

        self.c_head = EnCenterHead1(n_feats, n_view)
        self.c_body_0 = EnCenterBody(n_feats, n_view)
        self.c_body_3 = EnCenterBody(n_feats, n_view, is_tail=True)

    def my_norm(self, x):
        N, an2, c, h, w = x.shape
        lf_avg = torch.mean(x, dim=1, keepdim=False)  # [N, c, h, w]
        gray = 0.2989 * lf_avg[:, 0, :, :] + 0.5870 * lf_avg[:, 1, :, :] + 0.1140 * lf_avg[:, 2, :, :]  # [N, h, w]
        temp = (1 - gray) * gray
        ratio = (h * w) / (2 * torch.sum(temp.reshape(N, -1), dim=1))
        return ratio

    def prepare_data(self, x):
        N, an2, c, h, w = x.shape

        x = x.view(N, self.an, self.an, c, h, w)
        x_0 = x.view(N * self.an, self.an, c, h, w)
        x_0 = x_0.reshape(N * self.an, self.an * c, h, w)

        x_90 = torch.transpose(x, 1, 2)
        x_90 = x_90.reshape(N * self.an, self.an, c, h, w)
        x_90 = x_90.reshape(N * self.an, self.an * c, h, w)
        return x_0, x_90

    def post_process(self, out_0, out_90, x):
        N, an2, c, h, w = x.shape
        # [N*an, 3*an, h, w]
        out_0 = out_0.view(N * self.an, self.an, c, h, w)
        out_0 = out_0.view(N, self.an, self.an, c, h, w)
        out_0 = out_0.view(N, an2, c, h, w)

        out_90 = out_90.view(N * self.an, self.an, c, h, w)
        out_90 = out_90.view(N, self.an, self.an, c, h, w)
        out_90 = torch.transpose(out_90, 1, 2).reshape(N, an2, c, h, w)
        return out_0, out_90

    def forward(self, x):
        b,u,v,c,h,w = x.shape
        x = x.reshape(b,u*v,c,h,w)

        N, an2, c, h, w = x.shape
        ratio = self.my_norm(x).reshape(N, 1, 1, 1, 1).expand_as(x)
        x = x * ratio

        # stage1
        c_feats_0, central_view_0 = self.c_head(x)
        x_0, x_90 = self.prepare_data(x)  # [N*an, an*c, h, w]
        feats_0, head_0 = self.head_0(x_0)
        feats_90, head_90 = self.head_90(x_90)
        head_0, head_90 = self.post_process(head_0, head_90, x)  # [N, an2, c, h, w]
        central_view_0 = central_view_0.reshape(N, an2, c, h, w)
        out1 = (head_0 + head_90 + central_view_0) / 3
        
        # stage2
        c_feats_1, central_view_1 = self.c_body_0(out1, c_feats_0)
        x_0, x_90 = self.prepare_data(out1)
        feats_0, body_0 = self.body_0(feats_0, x_0)
        feats_90, body_90 = self.body_90(feats_90, x_90)
        body_0, body_90 = self.post_process(body_0, body_90, x)
        central_view_1 = central_view_1.reshape(N, an2, c, h, w)
        out2 = (body_0 + body_90 + central_view_1) / 3
        # stage3
        _, central_view_4 = self.c_body_0(out2, c_feats_1)
        x_0, x_90 = self.prepare_data(out2)
        tail_0 = self.tail_0(feats_0, x_0)
        tail_90 = self.tail_90(feats_90, x_90)
        tail_0, tail_90 = self.post_process(tail_0, tail_90, x)
        central_view_4 = central_view_4.reshape(N, an2, c, h, w)
        out = (tail_0 + tail_90 + central_view_4) / 3

        out1 = out1.reshape(b,u,v,c,h,w)
        out2 = out2.reshape(b,u,v,c,h,w)
        out  = out.reshape(b,u,v,c,h,w)

        return out1, out2,  out
    


class EnHead1(nn.Module):
    def __init__(self, n_view=5, n_feats=32):
        super(EnHead1, self).__init__()

        self.down0 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(n_view * 3, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(n_view, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.down = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(n_view * 3, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.conv_low_cat = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResBlock(n_feats * 2, kernel_size=3),
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(n_feats * 2, n_feats, kernel_size=2, stride=2)
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResBlock(n_feats * 2, kernel_size=3),
            nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Sequential(
            nn.Conv2d(n_feats, n_view * 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        _, _, h, w = x.shape
        x_low = self.down0(x)
        # x: [N*an, an*c, h, w]
        feats = self.conv(x)
        down = self.down(feats)

        feats_low = self.conv_1(x_low)
        low_cat = torch.cat([feats_low, down], dim=1)
        low_cat = self.conv_low_cat(low_cat)
        up = self.up(low_cat)
        high_cat = torch.cat([feats, up], dim=1)

        feats = self.conv_last(high_cat)

        out = self.out(feats)
        return feats, out

class EnBody(nn.Module):
    def __init__(self, n_view=5, n_feats=32):
        super(EnBody, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_view * 3, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(n_feats, kernel_size=3),
            ResBlock(n_feats, kernel_size=3),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(n_feats, n_view * 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats_pre, fusion):
        # feats_pre: [N*an, 32, h, w],
        # fusion: [N*an, an*c, h, w]
        feats_fusion = self.encoder(fusion)
        # feats = self.att(feats_pre, feats_fusion)
        feats = torch.cat([feats_pre, feats_fusion], dim=1)
        feats = self.conv(feats)
        out = self.decoder(feats)
        return feats, out


class EnTail(nn.Module):
    def __init__(self, n_view=5, n_feats=32):
        super(EnTail, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(n_view * 3, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(n_feats, kernel_size=3),
            ResBlock(n_feats, kernel_size=3),
            nn.Conv2d(n_feats, n_view * 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )


    def forward(self, feats_pre, fusion):
        feats_fusion = self.encoder(fusion)
        feats = torch.cat([feats_pre, feats_fusion], dim=1)
        # feats = self.att(feats_pre, feats_fusion)
        return self.conv(feats)


class EnCenterHead1(nn.Module):
    def __init__(self, n_feats=32, n_view=5):
        super(EnCenterHead1, self).__init__()

        self.an = n_view
        self.down0 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(9 * 3, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(9 * 3, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(9 * 1, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.conv_low_cat = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResBlock(n_feats * 2, kernel_size=3),
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(n_feats * 2, n_feats, kernel_size=2, stride=2)
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResBlock(n_feats * 2, kernel_size=3),
            nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Sequential(
            nn.Conv2d(n_feats, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def prepare_data(self, lf):
        N, an2, c, h, w = lf.shape
        an = self.an
        x = lf.reshape(N, an * an, c, h * w)

        x = x.reshape(N, an * an, c, h * w)
        x = torch.transpose(x, 1, 3)
        x = x.reshape(N, h * w, c, an, an)
        # x = lf.view(N, an, an, c, h, w)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), value=0)

        x = x.reshape(N, h * w, c, (an + 2) * (an + 2))
        x = torch.transpose(x, 1, 3)
        x = x.reshape(N, (an + 2) * (an + 2), c, h, w)
        x = x.reshape(N, (an + 2), (an + 2), c, h, w)

        x6 = x[:, :-2, :-2].reshape(N, an2, c, h, w)
        x2 = x[:, :-2, 1:-1].reshape(N, an2, c, h, w)
        x8 = x[:, :-2, 2:].reshape(N, an2, c, h, w)
        x4 = x[:, 1:-1, :-2].reshape(N, an2, c, h, w)

        x3 = x[:, 1:-1, 2:].reshape(N, an2, c, h, w)
        x7 = x[:, 2:, :-2].reshape(N, an2, c, h, w)
        x1 = x[:, 2:, 1:-1].reshape(N, an2, c, h, w)
        x5 = x[:, 2:, 2:].reshape(N, an2, c, h, w)
        focal_stack = torch.cat([x6, x2, x8, x4, lf, x3, x7, x1, x5], dim=2)
        return focal_stack

    def forward(self, lf):
        N, an2, c, h, w = lf.shape
        # an = sqrt(an2)
        fs = self.prepare_data(lf)
        x = fs.reshape(N * an2, c * 9, h, w)
        _, _, h, w = x.shape
        x_low = self.down0(x)
        # x: [N*an, an*c, h, w]
        feats = self.conv(x)
        down = self.down(feats)

        feats_low = self.conv_1(x_low)
        low_cat = torch.cat([feats_low, down], dim=1)
        low_cat = self.conv_low_cat(low_cat)
        up = self.up(low_cat)
        high_cat = torch.cat([feats, up], dim=1)

        feats = self.conv_last(high_cat)

        out = self.out(feats)
        return feats, out


class EnCenterBody(nn.Module):
    def __init__(self, n_feats=32, n_view=5, is_tail=False):
        super(EnCenterBody, self).__init__()
        self.an = n_view

        self.encoder = nn.Sequential(
            nn.Conv2d(27, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResBlock(n_feats, kernel_size=3),
            ResBlock(n_feats, kernel_size=3),
        )

        self.docoder = nn.Sequential(
            nn.Conv2d(n_feats, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def prepare_data1(self, lf):
        N, an2, c, h, w = lf.shape
        an = self.an
        device = lf.get_device()
        focal_stack = torch.zeros((N, an, an, c * 9, h, w)).to(device)
        x = lf.reshape(N, an * an, c, h * w)

        x = x.reshape(N, an * an, c, h * w)
        x = torch.transpose(x, 1, 3)
        x = x.reshape(N, h * w, c, an, an)
        # x = lf.view(N, an, an, c, h, w)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), value=0)

        x = x.reshape(N, h * w, c, (an + 2) * (an + 2))
        x = torch.transpose(x, 1, 3)
        x = x.reshape(N, (an + 2) * (an + 2), c, h, w)
        x = x.reshape(N, (an + 2), (an + 2), c, h, w)

        x6 = x[:, :-2, :-2].reshape(N, an2, c, h, w)
        x2 = x[:, :-2, 1:-1].reshape(N, an2, c, h, w)
        x8 = x[:, :-2, 2:].reshape(N, an2, c, h, w)
        x4 = x[:, 1:-1, :-2].reshape(N, an2, c, h, w)

        x3 = x[:, 1:-1, 2:].reshape(N, an2, c, h, w)
        x7 = x[:, 2:, :-2].reshape(N, an2, c, h, w)
        x1 = x[:, 2:, 1:-1].reshape(N, an2, c, h, w)
        x5 = x[:, 2:, 2:].reshape(N, an2, c, h, w)
        focal_stack = torch.cat([x6, x2, x8, x4, lf, x3, x7, x1, x5], dim=2)

        return focal_stack

    def forward(self, lf, feats):
        N, an2, c, h, w = lf.shape

        fs = self.prepare_data1(lf)
        #
        fs = fs.reshape(N * an2, c * 9, h, w)
        x = self.encoder(fs)

        x = torch.cat([x, feats], dim=1)
        x = self.conv(x)
        out = self.docoder(x)
        # out = out + lf

        return x, out


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', stride=1):
        super().__init__()
        self.ActivationLayer = nn.LeakyReLU(inplace=True)
        if activation == 'relu':
            self.ActivationLayer = nn.ReLU(inplace=True)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            self.ActivationLayer,
        )

    def forward(self, x):
        return self.conv_relu(x)


class ConvTranspose2D(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.ActivationLayer = nn.LeakyReLU(inplace=True)
        if activation == 'relu':
            self.ActivationLayer = nn.ReLU(inplace=True)
        self.deconv_relu = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            self.ActivationLayer,
        )

    def forward(self, x):
        return self.deconv_relu(x)


class MaxPooling2D(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.maxpool(x)


class AvgPooling2D(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.avgpool(x)


class Concat(nn.Module):
    def forward(self, x, y):
        _, _, xh, xw = x.size()
        _, _, yh, yw = y.size()
        diffY = xh - yh
        diffX = xw - yw
        y = F.pad(y, (diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2))
        return torch.cat((x, y), dim=1)


def conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    padding = kernel_size // 2 if dilation == 1 else dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, dilation=dilation)
    # return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, dilation=dilation)


class SASBlock(nn.Module):
    def __init__(self, n_view, n_feats):
        super(SASBlock, self).__init__()

        self.an = n_view
        self.relu = nn.ReLU(inplace=True)
        self.spaconv = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1)
        self.angconv = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1)

        # self.spaconv = ResBlock(n_feats=n_feats, kernel_size=3)
        # self.angconv = ResBlock(n_feats=n_feats, kernel_size=3)

    def forward(self, x):
        N, c, h, w = x.shape  # [N*an2,c,h,w]
        N = N // (self.an * self.an)

        out = self.relu(self.spaconv(x))  # [N*an2,c,h,w]
        out = out.reshape(N, self.an * self.an, c, h * w)
        out = torch.transpose(out, 1, 3)
        out = out.reshape(N * h * w, c, self.an, self.an)  # [N*h*w,c,an,an]

        out = self.relu(self.angconv(out))  # [N*h*w,c,an,an]
        out = out.reshape(N, h * w, c, self.an * self.an)
        out = torch.transpose(out, 1, 3)
        out = out.reshape(N * self.an * self.an, c, h, w)  # [N*an2,c,h,w]
        return out


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), dilation=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias, dilation=dilation))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        # self.se = SELayer(n_feats, reduction=8)
        # self.conv_skip = nn.Conv2d(n_feats, n_feats, 1)

    def forward(self, x):
        res = self.body(x)
        # res = self.se(res)
        res += x

        return res


class ResBlock3D(nn.Module):
    def __init__(self, n_feats, kernel_size, dilation=None):
        super(ResBlock3D, self).__init__()
        if dilation:
            padding = dilation
        else:
            padding = tuple((i // 2 for i in kernel_size))
            dilation = (1, 1, 1)

        self.body = nn.Sequential(
            nn.Conv3d(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, bias=True,
                      padding=padding, dilation=dilation),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, bias=True,
                      padding=padding, dilation=dilation)
        )

    def forward(self, x):
        res = self.body(x)
        # print(res.shape)
        res += x
        return res


class AM(nn.Module):
    def __init__(self, n_feats):
        super(AM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=2, dilation=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=3, dilation=3),
            nn.ReLU()
        )
        self.att = ChannelAttention(in_planes=n_feats * 3)
        self.agg = nn.Conv2d(in_channels=n_feats * 3, out_channels=n_feats, kernel_size=1, bias=False)
        self.out = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            # nn.ReLU()
        )

    def forward(self, x):
        fea_1 = self.conv1(x)
        fea_2 = self.conv2(x)
        fea_3 = self.conv3(x)

        feas = torch.cat([fea_1, fea_2, fea_3], dim=1)
        att = self.att(feas)

        feas_ = feas * att
        feas = feas + feas_
        agg = self.agg(feas)

        out = self.out(agg)

        return out + x



class ResASPP(nn.Module):
    def __init__(self, n_feats=32):
        super(ResASPP, self).__init__()
        self.head_1 = nn.Sequential(
            ASPPConv(in_channels=n_feats, out_channels=n_feats, dilation=1),
            ResBlock(n_feats=n_feats, kernel_size=3, dilation=1)
        )
        self.head_2 = nn.Sequential(
            ASPPConv(in_channels=n_feats, out_channels=n_feats, dilation=2),
            ResBlock(n_feats=n_feats, kernel_size=3, dilation=2)
        )
        self.head_3 = nn.Sequential(
            ASPPConv(in_channels=n_feats, out_channels=n_feats, dilation=3),
            ResBlock(n_feats=n_feats, kernel_size=3, dilation=3)
        )

        self.out = nn.Sequential(
            nn.Conv2d(3 * n_feats, n_feats, 1, bias=False),
        )

    def forward(self, x):
        res_list = [self.head_1(x), self.head_2(x), self.head_3(x)]
        res = torch.cat(res_list, dim=1)

        return self.out(res)


class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        if activation == 'relu':
            self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.8)
        self.cbam = CBAM(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.8)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        x1 = self.relu(bn1)
        cbam = self.cbam(x1)
        conv2 = self.conv2(cbam)
        bn2 = self.bn1(conv2)
        out = bn2 + x
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        # maxout = self.sharedMLP(self.max_pool(x))
        # return self.sigmoid(avgout + maxout)
        return self.sigmoid(avgout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        out = self.sa(x) * x
        return out


def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    downscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]
    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c, 1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)


class PixelUnShuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        downscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''
        return pixel_unshuffle(input, self.downscale_factor)


class MSASBlock(nn.Module):
    def __init__(self, n_view, n_feats=32):
        super(MSASBlock, self).__init__()

        self.an = n_view
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.down = PixelUnShuffle(2)

        self.angular_conv = nn.Sequential(
            nn.Conv2d(in_channels=n_feats*4, out_channels=n_feats*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up = nn.PixelShuffle(2)

        # self.spaconv = ResBlock(n_feats=n_feats, kernel_size=3)
        # self.angconv = ResBlock(n_feats=n_feats, kernel_size=3)

    def forward(self, x):
        N, c, h, w = x.shape  # [N*an2,c,h,w]
        N = N // (self.an * self.an)

        out = self.spatial_conv(x)  # [N*an2,c,h,w]
        out = self.down(out)
        out = out.reshape(N, self.an * self.an, c * 4, h * w // 4)

        out = torch.transpose(out, 1, 3)
        out = out.reshape(N * h * w // 4, c * 4, self.an, self.an)  # [N*h*w,c,an,an]

        out = self.angular_conv(out)  # [N*h*w,c,an,an]
        out = out.reshape(N, h * w // 4, c * 4, self.an * self.an)
        out = torch.transpose(out, 1, 3)
        out = out.reshape(N * self.an * self.an, c * 4, h * w // 4)  # [N*an2,c,h,w]
        out = out.reshape(N * self.an * self.an, c * 4, h // 2, w // 2)  # [N*an2,c,h,w]
        out = self.up(out)

        return out



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)