import torch
from torch import nn
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, biliner=False):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        # x = self.conv(x)
        return x



class EUnet(nn.Module):
    def __init__(self, n_class, bilinear=False):
        super(EUnet, self).__init__()
        self.bilinear = bilinear

        self.inc = conv_block(3, 32)
        self.inc_ir = conv_block(1, 32)  # *****
        self.inc_d = conv_block(1, 32)  # *****
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.downi1 = Down(32, 64)
        self.downi2 = Down(64, 128)
        self.downi3 = Down(128, 256)
        self.downi4 = Down(256, 512)

        self.downd1 = Down(32,64)
        self.downd2 = Down(64, 128)
        self.downd3 = Down(128, 256)
        self.downd4 = Down(256, 512)

        self.sf1 = ffm(32)  # 输出 16*9 16=dim
        self.sf2 = ffm(64)  # 输出 8*9  8=dim
        self.sf3 = ffm(128)  # 输出 4*9 4=dim
        self.sf4 = ffm(256)  # 输出 2*9 2=dim
        self.sf5 = ffm(512)  # 输出 1*9 1=dim

        self.up1 = Up(512, 256)  # 先上采样，然后cat
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)


        self.u1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.u2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.u3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.outc_2 = OutConv(64, 32)
        self.outc_3 = OutConv(128, 32)
        self.outc_4 = OutConv(256, 32)
        self.up_out = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.conv_up1 = conv_block(512, 256)
        self.conv_up2 = conv_block(256, 128)
        self.conv_up3 = conv_block(128, 64)
        self.conv_up4 = conv_block(64, 32)

        self.cbam_side = CBAM(channel=32)

        self.Conv_1x1 = nn.Conv2d(64, n_class, kernel_size=1, stride=1, padding=0)
        self.Conv_2x2 = nn.Conv2d(3, n_class, kernel_size=1, stride=1, padding=0)


        self.U = nn.Conv2d(32, n_class, kernel_size=1, stride=1, padding=0)


    def forward(self, x, ir, d):
        x_size = x.size()
        x1 = self.inc(x)
        ir1 = self.inc_ir(ir)
        d1 = self.inc_d(d)
        x1_, ir1_, d1_, sf1 = self.sf1(x1, ir1, d1)

        im_arr_x = x.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny_x = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            blurred = cv2.GaussianBlur(im_arr_x[i], (11, 11), 0)
            canny_x[i] = cv2.Canny(blurred, 50,
                                   180)
        canny_x = torch.from_numpy(canny_x).to(device).float()


        im_arr_ir = ir.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny_ir = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            blurred = cv2.GaussianBlur(im_arr_ir[i], (11, 11), 0)
            canny_ir[i] = cv2.Canny(blurred, 10,
                                    30)
        canny_ir = torch.from_numpy(canny_ir).to(device).float()


        im_arr_d = d.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny_d = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            blurred = cv2.GaussianBlur(im_arr_d[i], (11, 11), 0)
            canny_d[i] = cv2.Canny(blurred, 10,
                                   30)
        canny_d = torch.from_numpy(canny_d).to(device).float()


        x2 = self.down1(x1_)
        ir2 = self.downi1(ir1_)
        d2 = self.downd1(d1_)
        x2_, ir2_, d2_, sf2 = self.sf2(x2, ir2, d2)

        x3 = self.down2(x2_)
        ir3 = self.downi2(ir2_)
        d3 = self.downd2(d2_)
        x3_, ir3_, d3_, sf3 = self.sf3(x3, ir3, d3)

        x4 = self.down3(x3_)
        ir4 = self.downi3(ir3_)
        d4 = self.downd3(d3_)
        x4_, ir4_, d4_, sf4 = self.sf4(x4, ir4, d4)

        x5 = self.down4(x4_)
        ir5 = self.downi4(ir4_)
        d5 = self.downd4(d4_)
        x5_, ir5_, d5_, sf5 = self.sf5(x5, ir5, d5)

        D4 = self.up1(sf5)
        D4 = torch.cat((D4, sf4), dim=1)  #
        D4 = self.conv_up1(D4)
        U4 = self.u1(D4)
        U4 = self.outc_4(U4)

        D3 = self.up2(D4)
        D3 = torch.cat((D3, sf3), dim=1)
        D3 = self.conv_up2(D3)
        U3 = self.u2(D3)
        U3 = self.outc_3(U3)

        D2 = self.up3(D3)
        D2 = torch.cat((D2, sf2), dim=1)
        D2 = self.conv_up3(D2)
        U2 = self.u3(D2)
        U2 = self.outc_2(U2)

        D1 = self.up4(D2)
        D1 = torch.cat((D1, sf1), dim=1)
        D1 = self.conv_up4(D1)

        sifm = U2 + U3 + U4 + D1
        sifm = self.cbam_side(sifm)

        eem = canny_x + canny_ir + canny_d

        out = torch.cat((D1,sifm), dim=1)
        out = self.Conv_1x1(out)

        out = torch.cat((out, eem), dim=1)
        out = self.Conv_2x2(out)

        return out



# 首先定义一个包含conv与ReLu的基础卷积类
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True)



