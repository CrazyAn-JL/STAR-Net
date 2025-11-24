import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """轻量通道注意力模块 (SE Block 风格)"""
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 4 , 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w

class FrequencyResidualModulationBlock(nn.Module):
    def __init__(self, nc):
        super(FrequencyResidualModulationBlock, self).__init__()
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0),
            ChannelAttention(nc)
        )

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag_1x1 = self.branch_1x1(mag)
        # 使用残差融合 (元素相加)
        mag_out = mag_1x1 + mag  # 加上 mag 的残差连接
        real = mag_out * torch.cos(pha)
        imag = mag_out * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out + x


class AmplitudeNet_skip(nn.Module):
    def __init__(self, nc):
        super(AmplitudeNet_skip, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, nc, 1, 1, 0),
            FrequencyResidualModulationBlock(nc),
        )
        self.conv1 = FrequencyResidualModulationBlock(nc)
        self.conv2 = FrequencyResidualModulationBlock(nc)
        self.conv3 = FrequencyResidualModulationBlock(nc)

        self.conv4 = nn.Sequential(
            FrequencyResidualModulationBlock(nc * 2),
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
        )
        self.conv5 = nn.Sequential(
            FrequencyResidualModulationBlock(nc * 2),
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
        )

        self.conv6 = nn.Sequential(
            FrequencyResidualModulationBlock(nc * 2),
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
        )

        # 输出层
        self.convout = nn.Sequential(
            FrequencyResidualModulationBlock(nc * 2),
            nn.Conv2d(nc * 2, 3, 1, 1, 0),
        )

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x4 = self.conv4(torch.cat((x2, x3), dim=1))
        x5 = self.conv5(torch.cat((x1, x4), dim=1))

        x6 = self.conv6(torch.cat((x3, x5), dim=1))

        xout = self.convout(torch.cat((x0, x6), dim=1))

        return xout

