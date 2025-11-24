import torch
import torch.nn as nn
import torch.nn.functional as F
from net.KAN import KAN_CBAM
from net.transformer_utils import NormDownsample, NormUpsample, LayerNorm
from net.AMPLITUDE import AmplitudeNet_skip
from net.PHAEN import PhaseRefine


# --------------------- Net ---------------------
class Net(nn.Module):
    def __init__(self, channels=[36, 36, 72, 144], norm=False, kan_cbam_repeats=3):
        super(Net, self).__init__()
        [ch1, ch2, ch3, ch4] = channels
        self.AmpNet = nn.Sequential(
            AmplitudeNet_skip(8),
            nn.Sigmoid()
        )
        self.PhaNet = PhaseRefine()
        self.noise_scale = nn.Parameter(torch.tensor(1.0))
        self.NoiseNet = NoiseEstimator()
        self.conv_first = nn.Sequential(
            nn.Conv2d(7, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, 1, 1),
        )
        self.freq_mod0 = CrossDomainGatedModulation(ch1)
        self.freq_mod1 = CrossDomainGatedModulation(ch2)
        self.freq_mod2 = CrossDomainGatedModulation(ch3)
        self.MCON3 = ResidualDenoiseBlock(ch3)
        self.MCON2 = ResidualDenoiseBlock(ch2)
        self.MCON1 = ResidualDenoiseBlock(ch1)
        self.kan_cbam_layers = nn.ModuleList([
            nn.Sequential(
                ResidualDenoiseBlock(ch4),
                KAN_CBAM(ch4, reduction=8, kernel_size=7)
            ) for _ in range(kan_cbam_repeats)
        ])
        self.kan_cbam3 = KAN_CBAM(ch3, reduction=8, kernel_size=7)

        self.kan_cbam2 = KAN_CBAM(ch2, reduction=8, kernel_size=7)

        self.kan_cbam1 = KAN_CBAM(ch1, reduction=8, kernel_size=7)
        # Encoder
        self.E_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.E_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.E_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.E_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        # Decoder
        self.D_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.D_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.D_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.D_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 3, 3, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        image_fft = torch.fft.fft2(x, norm='backward')
        mag_image = torch.abs(image_fft)
        pha_image = torch.angle(image_fft)
        pha_image = self.PhaNet(pha_image)
        curve_amps = self.AmpNet(x)
        safe_curve_amps = torch.clamp(curve_amps, min=0.1)
        mag_image = mag_image / safe_curve_amps
        real_image_enhanced = mag_image * torch.cos(pha_image)
        imag_image_enhanced = mag_image * torch.sin(pha_image)
        img_amp_enhanced = torch.fft.ifft2(torch.complex(real_image_enhanced, imag_image_enhanced), s=(H, W),
                                           norm='backward').real

        noise_map = self.noise_scale * self.NoiseNet(x)
        x_center = self.conv_first(torch.cat((img_amp_enhanced, x, noise_map), dim=1))

        # ---- Encoder 下采样 ----
        enc0 = self.E_block0(x_center)
        enc0 = self.freq_mod0(enc0, img_amp_enhanced, noise_map)
        enc1 = self.E_block1(enc0)
        enc1 = self.freq_mod1(enc1, img_amp_enhanced, noise_map)
        enc2 = self.E_block2(enc1)
        enc2 = self.freq_mod2(enc2, img_amp_enhanced, noise_map)
        enc3 = self.E_block3(enc2)

        # ---- 保存跳跃连接 ----
        skip0 = enc0
        skip1 = enc1
        skip2 = enc2

        # ---- Bottleneck ----
        bottleneck_out = enc3
        for layer in self.kan_cbam_layers:
            bottleneck_out = layer(bottleneck_out)

        # ---- Decoder 上采样 ----
        dec3 = self.D_block3(bottleneck_out, skip2)
        dec3 = self.kan_cbam3(self.MCON3(dec3))

        dec2 = self.D_block2(dec3, skip1)
        dec2 = self.kan_cbam2(self.MCON2(dec2))

        dec1 = self.D_block1(dec2, skip0)
        dec1 = self.kan_cbam1(self.MCON1(dec1))

        dec0 = self.D_block0(dec1)
        return dec0

class NoiseEstimator(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(base, base//2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(base//2, 1, 3, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)
class ResidualDenoiseBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = LayerNorm(ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, groups=ch)
        self.conv2 = nn.Conv2d(ch, ch, 1)
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        return x + self.conv3(self.act(self.conv2(self.conv1(self.norm(x)))))


class CrossDomainGatedModulation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.noise_conv = nn.Conv2d(1, channels, 1, bias=True)
        self.freq_conv = nn.Conv2d(3, channels, 1, bias=True)
        self.spat_conv = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, spat_feat, freq_img, noise_map):
        freq_img = F.interpolate(freq_img, size=spat_feat.shape[-2:], mode='bilinear', align_corners=False)
        noise_map = F.interpolate(noise_map, size=spat_feat.shape[-2:], mode='bilinear', align_corners=False)
        freq_feat = self.freq_conv(freq_img) + self.noise_conv(noise_map)  # 简单相加即可
        spat_feat = self.spat_conv(spat_feat)
        gate = self.gate(freq_feat)
        return spat_feat * gate + spat_feat