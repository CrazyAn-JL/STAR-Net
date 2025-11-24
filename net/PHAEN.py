import torch
import torch.nn as nn


class PhaseRefine(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.Tanh()  # 相位微调 [-π, π] 范围
        )

    def forward(self, pha):
        return pha + 0.1 * self.refine(pha)

