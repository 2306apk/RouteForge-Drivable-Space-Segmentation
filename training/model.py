import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = ConvBlock(3, 64)
        self.p1 = nn.MaxPool2d(2)

        self.d2 = ConvBlock(64, 128)
        self.p2 = nn.MaxPool2d(2)

        self.bridge = ConvBlock(128, 256)

# Up path
        self.u1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c1 = ConvBlock(256, 128)

        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c2 = ConvBlock(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))

        b = self.bridge(self.p2(d2))

        u1 = self.u1(b)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.c1(u1)

        u2 = self.u2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.c2(u2)

        return torch.sigmoid(self.out(u2))