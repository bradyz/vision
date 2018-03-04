import torch.nn as nn
import torch.nn.functional as F


def conv(c_in, c_out, k, s, p):
    return nn.Conv2d(
            c_in, c_out, kernel_size=k,
            stride=s, padding=p)


def conv3x3(c_in, c_out, stride=1, pad=1):
    return conv(c_in, c_out, 3, stride, pad)


def conv1x1(c_in, c_out, stride=1, pad=0):
    return conv(c_in, c_out, 1, stride, pad)


class ResNetBlock(nn.Module):
    def __init__(self, c_in, c_out, s=1):
        super().__init__()

        self.ln1 = nn.InstanceNorm2d(c_in)
        self.conv1 = conv3x3(c_in, c_out, stride=s)

        self.ln2 = nn.InstanceNorm2d(c_out)
        self.conv2 = conv3x3(c_out, c_out)

        # Makes sure the shapes are the same.
        self.skip = None

        if s != 1 or c_in != c_out:
            self.skip = conv1x1(c_in, c_out, s)

    def forward(self, x):
        out = self.ln1(x)
        out = F.relu(out)

        if self.skip is not None:
            skip = self.skip(out)
        else:
            skip = x

        out = self.conv1(out)
        out = self.ln2(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = out + skip

        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, block_sizes, num_classes):
        super().__init__()

        self.c = 64

        self.conv1 = conv3x3(in_channels, 64)
        self.block1 = self._make_block(64, block_sizes[0], 1)
        self.block2 = self._make_block(128, block_sizes[1], 2)
        self.block3 = self._make_block(256, block_sizes[2], 2)
        self.block4 = self._make_block(512, block_sizes[3], 2)
        self.linear = nn.Linear(512, num_classes)

    def _make_block(self, c_out, block_size, stride):
        subblocks = []

        for s in [stride] + [1] * (block_size-1):
            subblocks.append(ResNetBlock(self.c, c_out, s))

            self.c = c_out

        return nn.Sequential(*subblocks)

    def forward(self, x):
        out = self.conv1(x)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return out


def ResNet18(in_channels, num_classes=10):
    return ResNet(in_channels, [2, 2, 2, 2], num_classes)


def ConvBlock(c_in, c_out):
    return nn.Sequential(
            nn.InstanceNorm2d(c_in),
            nn.LeakyReLU(),
            conv3x3(c_in, c_out),
            nn.InstanceNorm2d(c_out),
            nn.LeakyReLU(),
            conv3x3(c_out, c_out))


def UpBlock(c_in, c_out):
    return nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(c_in, c_out))


def DownBlock(c_in, c_out):
    return nn.Sequential(
            nn.AvgPool2d(2),
            ConvBlock(c_in, c_out))


def ProcessBlock(c_in, c_out):
    return nn.Sequential(
            ResNetBlock(c_in, c_out),
            ResNetBlock(c_out, c_out),
            ResNetBlock(c_out, c_out))


class MaskGenerator(nn.Module):
    def __init__(self, in_channels, k=32):
        super().__init__()

        self.down1 = DownBlock(in_channels, k)
        self.down2 = DownBlock(k * 1, k * 2)
        self.down3 = DownBlock(k * 2, k * 4)
        self.down4 = DownBlock(k * 4, k * 8)
        self.middle = ProcessBlock(k * 8, k * 8)
        self.up1 = UpBlock(k * 8, k * 4)
        self.up2 = UpBlock(k * 4, k * 2)
        self.up3 = UpBlock(k * 2, k * 1)
        self.up4 = UpBlock(k, 1)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        middle = self.middle(down4) + down4
        up1 = self.up1(middle) + down3
        up2 = self.up2(up1) + down2
        up3 = self.up3(up2) + down1
        up4 = self.up4(up3)

        return F.sigmoid(up4)
