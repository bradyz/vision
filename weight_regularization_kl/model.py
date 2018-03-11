import torch
import torch.nn as nn


def conv(c_in, c_out, k, s, p, scope):
    result = nn.Conv2d(
            c_in, c_out, kernel_size=k,
            stride=s, padding=p, bias=False)

    scope.append(list(result.parameters())[0])

    return result


def conv3x3(c_in, c_out, stride=1, pad=1, scope=None):
    return conv(c_in, c_out, 3, stride, pad, scope)


def conv3x3_down2x(c_in, c_out, scope=None):
    return conv(c_in, c_out, 3, 2, 1, scope)


def conv1x1(c_in, c_out, stride=1, pad=0, scope=None):
    return conv(c_in, c_out, 1, stride, pad, scope)


def conv1x1_down2x(c_in, c_out, scope=None):
    return conv(c_in, c_out, 1, 2, 0, scope)


def fc(c_in, c_out, scope):
    result = nn.Linear(c_in, c_out)

    scope.append(list(result.parameters())[0])

    return result


def ConvBlock(c_in, c_out, scope, downsample=False):
    if downsample:
        layer = conv3x3_down2x
    else:
        layer = conv3x3

    return nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.LeakyReLU(),
            layer(c_in, c_out, scope=scope),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(),
            conv3x3(c_out, c_out, scope=scope))


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.squeeze(x)


class ResNetBlock(nn.Module):
    def __init__(self, c_in, c_out, scope, downsample=False):
        super().__init__()

        self.conv = ConvBlock(c_in, c_out, scope, downsample)

        if downsample:
            self.skip = conv1x1_down2x(c_in, c_out, scope=scope)
        elif c_in != c_out:
            self.skip = conv1x1(c_in, c_out, scope=scope)
        else:
            self.skip = None

    def forward(self, x):
        if self.skip is not None:
            return self.conv(x) + self.skip(x)

        return self.conv(x) + x


def UpBlock(c_in, c_out, scope):
    return nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(c_in, c_out, scope))


def DownBlock(c_in, c_out, scope):
    return nn.Sequential(
            nn.AvgPool2d(2),
            ConvBlock(c_in, c_out, scope))


def BasicNetwork(c_in, num_classes, scope):
    return nn.Sequential(
            conv3x3(c_in, 64, scope=scope),

            ResNetBlock(64, 64, scope),
            ResNetBlock(64, 64, scope),

            ResNetBlock(64, 128, scope, True),
            ResNetBlock(128, 128, scope),

            ResNetBlock(128, 256, scope, True),
            ResNetBlock(256, 256, scope),

            ResNetBlock(256, 512, scope, True),
            ResNetBlock(512, 512, scope),

            nn.AdaptiveAvgPool2d(1),
            Squeeze(),
            nn.Dropout(0.5),
            fc(512, num_classes, scope))
