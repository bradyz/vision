import torch
import torch.nn as nn


def conv(c_in, c_out, k, s, p, scope):
    result = nn.Conv2d(
            c_in, c_out, kernel_size=k,
            stride=s, padding=p)

    scope.append(list(result.parameters())[0])

    return result


def conv3x3(c_in, c_out, stride=1, pad=1, scope=None):
    return conv(c_in, c_out, 3, stride, pad, scope)


def conv1x1(c_in, c_out, stride=1, pad=0, scope=None):
    return conv(c_in, c_out, 1, stride, pad, scope)


def fc(c_in, c_out, scope):
    result = nn.Linear(c_in, c_out)

    scope.append(list(result.parameters())[0])

    return result


def ConvBlock(c_in, c_out, scope):
    return nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.LeakyReLU(),
            conv3x3(c_in, c_out, scope=scope),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(),
            conv3x3(c_out, c_out, scope=scope))


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.squeeze(x)


class ResNetBlock(nn.Module):
    def __init__(self, channels, scope):
        super().__init__()

        self.conv = ConvBlock(channels, channels, scope)

    def forward(self, x):
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
            conv3x3(c_in, 32, scope=scope),
            DownBlock(32, 64, scope=scope),
            DownBlock(64, 128, scope=scope),
            ResNetBlock(128, scope=scope),
            ResNetBlock(128, scope=scope),
            nn.AdaptiveAvgPool2d(1),
            Squeeze(),
            nn.BatchNorm1d(128),
            fc(128, num_classes, scope))
