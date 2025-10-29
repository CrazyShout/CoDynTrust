"""
Class used to downsample features by 3*3 conv
"""
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Double convoltuion
    Args:
        in_channels: input channel num
        out_channels: output channel num
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownsampleConv(nn.Module):
    def __init__(self, config):
        super(DownsampleConv, self).__init__()
        self.layers = nn.ModuleList([])
        input_dim = config['input_dim']

        for (ksize, dim, stride, padding) in zip(config['kernal_size'],
                                                 config['dim'],
                                                 config['stride'],
                                                 config['padding']):
            self.layers.append(DoubleConv(input_dim,
                                          dim,
                                          kernel_size=ksize,
                                          stride=stride,
                                          padding=padding))
            input_dim = dim

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x
    
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AttentionDownsampleConv(nn.Module):
    def __init__(self, config):
        super(AttentionDownsampleConv, self).__init__()
        self.layers = nn.ModuleList([])
        input_dim = config['input_dim']

        # Initial 3x3 convolution for feature fusion
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.se_block = SEBlock(input_dim)

        for (ksize, dim, stride, padding) in zip(config['kernal_size'], config['dim'], config['stride'], config['padding']):
            self.layers.append(DoubleConv(input_dim, dim, kernel_size=ksize, stride=stride, padding=padding))
            input_dim = dim
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.se_block(x)
        for layer in self.layers:
            x = layer(x)
        return x