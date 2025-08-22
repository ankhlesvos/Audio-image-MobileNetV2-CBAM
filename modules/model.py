# modules/model.py

import torch
import torch.nn as nn
import sys

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, groups):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, attention_mode=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert attention_mode in ['pre_dw', 'post_dw', None]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, stride=1, groups=1))
        if attention_mode == 'pre_dw':
            layers.append(CBAM(hidden_dim))
        layers.extend([ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim)])
        if attention_mode == 'post_dw':
            layers.append(CBAM(hidden_dim))
        layers.extend([nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MyNet(nn.Module):
    def __init__(self, num_classes=1000, model_config=None, width_mult=1.0, in_channels=1):
        super(MyNet, self).__init__()

        if model_config is None:
            model_config = [
                [1, 16, 1, 1, 0],
                [6, 24, 2, 2, 0],
                [6, 32, 3, 2, 0],
                [6, 64, 4, 2, 0],
                [6, 96, 3, 1, 0],
                [6, 160, 3, 2, 0],
                [6, 320, 1, 1, 0],
            ]
        attn_map = {0: None, 1: 'post_dw', 2: 'pre_dw'}#调整第四个数字以修改注意力参数
        block = InvertedResidual
        stem_output_channel = 32
        last_channel = 1280
        stem_output_channel = int(stem_output_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = []
        features.append(ConvBNReLU(in_channels, stem_output_channel, kernel_size=3, stride=2, groups=1))
        current_channels = stem_output_channel
        for t, c, n, s, attn_code in model_config:
            output_channel = int(c * width_mult)
            attention_mode = attn_map.get(attn_code)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(current_channels, output_channel, stride, expand_ratio=t,
                                      attention_mode=attention_mode))
                current_channels = output_channel
        features.append(ConvBNReLU(current_channels, self.last_channel, kernel_size=1, stride=1, groups=1))
        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 测试
if __name__ == '__main__':
    test_input = torch.randn(2, 1, 80, 301)
    # 测试CBAM版模型
    cbam_config = [
        [1, 16, 1, 1, 0], [6, 24, 2, 2, 0], [6, 32, 3, 2, 2],
        [6, 64, 4, 2, 2], [6, 96, 3, 1, 2], [6, 160, 3, 2, 0],
        [6, 320, 1, 1, 0],
    ]
    model_cbam = MyNet(num_classes=11, in_channels=1, model_config=cbam_config)
    output_cbam = model_cbam(test_input)
    print(f"模型输出形状: {output_cbam.shape}")
    assert output_cbam.shape == (2, 11), "错误"
    print("通过")