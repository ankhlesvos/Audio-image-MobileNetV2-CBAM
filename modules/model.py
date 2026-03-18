# modules/model.py

import torch
import torch.nn as nn
import sys

try:
    from thop import profile
except ImportError:
    profile = None

try:
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FrequencyAttention(nn.Module):
    def __init__(self, in_planes, reduction=8):
        super(FrequencyAttention, self).__init__()
        # Pool time axis (W) to 1, keep frequency axis (H) and channels
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, max(1, in_planes // reduction), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_planes // reduction), in_planes, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class AsymmetricConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AsymmetricConv, self).__init__()
        stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
        self.conv = nn.Sequential(
             nn.Conv2d(in_planes, out_planes, kernel_size=(kernel_size, 1), stride=(stride_h, 1), padding=(padding, 0), groups=groups, bias=bias),
             nn.BatchNorm2d(out_planes),
             nn.ReLU6(inplace=True),
             nn.Conv2d(out_planes, out_planes, kernel_size=(1, kernel_size), stride=(1, stride_w), padding=(0, padding), groups=groups, bias=bias),
        )
    def forward(self, x):
        return self.conv(x)

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

class MixConv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, asymmetric=False):
        super(MixConv, self).__init__()
        
        if in_planes < 3:
            self.c1 = in_planes
            self.c2 = 0
            self.c3 = 0
        else:
            self.c1 = in_planes // 3
            self.c2 = in_planes // 3
            self.c3 = in_planes - self.c1 - self.c2
            
        if asymmetric:
            self.conv3 = AsymmetricConv(self.c1, self.c1, kernel_size=3, stride=stride, padding=1, groups=self.c1)
            if self.c2 > 0:
                self.conv5 = AsymmetricConv(self.c2, self.c2, kernel_size=5, stride=stride, padding=2, groups=self.c2)
            if self.c3 > 0:
                self.conv7 = AsymmetricConv(self.c3, self.c3, kernel_size=7, stride=stride, padding=3, groups=self.c3)
        else:
            self.conv3 = ConvBNReLU(self.c1, self.c1, kernel_size=3, stride=stride, groups=self.c1)
            if self.c2 > 0:
                self.conv5 = ConvBNReLU(self.c2, self.c2, kernel_size=5, stride=stride, groups=self.c2)
            if self.c3 > 0:
                self.conv7 = ConvBNReLU(self.c3, self.c3, kernel_size=7, stride=stride, groups=self.c3)

    def forward(self, x):
        if self.c2 == 0:
            return self.conv3(x)
        if self.c3 == 0:
            x1, x2 = torch.split(x, [self.c1, self.c2], dim=1)
            return torch.cat([self.conv3(x1), self.conv5(x2)], dim=1)
            
        x1, x2, x3 = torch.split(x, [self.c1, self.c2, self.c3], dim=1)
        out1 = self.conv3(x1)
        out2 = self.conv5(x2)
        out3 = self.conv7(x3)
        return torch.cat([out1, out2, out3], dim=1)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, attention_mode=None, asymmetric=False, multiscale=False, use_res_connect=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert isinstance(stride, tuple) or stride in [1, 2]
        assert attention_mode in ['pre_dw', 'post_dw', 'se', 'freq', None]
        hidden_dim = int(round(inp * expand_ratio))
        
        # Determine residuals: Only if stride=1, inp=oup, AND manually allowed
        self.use_res_connect = (self.stride == 1 and inp == oup) and use_res_connect
        
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, stride=1, groups=1))
            
        if attention_mode == 'pre_dw':
            layers.append(CBAM(hidden_dim))
            
        # Depthwise Conv (Standard, Asymmetric, or MixConv)
        if multiscale:
             layers.append(MixConv(hidden_dim, hidden_dim, stride=stride, asymmetric=asymmetric))
        elif asymmetric:
             # Using factorized 7x7 conv (7x1 then 1x7)
             padding = (7 - 1) // 2
             layers.append(AsymmetricConv(hidden_dim, hidden_dim, kernel_size=7, stride=stride, padding=padding, groups=hidden_dim, bias=False))
        else:
             layers.append(ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
             
        if attention_mode == 'post_dw':
            layers.append(CBAM(hidden_dim))
        elif attention_mode == 'se':
            layers.append(SELayer(hidden_dim))
        elif attention_mode == 'freq':
            layers.append(FrequencyAttention(hidden_dim))
            
        layers.extend([nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MyNet(nn.Module):
    def __init__(self, num_classes=1000, model_config=None, width_mult=1.0, in_channels=1,
                 asymmetric=False, multiscale=False, force_no_residual=False, audio_mode=False,
                 use_imagenet_pretrain=False, pretrain_mode='backbone_only'):
        super(MyNet, self).__init__()
        self.asymmetric = asymmetric
        self.multiscale = multiscale
        self.force_no_residual = force_no_residual
        self.audio_mode = audio_mode
        self.in_channels = in_channels

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
        
        # M1 config support: Force no residuals if specified
        # Standard MobileNetV2 usually has residuals where possible
        self.use_res_connect = True 
        
        attn_map = {0: None, 1: 'post_dw', 2: 'pre_dw', 3: 'se', 4: 'freq'} # 4 for Frequency Attention
        block = InvertedResidual
        stem_output_channel = 32
        last_channel = 1280
        stem_output_channel = int(stem_output_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = []
        
        stem_stride = (1, 2) if audio_mode else 2
        features.append(ConvBNReLU(in_channels, stem_output_channel, kernel_size=3, stride=stem_stride, groups=1))
        current_channels = stem_output_channel
        for t, c, n, s, attn_code in model_config:
            output_channel = int(c * width_mult)
            attention_mode = attn_map.get(attn_code)
            for i in range(n):
                if i == 0:
                    stride = s
                    if self.audio_mode and stride == 2:
                        stride = (1, 2) # keep H, pool W
                else:
                    stride = 1
                
                # Check for global flags
                asymmetric_flag = getattr(self, 'asymmetric', False)
                multiscale_flag = getattr(self, 'multiscale', False)
                use_res_connect = not getattr(self, 'force_no_residual', False)

                features.append(block(current_channels, output_channel, stride, expand_ratio=t,
                                      attention_mode=attention_mode, asymmetric=asymmetric_flag,
                                      multiscale=multiscale_flag,
                                      use_res_connect=use_res_connect))
                current_channels = output_channel
        features.append(ConvBNReLU(current_channels, self.last_channel, kernel_size=1, stride=1, groups=1))
        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # Cache for feature distillation (populated by get_features / forward)
        self._last_features: torch.Tensor | None = None

        # Load ImageNet pretrained weights if requested
        if use_imagenet_pretrain and in_channels == 3:
            self._load_imagenet_pretrain(pretrain_mode)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        # Cache features for distillation (before classifier)
        self._last_features = x
        x = self.classifier(x)
        return x

    def get_features(self) -> torch.Tensor:
        """Returns the 1280-dim feature vector from the last forward pass
        (after global avg pool, before classifier). Used for feature distillation."""
        if self._last_features is None:
            raise RuntimeError("get_features() called before forward(). Run forward first.")
        return self._last_features

    def _load_imagenet_pretrain(self, mode: str = 'backbone_only'):
        """Load MobileNetV2 ImageNet pretrained weights into matching layers.

        Since the input is 3-channel (Log-Mel + Δ + ΔΔ), the first conv layer's
        weights (3→32) are structurally compatible with ImageNet RGB and can be
        transferred directly.

        Args:
            mode: 'backbone_only' loads only feature layers;
                  'full' also loads classifier (only if num_classes matches 1000).
        """
        if not _HAS_TORCHVISION:
            print("[WARN] torchvision not available — skipping ImageNet pretrain.")
            return

        print("[INFO] Loading ImageNet pretrained MobileNetV2 weights...")
        pretrained = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        pretrained_dict = pretrained.state_dict()
        model_dict = self.state_dict()

        loaded, skipped = 0, 0
        for name, param in pretrained_dict.items():
            # Map torchvision names to our names
            # torchvision: features.0.0.weight → our: features.0.0.weight (same structure)
            if name in model_dict and model_dict[name].shape == param.shape:
                if mode == 'backbone_only' and 'classifier' in name:
                    skipped += 1
                    continue
                model_dict[name] = param
                loaded += 1
            else:
                skipped += 1

        self.load_state_dict(model_dict, strict=False)
        print(f"[INFO] ImageNet pretrain: loaded {loaded} params, skipped {skipped} (shape mismatch or excluded).")

    def profile_model(self, input_size=(1, 80, 301)):
        if profile is None:
            return 0, 0
        device = next(self.parameters()).device
        inputs = torch.randn(1, *input_size).to(device)
        flops, params = profile(self, inputs=(inputs,), verbose=False)
        return flops, params


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
    
    # Test M5 Asymmetric
    model_m5 = MyNet(num_classes=11, in_channels=1, model_config=cbam_config, asymmetric=True)
    output_m5 = model_m5(test_input)
    print(f"M5 输出形状: {output_m5.shape}")

    print("通过")