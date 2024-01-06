import torch
import torch.nn as nn


# 倒残差结构
class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        # expand_ratio：扩展因子
        hidden_channel = in_channel * expand_ratio  # output: h x w x （tk=hidden_channel)
        # 是否使用shortcut :当stride=1而且输入特征矩阵和输出特征矩阵shape相同时才有shortcut
        self.use_shortcut = (stride == 1 and in_channel == out_channel)

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear) 这里激活函数为linear(y=x)，所以不做任何处理
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


# 将ch调整为divisor最近的的整数倍
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# 执行流程：Conv-->BN-->ReLU6
class ConvBNReLU(nn.Sequential):
    # groups=1:普通卷积；groups = in_channel：DW卷积
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        # kernel_size=3时padding=1，kernel_size=1时padding=0
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class MobileNetV2(nn.Module):
    # alpha:宽度因子，卷积核个数的倍率，也就是输出通道数
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        # 第一层所使用卷积核的个数，_make_divisible：将卷积核个数调整为_make_divisible的整数倍
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        # 查表可得下面的配置
        inverted_residual_setting = [
            # t(扩展因子), c(输出特征矩阵深度channel), n(bottleneck的重复次数), s(步矩，针对第一层，其他为1)
            [1, 16, 1, 1],  # bottleneck1
            [6, 24, 2, 2],  # bottleneck2
            [6, 32, 3, 2],  # bottleneck3
            [6, 64, 4, 2],  # bottleneck4
            [6, 96, 3, 1],  # bottleneck5
            [6, 160, 3, 2],  # bottleneck6
            [6, 320, 1, 1],  # bottleneck7
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            # 调整输出后的channel
            output_channel = _make_divisible(c * alpha, round_nearest)
            # 重复n次bottleneck
            for i in range(n):
                # i=0，为第一层，则strider=s，其余stride=1
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel  # 更新，作为下一层的输入
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化下采样
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

