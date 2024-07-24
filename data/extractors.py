# from .resnet import *
#
#
# # 返回提取器以及提取的中间输出通道数
# def build_extractor(c):
#     if c.extractor == 'resnet18':
#         extractor = resnet18(pretrained=True, progress=True)
#     elif c.extractor == 'resnet34':
#         extractor = resnet34(pretrained=True, progress=True)
#     elif c.extractor == 'resnet50':
#         extractor = resnet50(pretrained=True, progress=True)
#     elif c.extractor == 'resnext50_32x4d':
#         extractor = resnext50_32x4d(pretrained=True, progress=True)
#     elif c.extractor == 'wide_resnet50_2':
#         extractor = wide_resnet50_2(pretrained=True, progress=True)
#
#     output_channels = []
#     if 'wide' in c.extractor:
#         for i in range(3):
#             output_channels.append(eval('extractor.layer{}[-1].conv3.out_channels'.format(i + 1)))
#     else:
#         for i in range(3):
#             output_channels.append(extractor.eval('layer{}'.format(i + 1))[-1].conv2.out_channels)
#
#     print("Channels of extracted features:", output_channels)
#     return extractor, output_channels

from efficientnet_pytorch import EfficientNet
from torch import nn
class FeatureExtractor(nn.Module):
    def __init__(self, layer_idx=35):
        super(FeatureExtractor, self).__init__()
        # 使用efficientnet-b5作为特征提取器
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b5')
        self.layer_idx = layer_idx

    def forward(self, x):
        x = self.feature_extractor._swish(self.feature_extractor._bn0(self.feature_extractor._conv_stem(x)))
        # Blocks
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == self.layer_idx:
                return x