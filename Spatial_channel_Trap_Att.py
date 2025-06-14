import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Flatten


def _is_contiguous(tensor: torch.Tensor) -> bool:
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)

class LayerNorm2d(nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x

# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction),
#             nn.ReLU(),
#             nn.Linear(in_channels // reduction, in_channels),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         avg_out = self.avg_pool(x).view(x.size(0), -1)
#         max_out = self.max_pool(x).view(x.size(0), -1)
#         out = self.fc(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
#         return out
class ChannelAtt(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelAtt, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        global channel_att_raw
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class TrapAttention(nn.Module):
    def __init__(self, channel):
        super(TrapAttention, self).__init__()
        self.depth_wise_conv = nn.Conv2d(channel, channel, kernel_size=(7, 7), groups=channel, padding=3)
        self.attn_conv = nn.Conv2d(channel * 4, channel, kernel_size=3, padding=1, groups=channel)
        self.norm1 = LayerNorm2d(channel)
        self.gamma1 = nn.Parameter(1e-5 * torch.ones(channel)) if 1e-5 > 0 else None
        self.channel_attention = ChannelAtt(channel)
    def trapped_inter(self, x):
        B, C, H, W = x.shape
        mask1 = torch.round(torch.abs(torch.sin(x)))
        mask2 = torch.round(torch.abs(2 * torch.sin(x) * torch.cos(x)))
        mask3 = torch.round(torch.abs(torch.cos(x)))
        mask4 = torch.round(torch.sin(x) ** 2)

        x1 = mask1 * x
        x2 = mask2 * x
        x3 = mask3 * x
        x4 = mask4 * x

        x = torch.cat([x1, x3, x2, x4], dim=1)
        x = x.view(B, 2, 2 * C, H, W)
        x = x.permute(0, 2, 3, 1, 4).flatten(2).contiguous()
        x = x.view(B, 2 * C, H * 2, W)
        x = x.view(B, 2, C, H * 2, W)
        x = x.permute(0, 2, 3, 4, 1).flatten(-1).contiguous()
        x = x.view(B, C, H * 2, W * 2)

        return x

    # def forward(self, feature):
    #     feature_out = self.depth_wise_conv(feature) + feature
    #     shortcut1 = feature_out
    #
    #     pixel_rearrangement = nn.PixelUnshuffle(2)(feature_out)     # B, C*4, h//2, w//2
    #     pixel_rearrangement = self.trapped_inter(pixel_rearrangement)  #
    #     att_feature = self.attn_conv(pixel_rearrangement)
    #     att_feature = self.norm1(att_feature)
    #     att_feature = att_feature.mul(self.gamma1.reshape(1, -1, 1, 1))
    #     att_feature = att_feature + shortcut1
    #     return att_feature

    def forward(self, feature):
        feature_out = self.depth_wise_conv(feature) + feature
        shortcut1 = feature_out
        pixel_rearrangement = nn.PixelUnshuffle(2)(feature_out)
        pixel_rearrangement = self.trapped_inter(pixel_rearrangement)
        att_feature = self.attn_conv(pixel_rearrangement)
        att_feature = self.norm1(att_feature)
        att_feature = att_feature.mul(self.gamma1.reshape(1, -1, 1, 1))

        # Apply channel attention
        channel_att = self.channel_attention(att_feature)
        att_feature = att_feature * channel_att

        att_feature = att_feature + shortcut1
        return att_feature

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels * 4, kernel_size, groups=in_channels, padding=padding,
                                        stride=stride)
        self.pointwise_conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=(1, 1), padding=0)

    def forward(self, feature):
        feature = self.depthwise_conv(feature)
        feature = nn.ReLU(inplace=True)(feature)
        feature = self.pointwise_conv(feature)
        return feature

class TrapAttentionModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TrapAttentionModel, self).__init__()
        self.depthwise_separable_conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=(3, 3))
        self.trap_attention = TrapAttention(out_channels)
        self.depthwise_separable_conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=(3, 3))

    def forward(self, feature):
        feature = self.depthwise_separable_conv1(feature)
        att_feature = self.trap_attention(feature)
        att_feature = self.depthwise_separable_conv2(att_feature)
        return att_feature


if __name__ == '__main__':
    feature = torch.rand(1, 64, 320, 384).cuda()
    model = TrapAttentionModel(in_channels=64, out_channels=64).cuda()
    out = model(feature)
    print(out.shape)
