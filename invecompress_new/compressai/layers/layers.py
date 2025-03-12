# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torchvision

from .gdn import GDN


class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args, mask_type="A", **kwargs):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x):
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch, out_ch, r=1):
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


def conv1x1(in_ch, out_ch, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch, upsample=2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out




class DeformableConv2d(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            *,
            offset_groups=1,
            with_mask=False
    ):
        super().__init__()
        assert in_dim % groups == 0, "in_dim±ØÐëÄÜ±»groupsÕû³ý"

        # ¹Ø¼üÐÞ¸´1£º½«ÕûÊý²ÎÊý×ªÎªÔª×é
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)  # ±ØÐë×ªÎª (ph, pw)
        self.dilation = _pair(dilation)

        # ¾í»ýºË²ÎÊý
        self.weight = nn.Parameter(
            torch.Tensor(out_dim, in_dim // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.bias = None

        # ²ÎÊýÉú³ÉÆ÷
        self.with_mask = with_mask
        self.param_generator = nn.Conv2d(
            in_dim,
            2 * offset_groups * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=3,
            stride=self.stride,
            padding=1,
            dilation=self.dilation
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        nn.init.normal_(self.param_generator.weight, mean=0, std=0.01)  # Ð¡³ß¶È³õÊ¼»¯Æ«ÒÆÁ¿Éú³ÉÆ÷
        nn.init.constant_(self.param_generator.bias, 0)

    def forward(self, x):
        # È·±£ÊäÈëºÍÄ£ÐÍÔÚÍ¬Ò»Éè±¸
        device = x.device
        self.weight.data = self.weight.data.to(device)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(device)

        # Éú³ÉÆ«ÒÆÁ¿ºÍÑÚÂë
        params = self.param_generator(x)
        if self.with_mask:
            offset_h, offset_w, mask = torch.chunk(params, 3, dim=1)
            offset = torch.cat([offset_h, offset_w], dim=1)
            mask = torch.sigmoid(mask)
        else:
            offset = params
            mask = None

        # ¹Ø¼üÐÞ¸´3£ºÑéÖ¤²ÎÊý³ß´ç

        return torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,  # ÒÑ×ªÎªÔª×é
            dilation=self.dilation,
            mask=mask
        )
def _pair(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)



class DeformConvNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # ¿É±äÐÎ¾í»ýÄ£¿é
        self.deform_conv = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )

        # Æ«ÒÆÁ¿Éú³ÉÆ÷£¨ÓëÖ÷¾í»ý²ÎÊýÒ»ÖÂ£©
        self.offset_gen = nn.Conv2d(
            out_ch,
            2 * kernel_size * kernel_size,  # 2*K^2
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )

    def forward(self, x):
        x = self.conv(x)
        offset = self.offset_gen(x)
        output = torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.deform_conv.weight,
            bias=self.deform_conv.bias,
            padding=self.deform_conv.padding,
            stride=self.deform_conv.stride
        )
        return output