import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import warnings

from torchvision.ops import DeformConv2d, deform_conv2d

from compressai.layers import *
from compressai.layers import DeformableConv2d


class AttModule(nn.Module):
    def __init__(self, N):
        super(AttModule, self).__init__()
        self.forw_att = AttentionBlock(N)
        self.back_att = AttentionBlock(N)

    def forward(self, x, rev=False):
        if not rev:
            return self.forw_att(x)
        else:
            return self.back_att(x)

class EnhModule(nn.Module):
    def __init__(self, nf):
        super(EnhModule, self).__init__()
        self.forw_enh = EnhBlock(nf)
        self.back_enh = EnhBlock(nf)

    def forward(self, x, rev=False):
        if not rev:
            return self.forw_enh(x)
        else:
            return self.back_enh(x)

class EnhBlock(nn.Module):
    def __init__(self, nf):
        super(EnhBlock, self).__init__()
        self.layers = nn.Sequential(
            DenseBlock(3, nf),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            DenseBlock(nf, 3)
        )

    def forward(self, x):
        return x + self.layers(x) * 0.2

class InvComp(nn.Module):
    def __init__(self, M):
        super(InvComp, self).__init__()
        self.in_nc = 3
        self.out_nc = M
        self.operations = nn.ModuleList()

        # 1st level
        b = InvertibleConv1x1(self.in_nc)
        self.operations.append(b)
        b = HaarWavelet(self.in_nc)
        self.operations.append(b)
        self.in_nc *= 4
        b = CouplingLayer4(self.in_nc // 4, 5)
        self.operations.append(b)
        b = CouplingLayer4(self.in_nc // 4, 5)
        self.operations.append(b)
        b = CouplingLayer4(self.in_nc // 4, 5)
        self.operations.append(b)

        # 2nd level
        b = InvertibleConv1x1(self.in_nc)
        self.operations.append(b)
        b = HaarWavelet(self.in_nc)
        self.operations.append(b)
        self.in_nc *= 4
        b = CouplingLayer4(self.in_nc // 4, 5)
        self.operations.append(b)
        b = CouplingLayer4(self.in_nc // 4, 5)
        self.operations.append(b)
        b = CouplingLayer4(self.in_nc // 4, 5)
        self.operations.append(b)

        # 3rd level
        b = InvertibleConv1x1(self.in_nc)
        self.operations.append(b)
        b = HaarWavelet(self.in_nc)
        self.operations.append(b)
        self.in_nc *= 4
        b = CouplingLayer4(self.in_nc // 4, 3)
        self.operations.append(b)
        b = CouplingLayer4(self.in_nc // 4, 3)
        self.operations.append(b)
        b = CouplingLayer4(self.in_nc // 4, 3)
        self.operations.append(b)

        # 4th level
        b = InvertibleConv1x1(self.in_nc)
        self.operations.append(b)
        b = HaarWavelet(self.in_nc)
        self.operations.append(b)
        self.in_nc *= 4
        b = CouplingLayer4(self.in_nc // 4, 3)
        self.operations.append(b)
        b = CouplingLayer4(self.in_nc // 4, 3)
        self.operations.append(b)
        b = CouplingLayer4(self.in_nc // 4, 3)
        self.operations.append(b)

    def forward(self, x, rev=False):
        if not rev:
            for op in self.operations:
                x = op.forward(x, False)
            b, c, h, w = x.size()
            x = torch.mean(x.view(b, c//self.out_nc, self.out_nc, h, w), dim=1)
        else:
            times = self.in_nc // self.out_nc
            x = x.repeat(1, times, 1, 1)
            for op in reversed(self.operations):
                x = op.forward(x, True)
        return x

class CouplingLayer1(nn.Module):
    def __init__(self, split_len1,split_len2, kernal_size, clamp=1.0):
        super(CouplingLayer1, self).__init__()
        self.split_len1 = split_len1
        self.split_len2 = split_len2
        self.clamp = clamp

        self.G1 = Bottleneck(self.split_len1 * 3, self.split_len1 , kernal_size)
        self.G2 = Bottleneck(self.split_len1 , self.split_len1 , kernal_size)
        self.G3 = Bottleneck(self.split_len1 , self.split_len1 *  2, kernal_size)
        self.H1 = Bottleneck(self.split_len1 * 3, self.split_len1 , kernal_size)
        self.H2 = Bottleneck(self.split_len1 , self.split_len1 , kernal_size)
        self.H3 = Bottleneck(self.split_len1 , self.split_len1 *  2, kernal_size)
        self.G4 = Bottleneck(self.split_len1 * 3, self.split_len1 *  2, kernal_size)
        self.G5 = Bottleneck(self.split_len1 * 2, self.split_len1, kernal_size)
        self.G6 = Bottleneck(self.split_len1, self.split_len1 , kernal_size)
        self.H4 = Bottleneck(self.split_len1 *  3, self.split_len1 *  2, kernal_size)
        self.H5 = Bottleneck(self.split_len1 * 2, self.split_len1, kernal_size)
        self.H6 = Bottleneck(self.split_len1, self.split_len1 , kernal_size)
        self.model = FrequencyDecomposer(self.split_len1)

    def forward(self, x, rev=False):
        in_nc = x.size(1)
        x1 = x.narrow(1, 0, self.split_len1)
        x2 = x.narrow(1, self.split_len1, self.split_len1)
        x3 = x.narrow(1, self.split_len1 * 2, self.split_len1 * 2 )
        x4 = torch.cat([x2, x3], dim=1)
        if not rev:
            y1 = x1.mul(torch.exp( self.clamp * (torch.sigmoid(self.G1(x4)) * 2 - 1) )) + self.H1(x4)
            y2 = x2.mul(torch.exp( self.clamp * (torch.sigmoid(self.G2(y1)) * 2 - 1) )) + self.H2(y1)
            y3 = x3.mul(torch.exp(self.clamp * (torch.sigmoid(self.G3(y2)) * 2 - 1))) + self.H3(y2)
        else:
            y3 = (x3 - self.H4(x4)).div(torch.exp(self.clamp * (torch.sigmoid(self.G4(x4)) * 2 - 1)))
            y2 = (x2 - self.H5(y3)).div(torch.exp(self.clamp * (torch.sigmoid(self.G5(y3)) * 2 - 1)))
            y1 = (x1 - self.H6(y2)).div(torch.exp(self.clamp * (torch.sigmoid(self.G6(y2)) * 2 - 1)))

        return torch.cat((y1, y2, y3), 1)

class CouplingLayer2(nn.Module):
    def __init__(self, split_len1, split_len2, kernal_size, clamp=1.0):
        super(CouplingLayer2, self).__init__()
        self.split_len1 = split_len1
        self.split_len2 = split_len2
        self.clamp = clamp

        self.G1 = Bottleneck(self.split_len1, self.split_len2, kernal_size)
        self.G2 = Bottleneck(self.split_len2, self.split_len1, kernal_size)
        self.H1 = Bottleneck(self.split_len1, self.split_len2, kernal_size)
        self.H2 = Bottleneck(self.split_len2, self.split_len1, kernal_size)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        if not rev:
            y1 = x1.mul(torch.exp( self.clamp * (torch.sigmoid(self.G2(x2)) * 2 - 1) )) + self.H2(x2)
            y2 = x2.mul(torch.exp( self.clamp * (torch.sigmoid(self.G1(y1)) * 2 - 1) )) + self.H1(y1)
        else:
            y2 = (x2 - self.H1(x1)).div(torch.exp( self.clamp * (torch.sigmoid(self.G1(x1)) * 2 - 1) ))
            y1 = (x1 - self.H2(y2)).div(torch.exp( self.clamp * (torch.sigmoid(self.G2(y2)) * 2 - 1) ))
        return torch.cat((y1, y2), 1)





class CouplingLayer3(nn.Module):
    def __init__(self, split_len1,  kernal_size, clamp=1.0):
        super(CouplingLayer3, self).__init__()
        self.split_len1 = split_len1
        self.clamp = clamp

        self.G1 = Bottleneck(self.split_len1 * 11, self.split_len1 , kernal_size)
        self.G2 = Bottleneck(self.split_len1 , self.split_len1 , kernal_size)
        self.G3 = Bottleneck(self.split_len1 , self.split_len1 *  2, kernal_size)
        self.G4 = Bottleneck(self.split_len1 * 2, self.split_len1 *  4, kernal_size)
        self.G5 = Bottleneck(self.split_len1 * 4, self.split_len1 *  4, kernal_size)
        self.G6 = Bottleneck(self.split_len1 * 11, self.split_len1 *  4 , kernal_size)
        self.G7 = Bottleneck(self.split_len1 *  4, self.split_len1 *  4, kernal_size)
        self.G8 = Bottleneck(self.split_len1 *  4, self.split_len1 *  2, kernal_size)
        self.G9 = Bottleneck(self.split_len1 *  2, self.split_len1, kernal_size)
        self.G10 = Bottleneck(self.split_len1, self.split_len1, kernal_size)
        self.H1 = Bottleneck(self.split_len1 * 11, self.split_len1 , kernal_size)
        self.H2 = Bottleneck(self.split_len1 , self.split_len1 , kernal_size)
        self.H3 = Bottleneck(self.split_len1 , self.split_len1 *  2, kernal_size)
        self.H4 = Bottleneck(self.split_len1 * 2, self.split_len1 *  4, kernal_size)
        self.H5 = Bottleneck(self.split_len1 * 4, self.split_len1 *  4, kernal_size)
        self.H6 = Bottleneck(self.split_len1 * 11, self.split_len1 *  4 , kernal_size)
        self.H7 = Bottleneck(self.split_len1 *  4, self.split_len1 *  4, kernal_size)
        self.H8 = Bottleneck(self.split_len1 *  4, self.split_len1 *  2, kernal_size)
        self.H9 = Bottleneck(self.split_len1 *  2, self.split_len1, kernal_size)
        self.H10 = Bottleneck(self.split_len1, self.split_len1, kernal_size)
        self.model = FrequencyDecomposer(self.split_len1)

    def forward(self, x, rev=False):
        in_nc = x.size(1)
        x1 = x.narrow(1, 0, self.split_len1)
        x2 = x.narrow(1, self.split_len1, self.split_len1)
        x3 = x.narrow(1, self.split_len1 * 2, self.split_len1 * 2 )
        x4 = x.narrow(1, self.split_len1 * 4, self.split_len1 * 4 )
        x5 = x.narrow(1, self.split_len1 * 8, self.split_len1 * 4 )
        x6 = torch.cat([x2, x3 ,x4 ,x5], dim=1)
        if not rev:
            y1 = x1.mul(torch.exp( self.clamp * (torch.sigmoid(self.G1(x6)) * 2 - 1) )) + self.H1(x6)
            y2 = x2.mul(torch.exp( self.clamp * (torch.sigmoid(self.G2(y1)) * 2 - 1) )) + self.H2(y1)
            y3 = x3.mul(torch.exp(self.clamp * (torch.sigmoid(self.G3(y2)) * 2 - 1))) + self.H3(y2)
            y4 = x4.mul(torch.exp(self.clamp * (torch.sigmoid(self.G4(y3)) * 2 - 1))) + self.H4(y3)
            y5 = x5.mul(torch.exp(self.clamp * (torch.sigmoid(self.G5(y4)) * 2 - 1))) + self.H5(y4)
        else:
            y5 = (x5 - self.H6(x6)).div(torch.exp(self.clamp * (torch.sigmoid(self.G6(x6)) * 2 - 1)))
            y4 = (x4 - self.H7(y5)).div(torch.exp(self.clamp * (torch.sigmoid(self.G7(y5)) * 2 - 1)))
            y3 = (x3 - self.H8(y4)).div(torch.exp(self.clamp * (torch.sigmoid(self.G8(y4)) * 2 - 1)))
            y2 = (x2 - self.H9(y3)).div(torch.exp(self.clamp * (torch.sigmoid(self.G9(y3)) * 2 - 1)))
            y1 = (x1 - self.H10(y2)).div(torch.exp(self.clamp * (torch.sigmoid(self.G10(y2)) * 2 - 1)))

        return torch.cat((y1, y2, y3, y4, y5), 1)



class CouplingLayer4(nn.Module):
    def __init__(self, split_len,  kernal_size, clamp=1.0):
        super(CouplingLayer4, self).__init__()
        self.split_len = split_len
        self.clamp = clamp

        self.G1 = Bottleneck(self.split_len * 3, self.split_len , kernal_size)
        self.G2 = Bottleneck(self.split_len , self.split_len , kernal_size)
        self.G3 = Bottleneck(self.split_len , self.split_len , kernal_size)
        self.G4 = Bottleneck(self.split_len , self.split_len , kernal_size)
        self.G5 = Bottleneck(self.split_len * 3, self.split_len, kernal_size)
        self.G6 = Bottleneck(self.split_len , self.split_len , kernal_size)
        self.G7 = Bottleneck(self.split_len , self.split_len , kernal_size)
        self.G8 = Bottleneck(self.split_len , self.split_len , kernal_size)
        self.H1 = Bottleneck(self.split_len * 3, self.split_len , kernal_size)
        self.H2 = Bottleneck(self.split_len , self.split_len , kernal_size)
        self.H3 = Bottleneck(self.split_len , self.split_len , kernal_size)
        self.H4 = Bottleneck(self.split_len , self.split_len , kernal_size)
        self.H5 = Bottleneck(self.split_len * 3, self.split_len, kernal_size)
        self.H6 = Bottleneck(self.split_len , self.split_len , kernal_size)
        self.H7 = Bottleneck(self.split_len , self.split_len , kernal_size)
        self.H8 = Bottleneck(self.split_len , self.split_len , kernal_size)
        self.model = FrequencyDecomposer(self.split_len)
        self.HaarWavelet = HaarWavelet(self.split_len * 4, grad=False)
    def forward(self, x, rev=False):
        in_nc = x.size(1)
        haar = self.HaarWavelet(x, rev=False)

        # ·ÖÀë×Ó´ø
        x1 = x.narrow(1, 0, self.split_len)
        x2 = x.narrow(1, self.split_len, self.split_len)
        x3 = x.narrow(1, self.split_len * 2, self.split_len)
        x4 = x.narrow(1, self.split_len * 3, self.split_len)
        x5 = torch.cat([x2, x3, x4], dim=1)
        if not rev:
            y1 = x1.mul(torch.exp( self.clamp * (torch.sigmoid(self.G1(x5)) * 2 - 1) )) + self.H1(x5)
            y2 = x2.mul(torch.exp( self.clamp * (torch.sigmoid(self.G2(y1)) * 2 - 1) )) + self.H2(y1)
            y3 = x3.mul(torch.exp(self.clamp * (torch.sigmoid(self.G3(y2)) * 2 - 1))) + self.H3(y2)
            y4 = x4.mul(torch.exp(self.clamp * (torch.sigmoid(self.G4(y3)) * 2 - 1))) + self.H4(y3)
        else:
            y4 = (x4 - self.H5(x5)).div(torch.exp(self.clamp * (torch.sigmoid(self.G5(x5)) * 2 - 1)))
            y3 = (x3 - self.H6(y4)).div(torch.exp(self.clamp * (torch.sigmoid(self.G6(y4)) * 2 - 1)))
            y2 = (x2 - self.H7(y3)).div(torch.exp(self.clamp * (torch.sigmoid(self.G7(y3)) * 2 - 1)))
            y1 = (x1 - self.H8(y2)).div(torch.exp(self.clamp * (torch.sigmoid(self.G8(y2)) * 2 - 1)))

        return torch.cat((y1, y2, y3, y4), 1)

class FrequencyDecomposer(nn.Module):
    def __init__(self, split_len1, kernel_size=3, padding=1):
        super(FrequencyDecomposer, self).__init__()
        self.split_len1 = split_len1
        self.low_freq_conv_expand = nn.Conv2d(in_channels=split_len1,
                                               out_channels=split_len1 * 4,
                                               kernel_size=1)
        self.low_freq_conv = self._make_depthwise_separable_conv(split_len1 * 4, split_len1, kernel_size, padding)
        self.mid_freq_conv = self._make_depthwise_separable_conv(split_len1 * 4, split_len1, kernel_size, padding)
        self.high_freq_conv = self._make_depthwise_separable_conv(split_len1 * 4, split_len1 * 2, kernel_size, padding)
    def _make_depthwise_separable_conv(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # Pointwise convolution
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("Input tensor must be 4D (batch_size, channels, height, width)")


        # device = next(self.parameters()).device
        # x = x.to(device)
        low_freq = self.low_freq_conv(x)
        low_freq_expand = self.low_freq_conv_expand(low_freq)
        mid_freq = self.mid_freq_conv(x - low_freq_expand)
        mid_freq_expand = self.low_freq_conv_expand(mid_freq)
        high_freq = self.high_freq_conv(x - low_freq_expand - mid_freq_expand)



        return low_freq, mid_freq, high_freq

class Bottleneck1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Bottleneck1, self).__init__()
        padding_standard = (kernel_size - 1) // 2
        dilation = 2
        padding_dilated = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding_standard)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding_standard)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding_dilated, dilation=dilation)
        self.conv5 = nn.Conv2d(out_channels, out_channels, 5, padding=2)
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

        # ³õÊ¼»¯ÐÞÕý£¨ÊÊÅäLeakyReLU£©
        nn.init.kaiming_normal_(self.conv4.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
        nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))
        x4 = self.lrelu(self.conv4(x2))
        x5 = self.lrelu(self.conv5(x2))
        del x2
        x = x3 +x4 + x5
        x = self.fusion(torch.cat([x1, x], dim=1))
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Bottleneck, self).__init__()
        padding_standard = (kernel_size - 1) // 2
        dilation = 2
        padding_dilated = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding_standard)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding_standard)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding_dilated, dilation=dilation)
        self.conv5 = nn.Conv2d(out_channels, out_channels, 5, padding=2)
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

        # ³õÊ¼»¯ÐÞÕý£¨ÊÊÅäLeakyReLU£©
        nn.init.kaiming_normal_(self.conv4.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
        nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))

        # ºÏ²¢¼ÆËã²¢¼õÉÙÖÐ¼ä±äÁ¿
        x = self.lrelu(self.conv3(x2)) + self.lrelu(self.conv4(x2)) + self.lrelu(self.conv5(x2))
        del x2  # ÌáÇ°É¾³ý²»ÔÙÐèÒªµÄÖÐ¼ä±äÁ¿

        x = self.fusion(torch.cat([x1, x], dim=1))
        del x1  # É¾³ýx1ÒÔÊÍ·ÅÄÚ´æ
        return x
class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=False):
        super(HaarWavelet, self).__init__()
        self.in_channels = in_channels

        self.haar_weights = torch.ones(4, 1, 2, 2)
        #h horizontal
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        #v vertical
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        #d diagonal
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = grad

    def forward(self, x, rev=False):
        if not rev:

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape([x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.in_channels)

class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, reverse=False):
        if not reverse:
            output = self.squeeze2d(input, self.factor)  # Squeeze in forward
            return output
        else:
            output = self.unsqueeze2d(input, self.factor)
            return output
        
    def jacobian(self, x, rev=False):
        return 0
        
    @staticmethod
    def squeeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        if factor == 1:
            return input
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
        x = input.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(B, factor * factor * C, H // factor, W // factor)
        return x

    @staticmethod
    def unsqueeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        factor2 = factor ** 2
        if factor == 1:
            return input
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert C % (factor2) == 0, "{}".format(C)
        x = input.view(B, factor, factor, C // factor2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(B, C // (factor2), H * factor, W * factor)
        return x

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not reverse:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            weight = torch.inverse(self.weight.double()).float() \
                .view(w_shape[0], w_shape[1], 1, 1)
        return weight

    def forward(self, input, reverse=False):
        weight = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            return z
        else:
            z = F.conv2d(input, weight)
            return z

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


    def __init__(self, in_shape, int_ch, numTraceSamples=0, numSeriesTerms=0,
                 stride=1, coeff=.97, input_nonlin=True,
                 actnorm=True, n_power_iter=5, nonlin="elu", train=False):
        """
        buid invertible bottleneck block
        :param in_shape: shape of the input (channels, height, width)
        :param int_ch: dimension of intermediate layers
        :param stride: 1 if no downsample 2 if downsample
        :param coeff: desired lipschitz constant
        :param input_nonlin: if true applies a nonlinearity on the input
        :param actnorm: if true uses actnorm like GLOW
        :param n_power_iter: number of iterations for spectral normalization
        :param nonlin: the nonlinearity to use
        """
        super(conv_iresnet_block_simplified, self).__init__()
        assert stride in (1, 2)
        self.stride = stride
        self.squeeze = IRes_Squeeze(stride)
        self.coeff = coeff
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
        self.n_power_iter = n_power_iter
        nonlin = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "softplus": nn.Softplus,
            "sorting": lambda: MaxMinGroup(group_size=2, axis=1)
        }[nonlin]

        # set shapes for spectral norm conv
        in_ch, h, w = in_shape
            
        layers = []
        if input_nonlin:
            layers.append(nonlin())

        in_ch = in_ch * stride**2
        kernel_size1 = 1
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(in_ch, int_ch, kernel_size=kernel_size1, padding=0),
                                                  (in_ch, h, w), kernel_size1))
        layers.append(nonlin())
        kernel_size3 = 1
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(int_ch, in_ch, kernel_size=kernel_size3, padding=0),
                                                  (int_ch, h, w), kernel_size3))
        self.bottleneck_block = nn.Sequential(*layers)
        if actnorm:
            self.actnorm = ActNorm2D(in_ch, train=train)
        else:
            self.actnorm = None

    def forward(self, x, rev=False, ignore_logdet=False, maxIter=25):
        if not rev:
            """ bijective or injective block forward """
            if self.stride == 2:
                x = self.squeeze.forward(x)
            if self.actnorm is not None:
                x, an_logdet = self.actnorm(x)
            else:
                an_logdet = 0.0
            Fx = self.bottleneck_block(x)
            if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
                trace = torch.tensor(0.)
            else:
                trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)
            y = Fx + x
            return y, trace + an_logdet
        else:
            y = x
            for iter_index in range(maxIter):
                summand = self.bottleneck_block(x)
                x = y - summand

            if self.actnorm is not None:
                x = self.actnorm.inverse(x)
            if self.stride == 2:
                x = self.squeeze.inverse(x)
            return x
    
    def _wrapper_spectral_norm(self, layer, shapes, kernel_size):
        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x1 convolutions
            return spectral_norm_fc(layer, self.coeff, 
                                    n_power_iterations=self.n_power_iter)
        else:
            # use spectral norm based on conv, because bound not tight
            return spectral_norm_conv(layer, self.coeff, shapes,
                                      n_power_iterations=self.n_power_iter)