import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import warnings

from einops import rearrange
from .function import trunc_normal_, DropPath
from torchvision.ops import DeformConv2d, deform_conv2d
from .transform import *
from compressai.layers import *

class InnTransBlock(nn.Module):
    def __init__(self, inn_dim, trans_dim, head_dim, window_size, drop_path, type='W',rev=False):
        """ SwinTransformer and Conv Block
        """
        super(InnTransBlock, self).__init__()
        self.inn_dim = inn_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']
        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type)
        #self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.inn_dim+self.trans_dim, self.inn_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.coupling_layer = CouplingLayer(self.inn_dim // 3, 5)
        #self.conv_block = ResidualBlock(self.conv_dim, self.conv_dim)
        self.rev = rev

    def forward(self, x):
        trans_x = x.narrow(1, 0, self.trans_dim)
        inn_x = x.narrow(1, self.trans_dim, self.inn_dim )
        if not self.rev:
            inn_x = self.coupling_layer(inn_x)
        else:
            inn_x = self.coupling_layer(inn_x,self.rev)
        trans_x = rearrange(trans_x, 'b c h w -> b h w c')
        trans_x = self.trans_block(trans_x)
        trans_x = rearrange(trans_x, 'b h w c -> b c h w')
        res = self.conv1_2(torch.cat((inn_x, trans_x), dim=1))
        x = x + res
        return x



class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.ln1(self.msa(x)))
        x = x + self.drop_path(self.ln2(self.mlp(x)))
        return x

    class WMSA(nn.Module):
        """ Self-attention module in Swin Transformer
        """

        def __init__(self, input_dim, output_dim, head_dim, window_size, type):
            super(WMSA, self).__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.head_dim = head_dim
            self.scale = self.head_dim ** -0.5
            self.n_heads = input_dim // head_dim
            self.window_size = window_size
            self.type = type
            self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)
            self.relative_position_params = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))

            self.linear = nn.Linear(self.input_dim, self.output_dim)

            trunc_normal_(self.relative_position_params, std=.02)
            self.relative_position_params = torch.nn.Parameter(
                self.relative_position_params.view(2 * window_size - 1, 2 * window_size - 1, self.n_heads).transpose(1,
                                                                                                                     2).transpose(
                    0, 1))

        def generate_mask(self, h, w, p, shift):
            """ generating the mask of SW-MSA
            Args:
                shift: shift parameters in CyclicShift.
            Returns:
                attn_mask: should be (1 1 w p p),
            """
            attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
            if self.type == 'W':
                return attn_mask

            s = p - shift
            attn_mask[-1, :, :s, :, s:, :] = True
            attn_mask[-1, :, s:, :, :s, :] = True
            attn_mask[:, -1, :, :s, :, s:] = True
            attn_mask[:, -1, :, s:, :, :s] = True
            attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
            return attn_mask

        def forward(self, x):
            """ Forward pass of Window Multi-head Self-attention module.
            Args:
                x: input tensor with shape of [b h w c];
                attn_mask: attention mask, fill -inf where the value is True;
            Returns:
                output: tensor shape [b h w c]
            """
            if self.type != 'W': x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)),
                                                dims=(1, 2))
            x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
            h_windows = x.size(1)
            w_windows = x.size(2)
            x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
            qkv = self.embedding_layer(x)
            q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
            sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
            sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
            if self.type != 'W':
                attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
                sim = sim.masked_fill_(attn_mask, float("-inf"))

            probs = nn.functional.softmax(sim, dim=-1)
            output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
            output = rearrange(output, 'h b w p c -> b w p (h c)')
            output = self.linear(output)
            output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

            if self.type != 'W': output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2),
                                                     dims=(1, 2))
            return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]
class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=False, rev=False):
        super(HaarWavelet, self).__init__()
        self.in_channels = int(in_channels)

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
        self.rev = rev

    def forward(self, x):
        if not self.rev:

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

class CouplingLayer(nn.Module):
    def __init__(self, split_len, kernal_size, clamp=1.0):
        super(CouplingLayer, self).__init__()
        self.split_len = split_len
        self.clamp = clamp

        self.G1 = Bottleneck(self.split_len * 2 , self.split_len , kernal_size)
        self.G2 = Bottleneck(self.split_len , self.split_len , kernal_size)
        self.G3 = Bottleneck(self.split_len , self.split_len , kernal_size)
        self.H1 = Bottleneck(self.split_len * 2, self.split_len , kernal_size)
        self.H2 = Bottleneck(self.split_len , self.split_len , kernal_size)
        self.H3 = Bottleneck(self.split_len , self.split_len , kernal_size)
        self.G4 = Bottleneck(self.split_len * 2, self.split_len , kernal_size)
        self.G5 = Bottleneck(self.split_len , self.split_len, kernal_size)
        self.G6 = Bottleneck(self.split_len, self.split_len , kernal_size)
        self.H4 = Bottleneck(self.split_len * 2, self.split_len , kernal_size)
        self.H5 = Bottleneck(self.split_len , self.split_len, kernal_size)
        self.H6 = Bottleneck(self.split_len, self.split_len , kernal_size)

    def forward(self, x, rev=False):
        in_nc = x.size(1)
        x1 = x.narrow(1, 0, self.split_len)
        x2 = x.narrow(1, self.split_len, self.split_len)
        x3 = x.narrow(1, self.split_len * 2, self.split_len )
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

class Bottleneck(nn.Module):
    def __init__(self, inp, oup, kernal_size):
        super(Bottleneck, self).__init__()
        hidden_dim = int(inp * 2)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)
class MSFDM(nn.Module):
    def __init__(self, dim_in, dim, need=False):
        super(MSFDM, self).__init__()
        self.need = need
        if need:
            self.first_conv = nn.Conv2d(dim_in, dim, kernel_size=1)
            self.HaarWavelet = HaarWavelet(dim, grad=False,rev=False)
            self.dim = dim
        else:
            self.HaarWavelet = HaarWavelet(dim_in, grad=False,rev=False)
            self.dim = dim_in
        self.convh1 = nn.Conv2d(dim_in * 3, dim_in, kernel_size=1, stride=1, padding=0, bias=True)
        self.Residual_Block = ResNet(dim_in)
        self.convh2 = nn.Conv2d(dim_in, dim_in * 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.convl = nn.Conv2d(dim_in * 2, dim_in, kernel_size=1, stride=1, padding=0, bias=True)
        self.Residual_Block = ResNet(dim_in)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
    def forward(self, x):
        haar = self.HaarWavelet(x)
        a = haar.narrow(1, 0, self.dim)
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim * 2, self.dim)
        d = haar.narrow(1, self.dim * 3, self.dim)
        high = torch.cat([h, v, d], 1)
        high1=self.convh1(high)
        high2=self.Residual_Block(high1)
        highf=self.convh2(high2)
        x2=self.avgpool(x)
        low1=torch.cat([a, x2], 1)
        low2 = self.convl(low1)
        lowf=self.Residual_Block(low2)

        out = torch.cat((lowf, highf), 1)

        return out

class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out1 = F.gelu(self.conv1(x))
        out2 = F.gelu(self.conv2(out1))
        out2 += x  # Residual connection
        return out2

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""

    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

class TIM(nn.Module):
    def __init__(self, M , N=128 ,config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0, num_slices=5, max_support_slices=5, **kwargs):
        super(TIM, self).__init__()
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        self.in_nc =  M
        self.out_nc = N
        self.operations = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0

        # 1st level
        b = MSFDM(self.in_nc,self.in_nc)
        self.operations.append(b)
        # b = HaarWavelet(self.in_nc,rev=False)
        # self.operations.append(b)
        self.in_nc *= 4
        for i in range(config[0]):
            b = InnTransBlock(
                inn_dim=self.in_nc * 3 // 4,
                trans_dim=self.in_nc // 4,
                head_dim=self.head_dim[0],
                window_size=self.window_size,
                drop_path=dpr[begin + i],
                type = 'W' if i % 2 == 0 else 'SW',
                rev = False
            )
            self.operations.append(b)

        # 2nd level
        b = HaarWavelet(self.in_nc,rev=False)
        self.operations.append(b)
        self.in_nc *= 4
        for i in range(config[0]):
            b=InnTransBlock(
                inn_dim=self.in_nc * 3 // 4,
                trans_dim=self.in_nc // 4,
                head_dim=self.head_dim[1],
                window_size=self.window_size,
                drop_path=dpr[begin + i],
                type='W' if i % 2 == 0 else 'SW',
                rev = False

            )
            self.operations.append(b)

        # 3rd level
        b = HaarWavelet(self.in_nc,rev=False)
        self.operations.append(b)
        self.in_nc *= 4
        for i in range(config[0]):
            b=InnTransBlock(
                inn_dim=self.in_nc * 3 // 4,
                trans_dim=self.in_nc // 4,
                head_dim=self.head_dim[2],
                window_size=self.window_size,
                drop_path=dpr[begin + i],
                type='W' if i % 2 == 0 else 'SW',
                rev = False
            )
            self.operations.append(b)

        # 4st level
        b = conv3x3(self.in_nc, self.in_nc, stride=2)
        self.operations.append(b)

    def forward(self, x):
        for op in self.operations:
            x = op.forward(x)
        b, c, h, w = x.size()
        x = torch.mean(x.view(b, c//self.out_nc, self.out_nc, h, w), dim=1)
        return x

class TIM_rev(nn.Module):
    def __init__(self, M , N=128 ,config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0, num_slices=5, max_support_slices=5, **kwargs):
        super(TIM_rev, self).__init__()
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        self.in_nc =  M
        self.out_nc  = N
        self.operations = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0

        # 1st level
        self.in_nc //= 4
        b = HaarWavelet(self.in_nc,rev = True)
        self.operations.append(b)
        self.in_nc //= 4
        b = HaarWavelet(self.in_nc,rev = True)
        self.operations.append(b)
        b = MSFDM(self.in_nc,self.in_nc)
        self.operations.append(b)
        self.in_nc *= 4
        for i in range(config[0]):
            b = InnTransBlock(
                inn_dim=self.in_nc * 3 // 4,
                trans_dim=self.in_nc  // 4,
                head_dim=self.head_dim[3],
                window_size=self.window_size,
                drop_path=dpr[begin + i],
                type = 'W' if i % 2 == 0 else 'SW',
                rev = True
            )
            self.operations.append(b)


        # 2nd level
        self.in_nc //= 4
        b = HaarWavelet(self.in_nc,rev=True)
        self.operations.append(b)
        self.in_nc //= 4
        b = HaarWavelet(self.in_nc,rev = True)
        self.operations.append(b)
        b = MSFDM(self.in_nc,self.in_nc)
        self.operations.append(b)
        self.in_nc *= 4
        for i in range(config[0]):
            b=InnTransBlock(
                inn_dim=self.in_nc * 3 // 4,
                trans_dim=self.in_nc  // 4,
                head_dim=self.head_dim[4],
                window_size=self.window_size,
                drop_path=dpr[begin + i],
                type='W' if i % 2 == 0 else 'SW',
                rev=True
            )
            self.operations.append(b)

        # 3rd level
        self.in_nc //= 4
        b = HaarWavelet(self.in_nc,rev = True)
        self.operations.append(b)
        self.in_nc //= 4
        b = HaarWavelet(self.in_nc,rev = True)
        self.operations.append(b)
        b = MSFDM(self.in_nc,self.in_nc)
        self.operations.append(b)
        self.in_nc *= 4
        for i in range(config[0]):
            b=InnTransBlock(
                inn_dim=self.in_nc * 3 // 4,
                trans_dim=self.in_nc  // 4,
                head_dim=self.head_dim[5],
                window_size=self.window_size,
                drop_path=dpr[begin + i],
                type='W' if i % 2 == 0 else 'SW',
                rev=True
            )
            self.operations.append(b)

        # 4st level
        self.in_nc //= 4
        b = subpel_conv3x3(self.in_nc * 4, self.in_nc , r=2)
        self.operations.append(b)


    def forward(self, x):
        times = self.in_nc * 256 // self.out_nc
        x = x.repeat(1, times, 1, 1)
        for op in self.operations:
            x = op.forward(x)
        return x
def test_Conv_TransBlock():
    model1 = TIM(M=16)
    model2 = TIM_rev(M=4096)
    x = torch.randn(1, 16, 256, 256)
    output1 = model1(x)
    output2 = model2(output1)
    print("Output1 shape:", output1.shape)
    print("Output2 shape:", output2.shape)

class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out