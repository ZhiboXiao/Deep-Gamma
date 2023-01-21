import torch
from torch import nn
from torch.nn import init
from einops import rearrange, repeat, reduce
import numpy as np


i = 0
j = 5

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x)
        x = x.view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x

def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

class Att(nn.Module):
    def __init__(self, channels, shape=None, out_channels=None, no_spatial=True):
        super(Att, self).__init__()
        self.Channel_Att = Channel_Att(channels)

    def forward(self, x):
        x_out1 = self.Channel_Att(x)

        return x_out1

def conv(in_f, out_f, kernel_size, stride=1, dilation=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        stride = 1
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias, dilation=dilation)
    layers = [x for x in [padder, convolver, downsampler] if x is not None]
    return nn.Sequential(*layers)

def init_weights(net, gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('Network initialized with weights sampled from N(0,0.02).')
    net.apply(init_func)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.encoder = nn.Sequential(*[conv(in_channels, out_channels, 5, 2, bias=True, pad='reflection'),nn.BatchNorm2d(out_channels)
                                       , nn.ReLU()])

    def forward(self, x):
        return self.encoder(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,size):
        super(DecoderBlock, self).__init__()
        self.modules1 = nn.Sequential(*[nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),nn.BatchNorm2d(in_channels+size)])
        self.modules2 = nn.Sequential(*[conv(in_channels+size, out_channels, 5, 1, bias=True, pad='reflection'),nn.BatchNorm2d(out_channels), nn.ReLU()])



    def forward(self, x,y,mode):
        if mode == 0:
            x = x
        elif mode == 1:
            x = torch.cat([x, y], 1)
        else:
            x = x+y
        x = self.modules1(x)
        x = self.modules2(x)
        return x

class Skip(nn.Module):
    def __init__(self, in_channels,size):
        super(Skip, self).__init__()

        self.skip = nn.Sequential(*[conv(in_channels, size, 1, bias=True, pad='reflection'),nn.BatchNorm2d(size),nn.LeakyReLU(0.2, inplace=True)])

    def forward(self, x):
        x = self.skip(x)
        return x



class Wnet(nn.Module):
    def __init__(self, depth=3):
        super(Wnet, self).__init__()
        head_dim = 32
        window_size = 8
        channels = 8
        self.window_size = 8
        self.scale = head_dim ** -0.5
        self.skip_size = [1, 1, 1]
        self.skip = nn.Sequential(
            *[conv(4, 1, 1), conv(4, 1, 1), conv(8, 1, 1)])

        self.encoder = nn.Sequential(
            *[EncoderBlock(depth, 4), EncoderBlock(4, 4), EncoderBlock(4, 8)])

        self.patch_partition = PatchMerging(in_channels=8, out_channels=64,
                                            downscaling_factor=2)
        self.to_qkv = nn.Linear(64, 64 * 2 * 3, bias=False)

        self.relative_indices = get_relative_distances(window_size) + window_size - 1
        self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        self.to_out = nn.Linear(64 * 2, 32)
        self.Relu = nn.ReLU()

        self.decoder = nn.Sequential(
            *[DecoderBlock(8, 4, self.skip_size[0]), DecoderBlock(4, 4, self.skip_size[1]),
              DecoderBlock(4, 4, self.skip_size[2])])

        self.conv_gamma = conv(64, 1, 1, bias=True, pad='reflection')
        # self.conv_beta = conv(64, 1, 1, bias=True, pad='reflection')
        self.pooling = torch.nn.AdaptiveAvgPool2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        output_a = []
        skip_total = []
        for layera, skip_layer in zip(self.encoder, self.skip):
            x = layera(x)
            _, C, W1, H1 = x.shape
            skip = skip_layer(x)
            output_a.append(x)
            skip_total.append(skip)
        skip_total.reverse()



        x1 = self.patch_partition(x)
        # x1 = self.sigmoid(x1)

        b, n_h, n_w, _, h = *x1.shape, 2
        qkv = self.to_qkv(x1).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=8, w_w=8), qkv)

        dots = torch.einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        dots += self.pos_embedding[
            self.relative_indices[:, :, 0].type(torch.long), self.relative_indices[:, :, 1].type(torch.long)]
        gamma_o = dots[:, 1, :, :, :]
        beta_o = dots[:, 0, :, :, :]
        # gamma = self.conv_gamma(gamma)
        # beta = self.conv_gamma(beta)
        gamma = self.sigmoid(gamma_o)
        gamma = self.pooling(gamma)
        gamma = torch.mean(gamma, dim=1, keepdim=True)
        beta = 3 * self.sigmoid(beta_o)
        beta = self.pooling(beta)
        beta = torch.mean(beta, dim=1, keepdim=True)
        # v = self.sigmoid(v)
        # out = torch.einsum('b h w i j, b h w j d -> b h w i d', attn, v)

        # out = beta * torch.pow(v, gamma)
        #
        #
        #
        # out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
        #                 h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        # out = self.to_out(out)
        # out = out.permute(0, 3, 1, 2)
        #
        # x = torch.reshape(out, [1, 8, 128, 128]) + x
        # x = beta * torch.pow(x, gamma)
        # out = self.in_conv(x)
        # out = self.out_conv(out)

        #
        #
        # for idx, [decoder_layer] in enumerate(zip(self.decoder)):
        #     if idx ==0:
        #         decoder_output = decoder_layer(x, skip_total[idx], 1)
        #     else:
        #         _, C, W1, H1 = decoder_output.shape
        #         decoder_output = decoder_layer(decoder_output, skip_total[idx], 1)
        # output1 = self.conv3x3(decoder_output)
        # output1 = output1+input
        # output1 = self.sigmoid(output1)
        return beta, gamma, x, dots


if __name__ == '__main__':
    inputs_a = torch.randn(1, 3, 1024, 1024).cuda()
    model = Wnet(3).cuda()
    out = model(inputs_a)
    print(out)

