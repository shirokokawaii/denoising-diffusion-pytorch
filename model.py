import math
from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from pydantic import StrictInt, StrictFloat, StrictBool


# def swish(input):
#     return input * torch.sigmoid(input)

'''
Unet and its components
'''
swish = F.silu


# tensor value init.
@torch.no_grad()
def variance_scaling_init_(tensor, scale=1, mode="fan_avg", distribution="uniform"):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        scale /= fan_in

    elif mode == "fan_out":
        scale /= fan_out

    else:
        scale /= (fan_in + fan_out) / 2

    if distribution == "normal":
        std = math.sqrt(scale)

        # normal distribution
        return tensor.normal_(0, std)

    else:
        bound = math.sqrt(3 * scale)

        # continuous uniform distribution
        return tensor.uniform_(-bound, bound)

# convolution
def conv2d(
    in_channel,
    out_channel,
    kernel_size,
    stride=1,
    padding=0,
    bias=True,
    scale=1,
    mode="fan_avg",
):
    conv = nn.Conv2d(
        in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias
    )

    # tensor value init
    variance_scaling_init_(conv.weight, scale, mode=mode)

    if bias:
        nn.init.zeros_(conv.bias)

    return conv

# linear layer
def linear(in_channel, out_channel, scale=1, mode="fan_avg"):
    lin = nn.Linear(in_channel, out_channel)

    variance_scaling_init_(lin.weight, scale, mode=mode)
    nn.init.zeros_(lin.bias)

    return lin

# silu
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return swish(input)


class Upsample(nn.Sequential):
    def __init__(self, channel):
        layers = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv2d(channel, channel, 3, padding=1),
        ]

        super().__init__(*layers)


class Downsample(nn.Sequential):
    def __init__(self, channel):
        layers = [conv2d(channel, channel, 3, stride=2, padding=1)]

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(
        self, in_channel, out_channel, time_dim, use_affine_time=False, dropout=0
    ):
        super().__init__()

        # time embedding
        self.use_affine_time = use_affine_time
        time_out_dim = out_channel
        time_scale = 1
        norm_affine = True

        if self.use_affine_time:
            time_out_dim *= 2
            time_scale = 1e-10
            norm_affine = False

        # input convolution
        self.norm1 = nn.GroupNorm(32, in_channel)
        self.activation1 = Swish()
        self.conv1 = conv2d(in_channel, out_channel, 3, padding=1)

        self.time = nn.Sequential(
            Swish(), linear(time_dim, time_out_dim, scale=time_scale)
        )

        # output convolution
        self.norm2 = nn.GroupNorm(32, out_channel, affine=norm_affine)
        self.activation2 = Swish()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv2d(out_channel, out_channel, 3, padding=1, scale=1e-10)

        if in_channel != out_channel:
            self.skip = conv2d(in_channel, out_channel, 1)

        else:
            self.skip = None

    def forward(self, input, time):
        # batch size
        batch = input.shape[0]

        # first convolution
        out = self.conv1(self.activation1(self.norm1(input)))

        # time embedding
        if self.use_affine_time:
            gamma, beta = self.time(time).view(batch, -1, 1, 1).chunk(2, dim=1)
            out = (1 + gamma) * self.norm2(out) + beta

        else:
            out = out + self.time(time).view(batch, -1, 1, 1)
            out = self.norm2(out)

        # output convolution
        out = self.conv2(self.dropout(self.activation2(out)))

        if self.skip is not None:
            input = self.skip(input)

        return out + input

# self-attention block
class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(32, in_channel)
        self.qkv = conv2d(in_channel, in_channel * 3, 1)
        self.out = conv2d(in_channel, in_channel, 1, scale=1e-10)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        # step1
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        # step2
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx
        # step3
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        # step4
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)
        # step5
        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))
        # step6
        return out + input


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        # sinusoidal position embedding
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim)
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)

        return pos_emb


class ResBlockWithAttention(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        time_dim,
        dropout,
        use_attention=False,
        attention_head=1,
        use_affine_time=False,
    ):
        super().__init__()

        self.resblocks = ResBlock(
            in_channel, out_channel, time_dim, use_affine_time, dropout
        )

        if use_attention:
            self.attention = SelfAttention(out_channel, n_head=attention_head)

        else:
            self.attention = None

    def forward(self, input, time):
        out = self.resblocks(input, time)

        if self.attention is not None:
            out = self.attention(out)

        return out

# transpose tensor for input
def spatial_fold(input, fold):
    if fold == 1:
        return input

    batch, channel, height, width = input.shape
    h_fold = height // fold
    w_fold = width // fold

    return (
        input.view(batch, channel, h_fold, fold, w_fold, fold)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(batch, -1, h_fold, w_fold)
    )


# transpose tensor for output
def spatial_unfold(input, unfold):
    if unfold == 1:
        return input

    batch, channel, height, width = input.shape
    h_unfold = height * unfold
    w_unfold = width * unfold

    return (
        input.view(batch, -1, unfold, unfold, height, width)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(batch, -1, h_unfold, w_unfold)
    )


class UNet(nn.Module):
    def __init__(
        self,
        in_channel: StrictInt,
        channel: StrictInt,
        channel_multiplier: List[StrictInt],
        n_res_blocks: StrictInt,
        attn_strides: List[StrictInt],
        attn_heads: StrictInt = 1,
        use_affine_time: StrictBool = False,
        dropout: StrictFloat = 0,
        fold: StrictInt = 1,
    ):
        super().__init__()

        self.fold = fold

        # time embedding channel
        time_dim = channel * 4

        n_block = len(channel_multiplier)

        # time embedding module
        self.time = nn.Sequential(
            TimeEmbedding(channel),
            linear(channel, time_dim),
            Swish(),
            linear(time_dim, time_dim),
        )

        # downsample block init
        down_layers = [conv2d(in_channel * (fold ** 2), channel, 3, padding=1)]
        feat_channels = [channel]
        in_channel = channel

        # for each layer, add resblock and downsample block
        for i in range(n_block):
            for _ in range(n_res_blocks):
                # present channel's number
                channel_mult = channel * channel_multiplier[i]

                down_layers.append(
                    ResBlockWithAttention(
                        in_channel,
                        channel_mult,
                        time_dim,
                        dropout,
                        use_attention=2 ** i in attn_strides,
                        attention_head=attn_heads,
                        use_affine_time=use_affine_time,
                    )
                )

                feat_channels.append(channel_mult)
                in_channel = channel_mult
            
            # ignore downsample in the deepest layer
            if i != n_block - 1:
                down_layers.append(Downsample(in_channel))
                feat_channels.append(in_channel)

        # into a module list
        self.down = nn.ModuleList(down_layers)

        # middle block, two resblocks.
        self.mid = nn.ModuleList(
            [
                ResBlockWithAttention(
                    in_channel,
                    in_channel,
                    time_dim,
                    dropout=dropout,
                    use_attention=True,
                    attention_head=attn_heads,
                    use_affine_time=use_affine_time,
                ),
                ResBlockWithAttention(
                    in_channel,
                    in_channel,
                    time_dim,
                    dropout=dropout,
                    use_affine_time=use_affine_time,
                ),
            ]
        )

        # upsample
        up_layers = []
        for i in reversed(range(n_block)):
            for _ in range(n_res_blocks + 1):
                # present channel's number
                channel_mult = channel * channel_multiplier[i]

                # resblock
                up_layers.append(
                    ResBlockWithAttention(
                        in_channel + feat_channels.pop(),
                        channel_mult,
                        time_dim,
                        dropout=dropout,
                        use_attention=2 ** i in attn_strides,
                        attention_head=attn_heads,
                        use_affine_time=use_affine_time,
                    )
                )

                in_channel = channel_mult

            # ignore upsample in the last step
            if i != 0:
                up_layers.append(Upsample(in_channel))

        # into module list
        self.up = nn.ModuleList(up_layers)

        # output convolution
        self.out = nn.Sequential(
            nn.GroupNorm(32, in_channel),
            Swish(),
            conv2d(in_channel, 3 * (fold ** 2), 3, padding=1, scale=1e-10),
        )

    # forward method
    def forward(self, input, time):
        # time embedding
        time_embed = self.time(time)

        feats = []

        # downsample
        out = spatial_fold(input, self.fold)
        for layer in self.down:
            # resblock
            if isinstance(layer, ResBlockWithAttention):
                out = layer(out, time_embed)

            else:
                out = layer(out)

            feats.append(out)

        # middle
        for layer in self.mid:
            out = layer(out, time_embed)
        
        # upsample
        for layer in self.up:
            # resblock
            if isinstance(layer, ResBlockWithAttention):
                out = layer(torch.cat((out, feats.pop()), 1), time_embed)

            else:
                out = layer(out)

        out = self.out(out)
        out = spatial_unfold(out, self.fold)

        return out
