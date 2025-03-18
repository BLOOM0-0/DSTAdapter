from collections import OrderedDict
from typing import Tuple, Union
from timm.models.layers import DropPath, to_2tuple
from mmengine.model.weight_init import trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip
from mmengine.logging import MMLogger
from einops import rearrange
from mmaction.registry import MODELS
import math
from ..common import CAFModule
# from ..common import LoRand


class Adapter(nn.Module):
    def __init__(self, in_channels, adapter_channels, kernel_size):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding='same',   # tuple(x // 2 for x in kernel_size)
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

        # std = 0.01
        # nn.init.trunc_normal_(
        #     self.fc1.weight, std=std, a=-2 * std, b=2 * std
        # )
        #
        # nn.init.zeros_(self.fc1.bias)
        # nn.init.trunc_normal_(
        #     self.fc2.weight, std=std, a=-2 * std, b=2 * std
        # )
        # nn.init.zeros_(self.fc2.bias)


        self.caf = CAFModule(channels=in_channels, reduction=16)


    def forward(self, x, T):


        L, BT, C = x.size()     # 197*64*768
        B = BT // T
        Ca = self.conv.in_channels
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        x_id = x.clone()
        x = x[1:, :, :]
        x = self.fc1(x)     # 196, 64, 384

        x = x.view(H, W, B, T, Ca).permute(2, 4, 3, 0, 1).contiguous()  # 14,14,8,8,384 -> 8 B， 384 C， 8 T， 14 H， 14 W

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled
        x = self.conv(x)
        torch.backends.cudnn.enabled = cudnn_enabled


        x = x.permute(3, 4, 0, 2, 1).contiguous().view(L - 1, BT ,Ca)   # B,C,T,H,W -> H W B T C


        x = self.fc2(x)

        x = rearrange(x,'(h w) (b t) c -> b c t h w', h=H, w=W, t=T)
        x = self.caf(x)
        x = rearrange(x, 'b c t h w -> (h w) (b t) c', h=H, w=W, t=T)

        x_id[1:, :, :] += x
        return x_id


#
# class STAdapter(nn.Module):
#     def __init__(self, in_channels, adapter_channels):
#         super().__init__()
#         self.fc1 = nn.Linear(in_channels, adapter_channels)
#         self.conv1 = nn.Conv3d(
#             adapter_channels, adapter_channels,
#             kernel_size=(3, 1, 1),
#             stride=(1, 1, 1),
#             padding='same',   # tuple(x // 2 for x in kernel_size)
#             groups=adapter_channels,
#         )
#         self.conv2 = nn.Conv3d(
#             adapter_channels, adapter_channels,
#             kernel_size=(1, 3, 3),
#             stride=(1, 1, 1),
#             padding='same',  # tuple(x // 2 for x in kernel_size)
#             groups=adapter_channels,
#         )
#         self.fc2 = nn.Linear(adapter_channels, in_channels)
#         nn.init.constant_(self.conv1.weight, 0.)
#         nn.init.constant_(self.conv1.bias, 0.)
#         nn.init.constant_(self.conv2.weight, 0.)
#         nn.init.constant_(self.conv2.bias, 0.)
#         nn.init.constant_(self.fc1.bias, 0.)
#         nn.init.constant_(self.fc2.bias, 0.)
#
#         #self.bn = nn.BatchNorm3d(adapter_channels)
#         # self.act = nn.GELU()
#         # self.a = nn.Parameter(torch.tensor(1.0))
#         # self.b = nn.Parameter(torch.tensor(1.0))
#
#
#     def forward(self, x, T):
#
#         # x = self.norm(x)
#         # x = self.norm(x) * self.gamma + self.gammax * x
#
#         L, BT, C = x.size()     # 197*64*768
#         B = BT // T
#         Ca = self.conv1.in_channels
#         H = W = round(math.sqrt(L - 1))
#         assert L - 1 == H * W
#         x_id = x.clone()
#         x = x[1:, :, :]
#         x = self.fc1(x)     # 196, 64, 384
#         x = x.view(H, W, B, T, Ca).permute(2, 4, 3, 0, 1).contiguous()  # 14,14,8,8,384 -> 8 B， 384 C， 8 T， 14 H， 14 W
#
#         cudnn_enabled = torch.backends.cudnn.enabled
#         torch.backends.cudnn.enabled = cudnn_enabled
#         xt = self.conv1(x)
#
#         # x = self.act(x)  # self.bn(x)
#
#         xs = self.conv2(x)
#
#         # x = self.a * xt + self.b * xs
#         # x = xt + xs
#
#         torch.backends.cudnn.enabled = cudnn_enabled
#
#
#         x = x.permute(3, 4, 0, 2, 1).contiguous().view(L - 1, BT ,Ca)   # B,C,T,H,W -> H W B T C
#
#
#         x = self.fc2(x)
#         x_id[1:, :, :] += x
#         return x_id
#

#
# class T_Adapter(nn.Module):
#     def __init__(self, in_channels, adapter_channels):
#         super().__init__()
#         self.fc1 = nn.Linear(in_channels, adapter_channels)
#         self.conv = nn.Conv3d(
#             adapter_channels, adapter_channels,
#             kernel_size=(3, 1, 1),
#             stride=(1, 1, 1),
#             padding='same',
#             groups=adapter_channels
#         )
#         self.fc2 = nn.Linear(adapter_channels, in_channels)
#         nn.init.constant_(self.conv.weight, 0.)
#         nn.init.constant_(self.conv.bias, 0.)
#         nn.init.constant_(self.fc1.bias, 0.)
#         nn.init.constant_(self.fc2.bias, 0.)
#
#
#
#     def forward(self, x, b):
#
#         T, BL, C = x.size()
#         L = BL // b
#         H = W = round(math.sqrt(BL // b))
#         x_id = x.clone()
#
#         Ca = self.conv.in_channels
#
#         x = self.fc1(x)     # 8, 8*196, 384
#
#         x = rearrange(x, 't (b l) c -> t b l c', b=b)
#         x = x[:, :, 1:, :]
#         x_class = x[:, :, :1, :]
#         x = rearrange(x, 't b (h w) c -> b c t h w', t=T,b=b,c=Ca, h=H, w=W)
#         # x = rearrange(x, 'b c t (h w) -> b c t h w', h=H, w=W)
#
#         cudnn_enabled = torch.backends.cudnn.enabled
#         torch.backends.cudnn.enabled = cudnn_enabled
#         x = self.conv(x)
#         torch.backends.cudnn.enabled = cudnn_enabled
#
#
#         x = rearrange(x, 'b c t h w -> t b (h w) c', h=H, w=W, b=b, c=Ca, t=T)
#         x = torch.cat((x_class, x), dim=2)
#         x = rearrange(x, 't b l c -> t (b l) c')
#         # x = rearrange(x, 'b c t -> t b c')
#
#         x = self.fc2(x)
#
#         x_id += x
#
#         return x_id


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_tadapter=1,
                 num_frames=8, drop_path=0.):
        super().__init__()
        self.num_tadapter = num_tadapter
        self.attn = nn.MultiheadAttention(d_model, n_head)
        # self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head

        # self.ln_a = LayerNorm(d_model)

        self.t_adapter = Adapter(d_model, d_model // 16, (3, 1, 1))    # LoRand(d_model, d_model // 2)  (3, 1, 1)
        self.s_adapter = Adapter(d_model, d_model // 16, (1, 1, 1))     # (1, 3, 3)


        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        ## x shape [HW+1, BT, D]
        n, bt, d = x.shape
        ## temporal adaptation

        x = self.t_adapter(x, self.num_frames)

        # x = self.s_adapter(x, self.num_frames)
        # xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_frames, n=n)
        # xt = self.ln_1(xt)
        #
        # xt = self.T_Adapter(self.attention(xt))
        # xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n, t=self.num_frames)
        # x = x + self.drop_path(xt)

        ## spatial adaptation
        xs = self.ln_1(x)
        xs = self.attention(xs)
        xs = self.s_adapter(xs, self.num_frames)
        x = x + xs

        ## joint adaptation
        # x = self.adapter_pre_mlp(x, self.num_frames)

        xn = self.ln_2(x)
        x = x + self.mlp(xn)    # [197, 64, 768]
        return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_tadapter=1,
                 scale=1., drop_path=0.1):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, scale, num_tadapter, num_frames, dpr[i]) for i in
              range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@MODELS.register_module()
class DST_Adapter(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int,
                 drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

        self.transformer = Transformer(num_frames, width, layers, heads, num_tadapter=num_tadapter, scale=adapter_scale,
                                       drop_path=drop_path_rate)

        self.ln_post = LayerNorm(width)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            ## Load OpenAI CLIP pretrained weights
            if self.layers == 12:
                clip_model, preprocess = clip.load("ViT-B/16", device="cpu")
            else:
                clip_model, preprocess = clip.load("ViT-L/14", device="cpu")
            pretrain_dict = clip_model.visual.state_dict()

            del clip_model
            del pretrain_dict['proj']
            msg = self.load_state_dict(pretrain_dict, strict=False)
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        # ## initialize S_Adapter
        # for n, m in self.transformer.named_modules():
        #     if 'S_Adapter' in n:
        #         for n2, m2 in m.named_modules():
        #             if 'D_fc2' in n2:
        #                 if isinstance(m2, nn.Linear):
        #                     nn.init.constant_(m2.weight, 0)
        #                     nn.init.constant_(m2.bias, 0)
        #
        # ## initialize T_Adapter
        # for n, m in self.transformer.named_modules():
        #     if 'T_Adapter' in n:
        #         for n2, m2 in m.named_modules():
        #             if 'D_fc2' in n2:
        #                 if isinstance(m2, nn.Linear):
        #                     nn.init.constant_(m2.weight, 0)
        #                     nn.init.constant_(m2.bias, 0)
        #
        # ## initialize MLP_Adapter
        # for n, m in self.transformer.named_modules():
        #     if 'MLP_Adapter' in n:
        #         for n2, m2 in m.named_modules():
        #             if 'D_fc2' in n2:
        #                 if isinstance(m2, nn.Linear):
        #                     nn.init.constant_(m2.weight, 0)
        #                     nn.init.constant_(m2.bias, 0)

        ## initialize Adapter
        for n, m in self.transformer.named_modules():
            if 'Adapter' in n or 'adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        #######  freeze some parameters
        for name, param in self.named_parameters():
            if 'temporal_embedding' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name\
                    and 'adapter' not in name:
                param.requires_grad = False

        for name, param in self.named_parameters():
            logger.info('{}: {}'.format(name, param.requires_grad))
        num_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.parameters())
        logger.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'temporal_embedding'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.shape

        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = x[:, 0]
        # x = x[:, 1:, :]
        # x = rearrange(x, '(b t) (h w) d -> b d t h w', h=14,t=T)

        x = rearrange(x, '(b t) d -> b d t', b=B, t=T)

        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        # x = x[:, 1:]
        # x = torch.mean(x, dim=1)
        # x = rearrange(x, '(b t) d -> b d t', b=B, t=T)
        # x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        return x
