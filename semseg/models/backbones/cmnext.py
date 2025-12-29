import torch
from torch import nn, Tensor
from torch.nn import functional as F
from semseg.models.layers import DropPath
import functools
from functools import partial
from fvcore.nn import flop_count_table, FlopCountAnalysis
from semseg.models.modules.ffm import FeatureFusionModule as FFM
from semseg.models.modules.ffm import FeatureRectifyModule as FRM
from semseg.models.modules.ffm import ChannelEmbed
from semseg.models.modules.mspa import MSPABlock
from semseg.utils.utils import nchw_to_nlc, nlc_to_nchw
from timm.models.layers import to_2tuple, trunc_normal_


class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W, mask=None, return_attention=False, fuse=None) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if fuse:
            if self.sr_ratio > 1:
                x_rgb = x[:, :int(N / 2), :]
                x_dte = x[:, int(N / 2):, :]
                x_rgb = x_rgb.permute(0, 2, 1).reshape(B, C, H, W)
                x_rgb = self.sr(x_rgb).reshape(B, C, -1).permute(0, 2, 1)
                x_rgb = self.norm(x_rgb)
                x_dte = x_dte.permute(0, 2, 1).reshape(B, C, H, W)
                x_dte = self.sr(x_dte).reshape(B, C, -1).permute(0, 2, 1)
                x_dte = self.norm(x_dte)
                x = torch.cat([x_rgb, x_dte], dim = 1)

                mask = torch.zeros(N, int(N / self.sr_ratio / self.sr_ratio)).to(x_rgb)
                mask[:int(N / 2), :int(N / self.sr_ratio / self.sr_ratio / 2)] = 1
                mask[int(N / 2):, int(N / self.sr_ratio / self.sr_ratio / 2):] = 1
                mask = mask == 1

        else:
            if self.sr_ratio > 1:
                x = x.permute(0, 2, 1).reshape(B, C, H, W)
                x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
                x = self.norm(x)

        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(1), float('-inf'),)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if return_attention:
            return x, attn
        else:
            return x

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: Tensor, H, W, fuse=False) -> Tensor:
        if fuse:
            x = self.fc1(x)
            x_rgb = F.gelu(self.dwconv(x[:, :int(H * W), :], H, W))
            x_dte = F.gelu(self.dwconv(x[:,int(H*W):,:], H, W))
            x = self.fc2(torch.cat([x_rgb, x_dte], dim=1))
            return x
        else:
            return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, padding)    # padding=(ps[0]//2, ps[1]//2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class PatchEmbedParallel(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0, num_modals=4):
        super().__init__()
        self.proj = ModuleParallel(nn.Conv2d(c1, c2, patch_size, stride, padding))    # padding=(ps[0]//2, ps[1]//2)
        self.norm = LayerNormParallel(c2, num_modals)

    def forward(self, x: list) -> list:
        x = self.proj(x)
        _, _, H, W = x[0].shape
        x = self.norm(x)
        return x, H, W

import math
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.ReLU, skip_connect=True, prompt_add=False):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.D_fc1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.D_fc2.weight)
            nn.init.zeros_(self.D_fc1.bias)
            nn.init.zeros_(self.D_fc2.bias)
        self.prompt_add = prompt_add
        if prompt_add:
            self.D_fc_prompt = nn.Linear(D_features, D_hidden_features)

    def forward(self, x, prompt=None):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        if self.prompt_add and prompt is not None:
            prompts = self.D_fc_prompt(prompt)
            xs = xs + prompts
        xs = self.act(xs)
        xs = nn.functional.dropout(xs, p=0.1, training=self.training)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x



class CEBlock(nn.Module):

    def __init__(self, dim, head, sr_ratio=1, dpr=0., scale=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4))

        self.MLP_RGB_Adapter = Adapter(dim, skip_connect=False)
        self.MLP_DTE_Adapter = Adapter(dim, skip_connect=False)
        self.scale = scale
        self.adapter_dict = {
            'rgb': {'mlp': self.MLP_RGB_Adapter},
            'dte': {'mlp': self.MLP_DTE_Adapter},
        }

    def forward(self, x, H, W, mask=None, sub_mode=None):
        x_attn, attn = self.attn(self.norm1(x), H, W, mask, True)

        #if sub_mode != 'rgb':
        #x_attn = self.adapter_dict[sub_mode]['space'](x_attn)

        x = x + self.drop_path(x_attn)

        xn = self.norm2(x)
        #if sub_mode != 'rgb':
        x = x + self.drop_path(self.mlp(xn, H, W) + self.scale * self.adapter_dict[sub_mode]['mlp'](x))
        # else:
        #     x = x + self.drop_path(self.mlp(xn, H, W))

        return x

class MPGBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #self.patch_embed = PatchEmbed(dim, dim, 3, 1, 1)
        self.D_fc1 = nn.Linear(dim, int(dim * 0.25))
        self.D_fc2 = nn.Linear(dim, int(dim * 0.25))
        self.U_fc1 = nn.Linear(int(dim * 0.25), dim)
        self.act = nn.GELU()

    def forward(self, x_rgb: Tensor, x_dte: Tensor, H, W) -> Tensor:
        # B, N ,C = x_dte.shape
        # x_dte = x_dte.permute(0, 2, 1).reshape(B, C, H, W)
        # x_dte = self.patch_embed(x_dte)
        # x_dte = x_dte.reshape(B, C, -1).permute(0, 2, 1)

        x_rgb = self.D_fc1(x_rgb)
        x_dte = self.D_fc2(x_dte)
        x = x_dte + x_rgb
        x = self.U_fc1(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0., is_fan=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4)) if not is_fan else ChannelProcessing(dim, mlp_hidden_dim=int(dim*4))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class ChannelProcessing(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., drop_path=0., mlp_hidden_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_v = MLP(dim, mlp_hidden_dim)
        self.norm_v = norm_layer(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, H, W, atten=None):
        B, N, C = x.shape

        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads,  C // self.num_heads).permute(0, 2, 1, 3)

        q = q.softmax(-2).transpose(-1,-2)
        _, _, Nk, Ck  = k.shape
        k = k.softmax(-2)
        k = torch.nn.functional.avg_pool2d(k, (1, Ck))

        attn = self.sigmoid(q @ k)

        Bv, Hd, Nv, Cv = v.shape
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd*Cv), H, W)).reshape(Bv, Nv, Hd, Cv).transpose(1, 2)
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x


class PredictorConv(nn.Module):
    def __init__(self, embed_dim=384, num_modals=4):
        super().__init__()
        self.num_modals = num_modals
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=(embed_dim)),
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid()
        )for _ in range(num_modals)])

    def forward(self, x):
        B, C, H, W = x[0].shape
        x_ = [torch.zeros((B, 1, H, W)) for _ in range(self.num_modals)]
        for i in range(self.num_modals):
            x_[i] = self.score_nets[i](x[i])
        return x_

class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]

class ConvLayerNorm(nn.Module):
    """Channel first layer norm
    """
    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class LayerNormParallel(nn.Module):
    def __init__(self, num_features, num_modals=4):
        super(LayerNormParallel, self).__init__()
        # self.num_modals = num_modals
        for i in range(num_modals):
            setattr(self, 'ln_' + str(i), ConvLayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'ln_' + str(i))(x) for i, x in enumerate(x_parallel)]

import einops
from timm.models.layers import DropPath
class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')



class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class DAttentionMM(nn.Module):

    def __init__(
            self, dims, q_size=(56, 56), kv_size=56, n_heads=2, n_groups=1,
            attn_drop=0, proj_drop=0, stride=8,
            offset_range_factor=-1, use_pe=True, dwc_pe=False,
            no_off=False, fixed_pe=False, ksize=9, log_cpb=False, dpr=0
    ):
    # k_size(9,7,5,3) stride(8,4,2,1) n_groups=(1,2,4,8) n_heads=(2,4,8,16)
        super().__init__()
        self.fp16_enabled = False
        self.drop_path = DropPath(dpr)
        n_head_channels = dims // n_heads
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.stride = stride
        self.log_cpb = log_cpb
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset_x = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        self.conv_offset_y = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        # self.proj_q = nn.Conv2d(
        #     self.nc, self.nc,
        #     kernel_size=1, stride=1, padding=0
        # )

        self.fuse_q = conv_bn_relu(int(dims * 2), dims)

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.get_sample_weight = nn.Sequential(
            nn.Conv2d(
                dims, dims,
                kernel_size=1, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                dims, 2,
                kernel_size=1, stride=1, padding=0
            ),
        )
        self.softmax = nn.Softmax(dim=1)


        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and (not self.no_off):
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                # Borrowed from Swin-V2
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None


    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        with torch.no_grad():
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
                torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
                #indexing='ij'
            )
            ref = torch.stack((ref_y, ref_x), -1)
            ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
            ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
            ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

            return ref

    def _get_q_grid(self, H, W, B, dtype, device):
        with torch.no_grad():
            ref_y, ref_x = torch.meshgrid(
                torch.arange(0, H, dtype=dtype, device=device),
                torch.arange(0, W, dtype=dtype, device=device),
                # indexing='ij'
            )
            ref = torch.stack((ref_y, ref_x), -1)
            ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
            ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
            ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

            return ref

    def forward(self, x, y):
        # x-rgb y-dte
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        xy = torch.cat([x, y], dim=1)

        xy = self.fuse_q(xy)
        q = self.proj_q(xy)
        x_off = einops.rearrange(x, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        y_off = einops.rearrange(y, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)

        x_offset = self.conv_offset_x(x_off).contiguous()  # B * g 2 Hg Wg
        y_offset = self.conv_offset_y(y_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = x_offset.size(2), x_offset.size(3)

        n_sample = Hk * Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        x_offset = einops.rearrange(x_offset, 'b p h w -> b h w p')
        y_offset = einops.rearrange(y_offset, 'b p h w -> b h w p')

        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos_x = (x_offset + reference).clamp(-1., +1.)
            pos_y = (y_offset + reference).clamp(-1., +1.)
            # pos_x = (x_offset + reference).tanh()
            # pos_y = (y_offset + reference).tanh()

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            Hk, Wk = x_sampled.size(2), x_sampled.size(3)
            n_sample = Hk * Wk
        else:
            x_sampled_x = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos_x[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
            x_sampled_y = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos_y[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
            y_sampled_x = F.grid_sample(
                input=y.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos_x[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
            y_sampled_y = F.grid_sample(
                input=y.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos_y[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
            q_sampled_x = F.grid_sample(
                input=q.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos_x[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
            q_sampled_y = F.grid_sample(
                input=q.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos_y[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled_x = x_sampled_x.reshape(B, C, 1, n_sample)
        x_sampled_y = x_sampled_y.reshape(B, C, 1, n_sample)
        x_sampled = torch.cat([x_sampled_x, x_sampled_y], dim=-1)
        y_sampled_x = y_sampled_x.reshape(B, C, 1, n_sample)
        y_sampled_y = y_sampled_y.reshape(B, C, 1, n_sample)
        y_sampled = torch.cat([y_sampled_x, y_sampled_y], dim=-1)
        q_sampled_x = q_sampled_x.reshape(B, C, 1, n_sample)
        q_sampled_y = q_sampled_y.reshape(B, C, 1, n_sample)
        q_sampled = torch.cat([q_sampled_x, q_sampled_y], dim=-1)

        sample_weight = self.get_sample_weight(q_sampled)
        sample_weight = self.softmax(sample_weight).squeeze(2).unsqueeze(1)
        sampled = torch.sum(sample_weight * torch.cat([x_sampled, y_sampled], dim=-2), dim=-2, keepdim=True)
        
        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample * 2)
        v = self.proj_v(sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample * 2)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe and (not self.no_off):

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels,
                                                                              H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn_bias = F.interpolate(attn_bias, size=(H * W, n_sample), mode='bilinear', align_corners=True)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    4.0)  # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement)  # B * g, H * W, n_sample, h_g
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)

                q_grid = self._get_q_grid(H, W, B, dtype, device)

                displacement_x = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos_x.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    0.5)

                displacement_y = (
                        q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos_y.reshape(B * self.n_groups,
                                                                                                 n_sample,
                                                                                                 2).unsqueeze(1)).mul(
                    0.5)

                attn_bias_x = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads,
                                           g=self.n_groups),
                    grid=displacement_x[..., (1, 0)],
                    mode='bilinear', align_corners=True
                )  # B * g, h_g, HW, Ns

                attn_bias_y = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads,
                                           g=self.n_groups),
                    grid=displacement_y[..., (1, 0)],
                    mode='bilinear', align_corners=True
                )  # B * g, h_g, HW, Ns
                attn_bias = torch.cat([attn_bias_x, attn_bias_y], dim=-1)
                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample * 2)

                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        out = self.proj_drop(self.proj_out(out))

        out = out + xy
        return out


class DeformMPGBlock(nn.Module):
    def __init__(self, dims, stride, n_groups, n_heads, dpr):
        super().__init__()
        #self.patch_embed = PatchEmbed(dim, dim, 3, 1, 1)
        self.D_fc1 = nn.Linear(dims, int(dims * 0.25))
        self.D_fc2 = nn.Linear(dims, int(dims * 0.25))
        self.U_fc1 = nn.Linear(int(dims * 0.25), dims)
        self.act = nn.GELU()
        self.deform_atten = DAttentionMM(dims = int(dims * 0.25), stride = stride, n_groups = n_groups, n_heads = n_heads, dpr=dpr)


    def forward(self, x_rgb: Tensor, x_dte: Tensor, H, W) -> Tensor:
        # B, N ,C = x_dte.shape
        # x_dte = x_dte.permute(0, 2, 1).reshape(B, C, H, W)
        # x_dte = self.patch_embed(x_dte)
        # x_dte = x_dte.reshape(B, C, -1).permute(0, 2, 1)
        B, N ,C = x_rgb.shape
        x_rgb = self.D_fc1(x_rgb)
        x_dte = self.D_fc2(x_dte)
        B, N, new_C = x_rgb.shape
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x_dte = x_dte.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x_fuse = self.deform_atten(x_rgb, x_dte)
        x_fuse = x_fuse.reshape(B, new_C, -1).permute(0, 2, 1)
        x_fuse = self.U_fc1(x_fuse)
        return x_fuse


cmnext_settings = {
    # 'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],
    # 'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    # 'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}


class CMNeXt(nn.Module):
    def __init__(self, model_name: str = 'B0', modals: list = ['rgb', 'depth', 'event', 'lidar']):
        super().__init__()
        assert model_name in cmnext_settings.keys(), f"Model name should be in {list(cmnext_settings.keys())}"
        embed_dims, depths = cmnext_settings[model_name]
        extra_depths = depths
        self.modals = modals[1:] if len(modals)>1 else []
        self.num_modals = len(self.modals)
        drop_path_rate = 0.1
        self.channels = embed_dims
        norm_cfg = dict(type='BN', requires_grad=True)

        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4, 7//2)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 3//2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 3//2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 3//2)
        self.extra_patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4, 7 // 2)
        self.extra_patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 3 // 2)
        self.extra_patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 3 // 2)
        self.extra_patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 3 // 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        k_size = [9, 7, 5, 3]
        stride = [8, 4, 2, 1]
        n_groups = [1, 2, 4, 8]
        n_heads = [2, 4, 8, 16]

        cur = 0
        self.MPGblock1 = MPGBlock(embed_dims[0])
        self.block1 = nn.ModuleList([CEBlock(embed_dims[0], 1, 8, dpr[cur+i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.extra_norm1 = nn.LayerNorm(embed_dims[0])
        self.DeformMPGBlock1 = DeformMPGBlock(dims=embed_dims[0], stride=stride[0],
                                              n_groups=n_groups[0], n_heads=n_heads[0], dpr=dpr[cmnext_settings[model_name][1][0]-1])

        cur += depths[0]
        self.MPGblock2 = MPGBlock(embed_dims[1])
        self.block2 = nn.ModuleList([CEBlock(embed_dims[1], 2, 4, dpr[cur+i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.extra_norm2 = nn.LayerNorm(embed_dims[1])
        self.DeformMPGBlock2 = DeformMPGBlock(dims=embed_dims[1], stride=stride[1],
                                              n_groups=n_groups[1], n_heads=n_heads[1], dpr=dpr[cmnext_settings[model_name][1][0] + cmnext_settings[model_name][1][1]-1])

        cur += depths[1]
        self.MPGblock3 = MPGBlock(embed_dims[2])
        self.block3 = nn.ModuleList([CEBlock(embed_dims[2], 5, 2, dpr[cur+i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.extra_norm3 = nn.LayerNorm(embed_dims[2])
        self.DeformMPGBlock3 = DeformMPGBlock(dims=embed_dims[2], stride=stride[2],
                                              n_groups=n_groups[2], n_heads=n_heads[2], dpr=dpr[cmnext_settings[model_name][1][0] + cmnext_settings[model_name][1][1]+ cmnext_settings[model_name][1][2]-1])

        cur += depths[2]
        self.MPGblock4 = MPGBlock(embed_dims[3])
        self.block4 = nn.ModuleList([CEBlock(embed_dims[3], 8, 1, dpr[cur+i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])
        self.extra_norm4 = nn.LayerNorm(embed_dims[3])
        self.DeformMPGBlock4 = DeformMPGBlock(dims=embed_dims[3], stride=stride[3],
                                              n_groups=n_groups[3], n_heads=n_heads[3], dpr=dpr[cmnext_settings[model_name][1][0] + cmnext_settings[model_name][1][1]+ cmnext_settings[model_name][1][2]+cmnext_settings[model_name][1][3]-1])

    def forward(self, x: list) -> list:
        x_rgb = x[0]
        if self.num_modals > 0:
            x_ext = x[1]
        B = x_rgb.shape[0]
        outs = []
        # stage 1
        x1_rgb, H, W = self.patch_embed1(x_rgb)
        x1_ext, _, _ = self.extra_patch_embed1(x_ext)
        x1_fuse = self.MPGblock1(x1_rgb, x1_ext, H, W)
        x1_rgb = x1_rgb + x1_fuse
        x1_ext = x1_ext + x1_fuse
        for i in range(len(self.block1)):
            x1_rgb = self.block1[i](x1_rgb, H, W, sub_mode='rgb')
            x1_ext = self.block1[i](x1_ext, H, W, sub_mode='dte')
        x1_rgb = self.norm1(x1_rgb)
        x1_ext = self.extra_norm1(x1_ext)
        x1_out = self.DeformMPGBlock1(x1_rgb, x1_ext, H, W).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x1_rgb = x1_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x1_ext = x1_ext.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x1_out)

        # stage 2
        x2_rgb, H, W = self.patch_embed2(x1_rgb)
        x2_ext, _, _ = self.extra_patch_embed2(x1_ext)
        x2_fuse = self.MPGblock2(x2_rgb, x2_ext, H, W)
        x2_rgb = x2_rgb + x2_fuse
        x2_ext = x2_ext + x2_fuse
        for i in range(len(self.block2)):
            x2_rgb = self.block2[i](x2_rgb, H, W, sub_mode='rgb')
            x2_ext = self.block2[i](x2_ext, H, W, sub_mode='dte')
        x2_rgb = self.norm2(x2_rgb)
        x2_ext = self.extra_norm2(x2_ext)
        x2_out = self.DeformMPGBlock2(x2_rgb, x2_ext, H, W).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x2_rgb = x2_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x2_ext = x2_ext.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x2_out)

        # stage 3
        x3_rgb, H, W = self.patch_embed3(x2_rgb)
        x3_ext, _, _ = self.extra_patch_embed3(x2_ext)
        x3_fuse = self.MPGblock3(x3_rgb, x3_ext, H, W)
        x3_rgb = x3_rgb + x3_fuse
        x3_ext = x3_ext + x3_fuse
        for i in range(len(self.block3)):
            x3_rgb = self.block3[i](x3_rgb, H, W, sub_mode='rgb')
            x3_ext = self.block3[i](x3_ext, H, W, sub_mode='dte')
        x3_rgb = self.norm3(x3_rgb)
        x3_ext = self.extra_norm3(x3_ext)
        x3_out = self.DeformMPGBlock3(x3_rgb, x3_ext, H, W).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x3_rgb = x3_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x3_ext = x3_ext.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x3_out)

        # stage 4
        x4_rgb, H, W = self.patch_embed4(x3_rgb)
        x4_ext, _, _ = self.extra_patch_embed4(x3_ext)
        x4_fuse = self.MPGblock4(x4_rgb, x4_ext, H, W)
        x4_rgb = x4_rgb + x4_fuse
        x4_ext = x4_ext + x4_fuse
        for i in range(len(self.block4)):
            x4_rgb = self.block4[i](x4_rgb, H, W, sub_mode='rgb')
            x4_ext = self.block4[i](x4_ext, H, W, sub_mode='dte')
        x4_rgb = self.norm4(x4_rgb)
        x4_ext = self.extra_norm4(x4_ext)
        x4_out = self.DeformMPGBlock4(x4_rgb, x4_ext, H, W).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # x4_rgb = x4_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # x4_ext = x4_ext.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x4_out)

        return outs


if __name__ == '__main__':
    modals = ['img', 'depth', 'event', 'lidar']
    x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024)*2, torch.ones(1, 3, 1024, 1024) *3]
    model = CMNeXt('B2', modals)
    outs = model(x)
    for y in outs:
        print(y.shape)

